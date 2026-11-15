from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class NativeMotionTargets:
    flow: torch.Tensor
    visibility: torch.Tensor
    object_mask: torch.Tensor
    future_mask: torch.Tensor
    instance_id: torch.Tensor
    centers: torch.Tensor
    center_valid: torch.Tensor
    object_ids: torch.Tensor
    particle_objects: int
    reconstruction_coverage: float


def decode_image(dataset: h5py.Dataset) -> np.ndarray:
    """Decode a TDW image-pass byte array without relying on a sidecar video."""
    value = dataset[()]
    if isinstance(value, np.ndarray) and value.ndim >= 2:
        return np.asarray(value)
    return np.asarray(Image.open(io.BytesIO(np.asarray(value).tobytes())))


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return the active rotation matrix for TDW's [x, y, z, w] quaternion."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_vertices(
    vertices: np.ndarray,
    scale: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    local = np.asarray(vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    return local @ quaternion_matrix(quaternion).T + np.asarray(position, dtype=np.float64)


def project_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project TDW world points to zero-based pixel coordinates and positive camera depth."""
    points = np.asarray(points, dtype=np.float64)
    camera = np.asarray(camera_matrix, dtype=np.float64).reshape(4, 4)
    projection = np.asarray(projection_matrix, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    clip = (projection @ camera @ homogeneous.T).T
    normalized = clip[:, :2] / clip[:, 3:4]
    pixels = np.stack(
        (
            (normalized[:, 0] + 1) * ((width - 1) / 2),
            (1 - normalized[:, 1]) * ((height - 1) / 2),
        ),
        axis=1,
    )
    return pixels.astype(np.float32), clip[:, 3].astype(np.float32)


def resize_id_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image = np.asarray(image)[..., :3]
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image.astype(np.uint8, copy=False)


def _triangle_pixels(
    triangle: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    x0 = max(0, int(np.floor(triangle[:, 0].min())))
    x1 = min(width - 1, int(np.ceil(triangle[:, 0].max())))
    y0 = max(0, int(np.floor(triangle[:, 1].min())))
    y1 = min(height - 1, int(np.ceil(triangle[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return None
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    denominator = (
        (triangle[1, 1] - triangle[2, 1]) * (triangle[0, 0] - triangle[2, 0])
        + (triangle[2, 0] - triangle[1, 0]) * (triangle[0, 1] - triangle[2, 1])
    )
    if abs(float(denominator)) < 1e-8:
        return None
    first = (
        (triangle[1, 1] - triangle[2, 1]) * (xx - triangle[2, 0])
        + (triangle[2, 0] - triangle[1, 0]) * (yy - triangle[2, 1])
    ) / denominator
    second = (
        (triangle[2, 1] - triangle[0, 1]) * (xx - triangle[2, 0])
        + (triangle[0, 0] - triangle[2, 0]) * (yy - triangle[2, 1])
    ) / denominator
    third = 1 - first - second
    barycentric = np.stack((first, second, third), axis=-1).astype(np.float32)
    inside = np.all(barycentric >= -1e-4, axis=-1)
    if not inside.any():
        return None
    return yy[inside], xx[inside], barycentric[inside]


def rasterize_surface(
    vertices_xy: np.ndarray,
    vertices_depth: np.ndarray,
    faces: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize the nearest source face and its screen-space barycentric coordinates."""
    depth = np.full((height, width), np.inf, dtype=np.float32)
    face_map = np.full((height, width), -1, dtype=np.int32)
    barycentric_map = np.zeros((height, width, 3), dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    for face_index, indices in enumerate(faces):
        if np.any(indices < 0) or np.any(indices >= len(vertices_xy)):
            continue
        triangle_xy = vertices_xy[indices]
        triangle_depth = vertices_depth[indices]
        if not np.isfinite(triangle_xy).all() or not np.isfinite(triangle_depth).all():
            continue
        if np.any(triangle_depth <= 1e-6):
            continue
        pixels = _triangle_pixels(triangle_xy, height=height, width=width)
        if pixels is None:
            continue
        yy, xx, barycentric = pixels
        candidate = (barycentric * triangle_depth[None]).sum(axis=1)
        closer = candidate < depth[yy, xx]
        if not closer.any():
            continue
        yy, xx = yy[closer], xx[closer]
        depth[yy, xx] = candidate[closer]
        face_map[yy, xx] = face_index
        barycentric_map[yy, xx] = barycentric[closer]
    return face_map, barycentric_map, depth


def segmentation_mask(segmentation: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.all(segmentation[..., :3] == np.asarray(color, dtype=np.uint8), axis=-1)


def _frame_geometry(
    handle: h5py.File,
    frame_key: str,
    object_index: int,
    vertices: np.ndarray,
    scale: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = handle["frames"][frame_key]
    positions = np.asarray(frame["objects/positions"])
    rotations = np.asarray(frame["objects/rotations"])
    world = transform_vertices(vertices, scale, positions[object_index], rotations[object_index])
    return project_points(
        world,
        frame["camera_matrices/camera_matrix"][()],
        frame["camera_matrices/projection_matrix"][()],
        height=height,
        width=width,
    )


def _particle_geometry(
    handle: h5py.File,
    frame_key: str,
    object_id: int,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = handle["frames"][frame_key]
    particles = np.asarray(frame[f"particles/{object_id}"], dtype=np.float32)
    if particles.ndim != 2 or particles.shape[1] < 3:
        raise ValueError(
            f"invalid particles/{object_id} in frame {frame_key}: {particles.shape}"
        )
    pixels, depth = project_points(
        particles[:, :3],
        frame["camera_matrices/camera_matrix"][()],
        frame["camera_matrices/projection_matrix"][()],
        height=height,
        width=width,
    )
    return pixels, depth, particles[:, :3]


def _project_centers(
    handle: h5py.File,
    frame_key: str,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    frame = handle["frames"][frame_key]
    pixels, _ = project_points(
        np.asarray(frame["objects/center"]),
        frame["camera_matrices/camera_matrix"][()],
        frame["camera_matrices/projection_matrix"][()],
        height=height,
        width=width,
    )
    pixels[:, 0] /= width
    pixels[:, 1] /= height
    return pixels


def native_motion_targets(
    handle: h5py.File,
    frame_keys: Iterable[str],
    *,
    height: int,
    width: int,
    max_instances: int,
    minimum_pixels: int = 4,
    depth_tolerance: float = 0.02,
) -> NativeMotionTargets:
    """Build simulator-native, multi-object forward flow from TDW meshes and poses.

    `frame_keys` contains the source frame followed by every requested horizon.
    HDF5 instance colors are supervision only; callers must not feed them to the model.
    """
    frame_keys = list(frame_keys)
    if len(frame_keys) < 2:
        raise ValueError("at least one source and one future frame are required")
    object_ids = np.asarray(handle["static/object_ids"], dtype=np.int64)
    colors = np.asarray(handle["static/object_segmentation_colors"], dtype=np.uint8)
    scales = np.asarray(handle["static/scale"], dtype=np.float32)
    if not (len(object_ids) == len(colors) == len(scales)):
        raise ValueError("static object metadata lengths disagree")
    if len(object_ids) > max_instances:
        raise ValueError(f"{len(object_ids)} objects exceed max_instances={max_instances}")

    segmentations = [
        resize_id_image(decode_image(handle["frames"][key]["images/_id"]), height, width)
        for key in frame_keys
    ]
    horizons = len(frame_keys) - 1
    flow = np.zeros((horizons, 2, height, width), dtype=np.float32)
    visibility = np.zeros((horizons, 1, height, width), dtype=bool)
    object_mask = np.zeros((1, height, width), dtype=bool)
    instance_id = np.zeros((1, height, width), dtype=np.int16)
    future_mask = np.zeros((horizons, 1, height, width), dtype=bool)
    centers = np.zeros((max_instances, horizons + 1, 2), dtype=np.float32)
    center_valid = np.zeros((max_instances, horizons + 1), dtype=bool)
    padded_object_ids = np.full((max_instances,), -1, dtype=np.int64)
    source_instance_pixels = 0
    reconstructed_pixels = 0
    particle_objects = 0

    projected_centers = [
        _project_centers(handle, key, height=height, width=width) for key in frame_keys
    ]
    for future_index, segmentation in enumerate(segmentations[1:]):
        union = np.zeros((height, width), dtype=bool)
        for color in colors:
            union |= segmentation_mask(segmentation, color)
        future_mask[future_index, 0] = union

    for object_index, (object_id, color, scale) in enumerate(zip(object_ids, colors, scales)):
        source_segmentation = segmentation_mask(segmentations[0], color)
        source_instance_pixels += int(source_segmentation.sum())
        if source_segmentation.sum() < minimum_pixels:
            continue
        particle_key = str(int(object_id))
        use_particles = all(
            "particles" in handle["frames"][key]
            and particle_key in handle["frames"][key]["particles"]
            for key in frame_keys
        )
        particle_indices: np.ndarray | None = None
        particle_centers: list[np.ndarray] | None = None
        if use_particles:
            source_all_xy, source_all_depth, _ = _particle_geometry(
                handle,
                frame_keys[0],
                int(object_id),
                height=height,
                width=width,
            )
            usable = np.isfinite(source_all_xy).all(axis=1) & (source_all_depth > 0)
            rounded = np.round(source_all_xy[usable], decimals=4)
            _, unique_indices = np.unique(rounded, axis=0, return_index=True)
            particle_indices = np.where(usable)[0][np.sort(unique_indices)]
            if len(particle_indices) < 3:
                raise ValueError(
                    f"Flex object {int(object_id)} has only {len(particle_indices)} projectable particles"
                )
            source_xy = source_all_xy[particle_indices]
            source_depth = source_all_depth[particle_indices]
            faces = Delaunay(source_xy).simplices.astype(np.int64)
            particle_centers = []
            for key in frame_keys:
                _, _, particles_xyz = _particle_geometry(
                    handle,
                    key,
                    int(object_id),
                    height=height,
                    width=width,
                )
                frame = handle["frames"][key]
                center_xy, _ = project_points(
                    particles_xyz.mean(axis=0, keepdims=True),
                    frame["camera_matrices/camera_matrix"][()],
                    frame["camera_matrices/projection_matrix"][()],
                    height=height,
                    width=width,
                )
                center_xy[0, 0] /= width
                center_xy[0, 1] /= height
                particle_centers.append(center_xy[0])
            particle_objects += 1
            vertices = None
        else:
            vertices = np.asarray(
                handle[f"static/mesh/vertices_{object_index}"], dtype=np.float32
            )
            faces = np.asarray(
                handle[f"static/mesh/faces_{object_index}"], dtype=np.int64
            ).reshape(-1, 3)
            source_xy, source_depth = _frame_geometry(
                handle,
                frame_keys[0],
                object_index,
                vertices,
                scale,
                height=height,
                width=width,
            )
        face_map, barycentric_map, _ = rasterize_surface(
            source_xy, source_depth, faces, height=height, width=width
        )
        supported = source_segmentation & (face_map >= 0)
        if supported.sum() < minimum_pixels:
            continue
        slot = object_index
        padded_object_ids[slot] = int(object_id)
        object_mask[0, supported] = True
        instance_id[0, supported] = slot + 1
        reconstructed_pixels += int(supported.sum())
        if particle_centers is not None:
            centers[slot] = np.stack(particle_centers)
        else:
            centers[slot] = np.stack(
                [value[object_index] for value in projected_centers]
            )
        center_valid[slot] = np.isfinite(centers[slot]).all(axis=-1)

        source_y, source_x = np.where(supported)
        source_faces = face_map[source_y, source_x]
        source_barycentric = barycentric_map[source_y, source_x]
        vertex_indices = faces[source_faces]
        source_coordinates = np.stack((source_x, source_y), axis=1).astype(np.float32)
        for horizon, frame_key in enumerate(frame_keys[1:]):
            if use_particles:
                if particle_indices is None:
                    raise RuntimeError("missing selected particle indices")
                target_all_xy, target_all_depth, _ = _particle_geometry(
                    handle,
                    frame_key,
                    int(object_id),
                    height=height,
                    width=width,
                )
                if particle_indices[-1] >= len(target_all_xy):
                    raise ValueError(
                        f"Flex particle count changed for object {int(object_id)}"
                    )
                target_xy = target_all_xy[particle_indices]
                target_depth = target_all_depth[particle_indices]
            else:
                if vertices is None:
                    raise RuntimeError("missing rigid mesh vertices")
                target_xy, target_depth = _frame_geometry(
                    handle,
                    frame_key,
                    object_index,
                    vertices,
                    scale,
                    height=height,
                    width=width,
                )
            destination = (source_barycentric[..., None] * target_xy[vertex_indices]).sum(axis=1)
            destination_depth = (
                source_barycentric * target_depth[vertex_indices]
            ).sum(axis=1)
            displacement = destination - source_coordinates
            flow[horizon][:, source_y, source_x] = displacement.T

            target_depth_buffer = None
            if not use_particles:
                _, _, target_depth_buffer = rasterize_surface(
                    target_xy, target_depth, faces, height=height, width=width
                )
            destination_x = np.rint(destination[:, 0]).astype(np.int64)
            destination_y = np.rint(destination[:, 1]).astype(np.int64)
            in_bounds = (
                np.isfinite(destination).all(axis=1)
                & (destination_x >= 0)
                & (destination_x < width)
                & (destination_y >= 0)
                & (destination_y < height)
                & (destination_depth > 0)
            )
            visible = np.zeros(len(destination), dtype=bool)
            if in_bounds.any():
                selected = np.where(in_bounds)[0]
                target_matches = segmentation_mask(segmentations[horizon + 1], color)[
                    destination_y[selected], destination_x[selected]
                ]
                if target_depth_buffer is None:
                    self_visible = np.ones(len(selected), dtype=bool)
                else:
                    nearest_depth = target_depth_buffer[
                        destination_y[selected], destination_x[selected]
                    ]
                    self_visible = np.isfinite(nearest_depth) & (
                        destination_depth[selected]
                        <= nearest_depth + depth_tolerance * np.maximum(1.0, nearest_depth)
                    )
                visible[selected] = target_matches & self_visible
            visibility[horizon, 0, source_y, source_x] = visible

    coverage = reconstructed_pixels / max(source_instance_pixels, 1)
    if not object_mask.any():
        raise ValueError("no visible object pixels could be reconstructed from meshes or particles")
    return NativeMotionTargets(
        flow=torch.from_numpy(flow),
        visibility=torch.from_numpy(visibility),
        object_mask=torch.from_numpy(object_mask),
        future_mask=torch.from_numpy(future_mask),
        instance_id=torch.from_numpy(instance_id),
        centers=torch.from_numpy(centers),
        center_valid=torch.from_numpy(center_valid),
        object_ids=torch.from_numpy(padded_object_ids),
        particle_objects=particle_objects,
        reconstruction_coverage=float(coverage),
    )
