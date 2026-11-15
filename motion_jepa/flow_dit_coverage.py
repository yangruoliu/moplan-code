"""Deterministic source cycles and coverage of actually consumed minibatches."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import PurePosixPath

import torch

from .flow_dit_data import FixedMixtureSampler


class SourceCycleSampler(FixedMixtureSampler):
    """Keep fixed mixture quotas, visiting each source without replacement.

Per-source cycles continue across epoch boundaries. Shuffling source labels,
then consuming each source's ordered permutation, avoids mixing two cycles
before every row in the first has been visited. The stream is a pure function
of seed, epoch and populations; loader prefetch cannot advance saved state.
"""
    def source_counts(self):
        raw = [self.num_samples * fraction for fraction in self.fractions]
        counts = [int(value) for value in raw]
        order = sorted(range(len(counts)), key=lambda i: raw[i] - counts[i], reverse=True)
        for i in order[:self.num_samples - sum(counts)]:
            counts[i] += 1
        return counts

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        counts = self.source_counts()
        source_order = [i for i, count in enumerate(counts) for _ in range(count)]
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(self.num_samples, generator=generator).tolist()
        cursors = [epoch * count for count in counts]
        cycles, permutations = {}, {}
        indices = []
        for position in order:
            source = source_order[position]
            size = len(self.dataset.children[source])
            cycle, offset = divmod(cursors[source], size)
            if cycles.get(source) != cycle:
                # Stable integer hash avoids dependence on process hash seeds.
                key = f"source-cycle-v1:{self.seed}:{source}:{cycle}".encode()
                seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)
                permutations[source] = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
                cycles[source] = cycle
            indices.append(self.dataset._offsets[source] + permutations[source][offset])
            cursors[source] += 1
        return iter(indices)


def canonical_scene_id(row):
    """Use source identity before a possibly relocated training asset path."""
    get = row.get if isinstance(row, dict) else lambda key, default=None: getattr(row, key, default)
    # scene_id is preferred when supplied by an enriched manifest row.
    scene = get("scene_id")
    if scene:
        return "scene:" + str(scene)
    video = get("original_video_path") or get("video_path")
    if not isinstance(video, str) or not video:
        raise ValueError("coverage requires canonical scene_id or source video_path")
    return "video:" + str(PurePosixPath(video))


class ConsumedCoverage:
    """Count only batches the trainer reports after backward/optimizer handling.

All ranks receive the same gathered sample IDs at optimizer boundaries. A saved
state is therefore independent of worker prefetch and reproducible on resume.
"""
    def __init__(self, dataset, minimum=20000, *, minimum_by_source=None, source_names=None):
        self.minimum = int(minimum)
        self.source_names = list(source_names) if source_names is not None else [str(i) for i in range(len(dataset.children))]
        if (len(self.source_names) != len(dataset.children) or len(set(self.source_names)) != len(self.source_names)
                or any(not isinstance(name, str) or not name for name in self.source_names)):
            raise ValueError("coverage source_names must uniquely name every dataset child")
        self.minimum_by_source = dict(minimum_by_source or {})
        if (set(self.minimum_by_source) - set(self.source_names)
                or any(type(value) is not int or value < 0 for value in self.minimum_by_source.values())):
            raise ValueError("coverage source minima must name existing sources with nonnegative integer limits")
        self.population = {}
        for source, child in enumerate(dataset.children):
            for row in child.rows:
                sid = row["sample_id"] if isinstance(row, dict) else row.sample_id
                value = (source, canonical_scene_id(row))
                if sid in self.population and self.population[sid] != value:
                    raise ValueError(f"ambiguous sample ID across sources: {sid}")
                self.population[sid] = value
        self.population_digest = hashlib.sha256(json.dumps(self.population, sort_keys=True).encode()).hexdigest()
        self.seen_ids, self.seen_scenes = set(), set()
        self.updated_scenes_by_source = {name: set() for name in self.source_names}
        self.source_history_complete = True
        self.draws_by_source = Counter()
        self.total_draws = 0
        self.optimizer_updates = 0
        self.skipped_updates = 0
        eligible = len({value[1] for value in self.population.values()})
        if self.minimum > eligible:
            raise ValueError(f"minimum unique scenes {self.minimum} exceeds eligible population {eligible}")
        self.eligible_by_source = {
            name: {scene for source, scene in self.population.values() if source == index}
            for index, name in enumerate(self.source_names)
        }
        for name, minimum in self.minimum_by_source.items():
            if minimum > len(self.eligible_by_source[name]):
                raise ValueError(f"minimum unique scenes for {name} exceeds eligible population")

    def consume(self, sample_ids, *, optimizer_step_applied=True):
        # Validate the whole batch before mutating the ledger.
        unknown = set(sample_ids) - self.population.keys()
        if unknown:
            raise ValueError(f"consumed sample absent from manifest: {sorted(unknown)[:3]}")
        self.total_draws += len(sample_ids)
        self.optimizer_updates += int(optimizer_step_applied)
        self.skipped_updates += int(not optimizer_step_applied)
        for sid in sample_ids:
            source, canonical = self.population[sid]
            self.draws_by_source[source] += 1
            self.seen_ids.add(sid)
            if optimizer_step_applied:
                self.seen_scenes.add(canonical)
                self.updated_scenes_by_source[self.source_names[source]].add(canonical)

    @property
    def complete(self):
        return len(self.seen_scenes) >= self.minimum and all(
            len(self.updated_scenes_by_source[name]) >= minimum
            for name, minimum in self.minimum_by_source.items())

    def summary(self):
        return {"schema": "stage1_consumed_coverage_v1", "minimum_unique_scenes": self.minimum,
                "eligible_unique_scenes": len({v[1] for v in self.population.values()}),
                "unique_scenes_updated": len(self.seen_scenes), "unique_rows_consumed": len(self.seen_ids),
                "total_draws_consumed": self.total_draws,
                "source_draw_cursors": {str(k): v for k, v in sorted(self.draws_by_source.items())},
                "source_names": self.source_names,
                "minimum_unique_scenes_by_source": self.minimum_by_source,
                "source_unique_scenes_updated": {name: len(values) for name, values in self.updated_scenes_by_source.items()},
                "source_eligible_unique_scenes": {name: len(values) for name, values in self.eligible_by_source.items()},
                "source_coverage_history_complete": self.source_history_complete,
                "optimizer_updates": self.optimizer_updates, "skipped_updates": self.skipped_updates,
                "coverage_complete": self.complete}

    def state_dict(self):
        return {**self.summary(), "population_digest": self.population_digest,
                "seen_ids": sorted(self.seen_ids), "seen_scenes": sorted(self.seen_scenes),
                "updated_scenes_by_source": {name: sorted(values) for name, values in self.updated_scenes_by_source.items()}}

    def load_state_dict(self, state):
        if state.get("schema") != "stage1_consumed_coverage_v1" or state.get("population_digest") != self.population_digest:
            raise ValueError("coverage checkpoint population changed")
        if state.get("minimum_unique_scenes") != self.minimum:
            raise ValueError("coverage checkpoint minimum changed")
        if state.get("minimum_unique_scenes_by_source", {}) != self.minimum_by_source:
            raise ValueError("coverage checkpoint per-source minimum changed")
        if "source_names" in state and state["source_names"] != self.source_names:
            raise ValueError("coverage checkpoint source names changed")
        self.seen_ids = set(state["seen_ids"])
        self.seen_scenes = set(state["seen_scenes"])
        if (self.seen_ids - self.population.keys()
                or self.seen_scenes - {self.population[sid][1] for sid in self.seen_ids}):
            raise ValueError("coverage checkpoint contains unknown scenes/rows")
        self.total_draws = int(state["total_draws_consumed"])
        self.draws_by_source = Counter({int(k): int(v) for k, v in state["source_draw_cursors"].items()})
        self.optimizer_updates = int(state["optimizer_updates"])
        self.skipped_updates = int(state["skipped_updates"])
        if sum(self.draws_by_source.values()) != self.total_draws or self.total_draws < len(self.seen_ids):
            raise ValueError("coverage checkpoint consumed counts disagree")
        updated = state.get("updated_scenes_by_source")
        if updated is None:
            # Old states did not record which source got an applied update.
            # With skipped updates, seen_ids cannot safely reconstruct this.
            self.source_history_complete = self.skipped_updates == 0
            self.updated_scenes_by_source = {name: set() for name in self.source_names}
            if self.source_history_complete:
                for sid in self.seen_ids:
                    source, scene = self.population[sid]
                    self.updated_scenes_by_source[self.source_names[source]].add(scene)
        else:
            if not isinstance(updated, dict) or set(updated) != set(self.source_names):
                raise ValueError("coverage checkpoint per-source scene sets disagree")
            self.source_history_complete = state.get("source_coverage_history_complete", True)
            if type(self.source_history_complete) is not bool:
                raise ValueError("coverage checkpoint source history flag must be boolean")
            self.updated_scenes_by_source = {name: set(values) for name, values in updated.items()}
        for index, name in enumerate(self.source_names):
            consumed = {self.population[sid][1] for sid in self.seen_ids if self.population[sid][0] == index}
            if self.updated_scenes_by_source[name] - consumed:
                raise ValueError("coverage checkpoint credits unconsumed source scenes")
        union = set().union(*self.updated_scenes_by_source.values())
        if union - self.seen_scenes or (self.source_history_complete and union != self.seen_scenes):
            raise ValueError("coverage checkpoint source/global updated scenes disagree")
