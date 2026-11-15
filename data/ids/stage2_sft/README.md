# Stage-2 GT-SFT clip lists

These lists define the training pools and the validation set of the Stage-2 GT-condition SFT. They are frozen: the
released generator was trained on exactly these clips. Sample ids follow the cache file names (`<sample_id>.pt`).

| file | contents |
|---|---|
| `train/p_fall.txt`, `train/p_ramp.txt`, `train/p_spring.txt`, `train/p_multi.txt`, `train/p_liquid.txt` | Physics 101 clips per scenario. Ids are `p101_<scenario>_<object>_<setting>_<trial>__<view>`, with view `cam1` (Camera_1), `kin` (Kinect_RGB) or `top` (Camera_2, ramp and multi only). |
| `train/ssv2.txt` | Something-Something v2 clips from the official train split (`ssv2_<video id>`). |
| `train/phyco.txt` | PhyCo Kubric clips (`phyco_<scenario>_<render batch>_<id>`). |
| `val/ids_p101.txt`, `val/ids_phyco.txt`, `val/ids_ssv2.txt` | Held-out validation clips (validation flow-matching loss only). None of them is in a training list. |
| `exclusion_lists.json` | Evaluation and held-out ids that must never be trained on: Physics 101 dev / test / RL hold-out lists, the SSv2 hold-out and the official SSv2 validation and test ids, the PhyCo validation / test lists and the RAFT hold-out. A Physics 101 clip is excluded when any camera of the same trial (the id before `__`) is listed. `scripts/s2sft/v2/make_pools_v2.py` checks every training id against it. |
| `fallback_29frame/phyco_train.txt`, `fallback_29frame/phyco_val.txt` | PhyCo clips whose raw render was not available to the 41-frame build. The reference run used their 29-frame payloads (8 latent frames). |

Pool weights are in `configs/stage2/sft_overrides.json` (`mix`). Physics 101 has 50 % of the draws, split over the
five scenarios in proportion to (clip count)^0.75; SSv2 has 10 % and PhyCo 40 %. `make_pools_v2.py` recomputes these
weights from the list sizes.

The lists were produced by `scripts/s2sft/aux/inventory.py` (exclusions, per-source candidates),
`scripts/s2sft/select_phyco_new.py` (additional PhyCo clips, balanced over scenarios) and `scripts/s2sft/make_pools.py`
(the pools). These provenance scripts read intermediate files of the original data pipeline; they are not needed to
train.
