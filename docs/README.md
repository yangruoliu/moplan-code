# Documentation

| page | content |
|---|---|
| [INSTALL.md](INSTALL.md) | Python environment, SAM 2 and co-tracker, weight download and checksums, environment variables, checks |
| [INFERENCE.md](INFERENCE.md) | `generate.py`, the MoProbe and Physics-IQ protocols, intermediate files, the batch path, sources of non-bitwise differences |
| [TRAINING_STAGE1.md](TRAINING_STAGE1.md) | Stage-1 motion planner: model, losses, optimisation, training, resume, smoke test |
| [DATA_STAGE1.md](DATA_STAGE1.md) | Stage-1 training data: file contract, sources, build commands, what cannot be rebuilt exactly |
| [TRAINING_SFT.md](TRAINING_SFT.md) | Stage-2 GT-SFT: recipe, launcher, resume, smoke test |
| [DATA_STAGE2.md](DATA_STAGE2.md) | Stage-2 GT-SFT data: payload format, build commands, real-clip label chain, pools, gaps |
| [TRAINING_RAFT.md](TRAINING_RAFT.md) | the RAFT round: sampling, reward, selection, training, reproducibility checks |
| [EVALUATION.md](EVALUATION.md) | Physics-IQ, PisaBench, RigidBench, Morpheus and VBench-I2V with their official scorers |
| [MOPROBE.md](MOPROBE.md) | MoProbe: probe sets, measurement, metrics, bundle layout, commands |
| [eval_envs/](eval_envs/) | package lists of the scorer environments used for the reference evaluation |

Other entry points:

- [../README.md](../README.md): overview and quickstart.
- [../assets/showcase/README.md](../assets/showcase/README.md): the showcase cases and how to regenerate them.
- [../motion_jepa/README.md](../motion_jepa/README.md): the core library.
- [../data/ids/stage1/README.md](../data/ids/stage1/README.md), [../data/ids/stage2_sft/README.md](../data/ids/stage2_sft/README.md):
  the training id lists.
- [../tests/README.md](../tests/README.md): the CPU tests.
- [../LICENSE_NOTICE.md](../LICENSE_NOTICE.md): third-party licences, attributions and licence texts.
