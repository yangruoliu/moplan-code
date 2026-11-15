# Stage-1 ID lists

One file per source and split (`<source>_{train,val}.tsv`). Together they list every training and validation row of
the reference Stage-1 run: 71,055 training rows over seven sources. Columns are `sample_id`, `scenario` and
`scene_id`. `scene_id` is empty where the manifest has none, and the scene is then the sample's source clip.

Rows are in manifest order. The source-cycle sampler indexes rows by position, so keep this order when you build
the manifests. `scripts/clean_r2/prepare_unified_stage1_data.py` does that. See `docs/DATA_STAGE1.md`.
