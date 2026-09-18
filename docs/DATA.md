# Training data

The release keeps the original `UnifiedDataset` in
`diffsynth/trainers/relit_dataset.py`. The example launch uses paired videos
(`--dataset_type 3`). Metadata can be CSV, JSON (a list of objects), or JSONL.
Each record needs `video` (a path relative to the RGB root) and `prompt`.

```text
data/
├── rgb/scene_000/{light_a,light_b,light_c}.mp4
├── shading_gt/scene_000/{light_a,light_b,light_c}.mp4
└── shading_cue/scene_000/{light_a,light_b,light_c}.mp4
```

`shading_gt` contains target ground-truth shading; `shading_cue` contains estimated
shading used during shading matching. They are not necessarily the same media.
For a metadata row `scene_000/light_a.mp4`, the loader resolves that same relative
path under each root. `examples/metadata*.csv` are schema examples only; no dataset
media are included. Replace these files with metadata for your own data.

For paired types 0, 1, 3 and 4, the conditioning video/image is sampled from another
lighting in the same scene directory. Keep at least three lighting files per
scene to support every RGB/shading conditioning branch. Inputs must have matching
frame counts and geometry; the loader excludes `Unlit.mp4`. With `--uniform_input`,
the scene must contain `all_white.mp4`.

With `--shading_cue_as_input True`, matching records come from
`--dataset_metadata_match_path`. The matching probability ramps from 0 to 0.5
between `--start_match_shading` and `--end_match_shading`. Keep the latter larger
than the former. The original loader also samples single frames and randomly
drops RGB conditioning during training.

For multiple datasets, pass equally sized JSON/Python lists to the five path
arguments and equally sized comma-separated lists to `--dataset_type`,
`--dataset_ratio` and `--num_frames`. Each ratio must be positive. For example,
`--dataset_type 3,3 --dataset_ratio 1,2 --num_frames 81,81` with two entries in each
path list.

The original type 6 layout uses `<scene>/input/deg.mp4` for conditioning and
`<scene>/shading.mp4` for ground-truth shading; it disables shading matching.
Other original dataset variants remain in the loader, but this first release's
example is the paired video path. Cached preprocessing is not part of the package.
