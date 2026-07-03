# data/

Point-cloud frames, grouped by scenario. The eight scenario folders are already
created for you. Drop each scenario's `.pcd` files into its matching folder,
keeping the original frame names (000000.pcd, 000001.pcd, ...).

```
data/
├── S1_empty_field/       <- put S1 .pcd files here
├── S2_idle_standing/
├── S3_walking/
├── S4_squatting_static/
├── S5_squatting_moving/
├── S6_overhead_motion/
├── S7_rhythmic_ground/
└── S8_mixed/
```

Each folder currently holds a `.gitkeep` placeholder so the empty folder is
kept. You can delete `.gitkeep` after adding your `.pcd` files, or just leave it.

## Labels are folder-based
No separate label file is needed. The scenario folder name is the activity
label for every frame inside it (see the mapping in ../README.md). `S8_mixed` is
the unscripted test sequence and contains several activities.

## Raw ROS bags (.db3)
The original `.db3` recordings are NOT included in this repository by choice.
They remain the private source of these processed `.pcd` frames.
