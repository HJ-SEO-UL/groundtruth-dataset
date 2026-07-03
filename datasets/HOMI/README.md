# HOMI

**A ground-level LiDAR dataset of agricultural worker activity.**
*소규모 농업 현장 작업자 행동 인식을 위한 LiDAR 데이터셋.*

HOMI is the first dataset released under the [GroundTruth](../../README.md) system. It captures the motion of farming work at ground level using a single fixed LiDAR sensor, with **no RGB camera**, so it records geometry and movement rather than identifiable images.

To our knowledge, HOMI is the first ground-level LiDAR dataset of agricultural worker activity in an unstructured outdoor farming environment. Existing LiDAR datasets (roads, buildings, indoor scenes) were never designed for a small farming plot.

---

## At a glance

| | |
|---|---|
| Frames | 16,676 point-cloud frames |
| Sensor | Single fixed Unitree L2 LiDAR (no RGB) |
| Site | Citizen farm, Suwon, Republic of Korea |
| Scenarios recorded | 8 (S1–S8) |
| Activity classes classified | 6 |
| Model | Random Forest, 16 geometric + temporal features |
| Accuracy | 95.1% (5-fold cross-validation) |
| Hand-tool (Ho-Mi) recall | 92% |
| Metric validation | 0.1% error on a 100 cm reference object |

Full results, with figures: [`results.md`](results.md)
Sensor and collection details: [`metadata.md`](metadata.md)

## What is in this folder

```
HOMI/
├── README.md            ← this file (dataset card)
├── LICENSE-DATA         ← CC BY 4.0 (the data license)
├── metadata.md          ← sensor, site, collection conditions
├── results.md           ← accuracy, confusion matrix, per-class recall
├── results_figures/     ← the result charts (PNG)
├── data/                ← point-cloud frames (see data/README.md)
└── labels/              ← activity labels per frame (see labels/README.md)
```

## Recording scenarios (S1–S8)

The data was recorded as eight scenarios. S1 is background (empty field), S8 is an unscripted mixed sequence used for testing, and the middle six correspond to the six activity classes used by the classifier.

| Folder | Scenario | Classifier class |
|---|---|---|
| S1_empty_field | Empty field (background, ridge detection) | (background) |
| S2_idle_standing | Idle standing | IdleStanding |
| S3_walking | Walking | Walking |
| S4_squatting_static | Static squatting | Squatting |
| S5_squatting_moving | Moving squatting | SquattingMoving |
| S6_overhead_motion | Watering (overhead arc) | Watering |
| S7_rhythmic_ground | Ho-Mi hand-tool use (rhythmic strike at ground) | GroundToolUse/(Ho-Mi) |
| S8_mixed | Unscripted mixed sequence | (test / all classes) |

*Please confirm this mapping against your own notes before publishing; adjust if any scenario maps differently.*

## Intended use

HOMI is intended as ground evidence for verifying climate-smart farm practices from labour motion, and as training data for privacy-preserving agricultural activity recognition. It contains no faces and no RGB imagery.

## License and citation

Data is licensed under [CC BY 4.0](LICENSE-DATA). Please cite:

> Seo, H. (2026). *HOMI: a ground-level LiDAR dataset of agricultural worker activity.* Part of the GroundTruth system. https://github.com/<owner>/groundtruth
