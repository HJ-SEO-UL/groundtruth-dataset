# HOMI — Metadata

## Sensor
- **LiDAR:** Single Unitree L2, fixed (not moving) during capture.
- **RGB camera:** None. No images or faces are recorded.
- **Output:** 3D point-cloud frames.

## Site
- **Location:** Citizen farm, Suwon, Republic of Korea.
- **Environment:** Unstructured outdoor farming plot with agricultural ridges.
- *For privacy, publish only a coarse location (city level). Do not publish
  precise GPS coordinates. See the checklist below.*

## Dataset size
- **Total:** 16,676 point-cloud frames.
- **Scenarios:** 8 (S1–S8). See the dataset README for the scenario table.
- **Activity classes:** 6.

## Method summary
- **Features:** 16 geometric and temporal features per frame
  (e.g. h_mean, upper_z_mean, upper_z, n, pos_total, h_std, pos_std,
  h_range, h, pos_mean, upper_z_range, d, upper_z_std, w, h/w, delta_pos).
- **Model:** Random Forest classifier.
- **Validation:** 5-fold cross-validation.
- **Ridge detection:** Six agricultural ridges detected from the static
  background scan (S1).
- **Metric validation:** A 100.0 cm reference object was measured at 99.9 cm,
  a 0.1% error (see results.md).

## Raw recordings (ROS bags)
Each scenario folder (S1_empty_field, … , S8_mixed) contains the raw recording
as ROS 2 bag files (for example S1_empty_field_0.db3) together with a
metadata.yaml. These are large. See the top-level README for how to publish
large files (Git LFS or an external archive).

## Privacy checklist before publishing
- [ ] Open each metadata.yaml and confirm it contains no precise GPS / personal data.
- [ ] Reduce site location to city level (Suwon) only.
- [ ] Confirm no RGB/image topics were recorded in the bags.
- [ ] Remove the screencast .webm files (not needed in the repository).
