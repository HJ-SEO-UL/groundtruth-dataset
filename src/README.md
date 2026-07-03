# src/

The code that produced the HOMI dataset and results. Each script maps to a
step in the pipeline.

| File | What it does |
|---|---|
| `background_model.py` | Builds the static background model from the empty field (S1). |
| `ridge_detection.py` | Detects the agricultural ridges from the background scan (6 ridges found). |
| `feature_extraction_v3.py` | Extracts the 16 geometric + temporal features from each point-cloud frame. |
| `label_timestamps.py` | Assigns an activity label to each frame by timestamp. |
| `rf_classifier_v3.py` | Trains and evaluates the Random Forest with 5-fold cross-validation (95.1%). |
| `test_s8.py` | Runs the trained model on the unscripted mixed sequence (S8). |
| `tank_validation.py` | Metric accuracy check against a 100 cm reference object (0.1% error). |
| `results_visualization.py` | Generates the figures in ../datasets/HOMI/results_figures/. |
| `homi_visualizer.py` | Interactive / visual inspection of point clouds and predictions. |

## How to run

Requires Python 3 and the following packages (adjust to what your scripts import):

```
pip install numpy scikit-learn open3d matplotlib pyyaml
```

Typical order:

```
python background_model.py
python ridge_detection.py
python feature_extraction_v3.py
python label_timestamps.py
python rf_classifier_v3.py
python test_s8.py
python results_visualization.py
```

Paths: update any hard-coded file paths inside the scripts to point at
`../datasets/HOMI/data/` before running.
