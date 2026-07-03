# models/

The trained classification model goes here.

You do not currently have a saved model file, and that is fine. The model is
reproducible: running `../src/rf_classifier_v3.py` trains the Random Forest and
reaches 95.1% (5-fold CV).

Two options:

1. **Save and commit the model (recommended).** At the end of training, save it:

   ```python
   import joblib
   joblib.dump(clf, "models/homi_random_forest.joblib")
   ```

   Then place `homi_random_forest.joblib` in this folder and note the library
   version below (for example: scikit-learn 1.4).

2. **Leave it empty.** Anyone can regenerate the model by running
   `src/rf_classifier_v3.py`. The README explains this, so an empty folder is
   acceptable.

Trained with: scikit-learn <fill in version>
