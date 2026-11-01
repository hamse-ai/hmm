# Human Activity Recognition with a Hidden Markov Model

A small, reproducible HMM project using Sensor Logger accelerometer and gyroscope recordings for four activities: standing, walking, jumping, and still.

## Contents

- `data/`: 50 labeled recording folders per activity. Each recording retains its original sensor CSVs and metadata.
- `src/activity_hmm.py`: timestamp-aware feature extraction, z-score normalization, diagonal Gaussian HMM, Baum–Welch training, Viterbi decoding, window- and recording-level metrics, training convergence history, model parameters, and SVG figures.
- `notebooks/activity_hmm.ipynb`: walkthrough of data inspection, raw signals, features, model training, test evaluation, and plots.
- `reports/report.md`: concise report aligned to the formative rubric.
- `results/`: outputs generated from the current 40/10 recording-level split.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python src/activity_hmm.py data results
jupyter notebook notebooks/activity_hmm.ipynb
```

The analysis uses only `Accelerometer.csv` and `Gyroscope.csv` for model features. Original exports are preserved. In the notebook, the 2-second window has a 1-second hop. Recordings with numeric suffixes 0–39 are training data; suffixes 40–49 form the unseen, later-recorded test split for every class.

## Important context

All supplied recordings are from one Android handset and one participant. The holdout evaluates later recordings from that setup; it does not establish cross-person or cross-device generalization. The four collection videos are hosted externally and linked in the separately submitted report; video files are not stored in this repository.
