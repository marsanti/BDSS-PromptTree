# Prompt Tree

## Dependencies
- pandas
- numpy
- scipy
- sklearn
- graphviz
- matplotlib

## Usage

The project can be run in different modes: Single Tree, Random Forest, Conformal Prediction, or Unsupervised.

### General Arguments

- `--dataset`: **(Required)** Name of the dataset to load.
  - Choices: `ECG200`, `ECG5000`, `ECGFiveDays`, `PenDigits`, `Heartbeat`
- `--mode`: Mode to run. Default: `single_tree`.
  - Choices: `single_tree`, `forest`, `conformal`, `unsupervised`
- `--val_size`: Proportion of data to use for validation set. Default: `0.2`.
  - *Note: Ignored in `conformal` and `unsupervised` modes.*
- `--seed`: Random seed for data splitting and forest building. Default: `42`.

### Mode-Specific Arguments

#### Single Tree (`--mode single_tree`)
- `--relaxed_stopping`: Use relaxed stopping criteria before pruning. (Flag)

#### Forest (`--mode forest`)
- `--n_estimators`: Number of trees in the forest. Default: `100`.
- `--voting`: Voting mechanism to use. Default: `weighted`.
  - Choices: `majority`, `weighted`, `track-record`

#### Conformal Prediction (`--mode conformal`)
- `--cal_ratio`: Proportion of training data for calibration (0 < ratio < 1). Default: `0.3`.
- `--epsilon`: Desired significance level (0 < epsilon < 1). Default: `0.1`.

### Examples

**Run Single Tree on ECG200:**
```bash
python src/main.py --dataset ECG200 --mode single_tree
```

**Run Random Forest on PenDigits with 20 trees:**
```bash
python src/main.py --dataset PenDigits --mode forest --n_estimators 20 --voting majority
```

**Run Conformal Prediction on Heartbeat with epsilon 0.05:**
```bash
python src/main.py --dataset Heartbeat --mode conformal --epsilon 0.05
```
