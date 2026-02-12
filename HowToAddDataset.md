# How to Add a New Dataset

This project supports time series datasets in the `.ts` (UCR/UEA archive) or `.arff` format. To add a new dataset, follow these steps:

## 1. Prepare the Data Files

1.  Locate the `datasets` folder in the project root.
2.  Create a subfolder for your dataset, typically under `Univariate` or `Multivariate` depending on the data type.
3.  Ensure you have the Train and Test files, typically named `{DatasetName}_TRAIN.ts` and `{DatasetName}_TEST.ts`.

Example structure:
```text
datasets/
└── Multivariate/
    └── MyNewDataset/
        ├── MyNewDataset_TRAIN.ts
        └── MyNewDataset_TEST.ts
```

## 2. Create the Dataset Class

Create a new Python file in `src/dataset/` (e.g., `src/dataset/my_new_dataset.py`). Define a class that inherits from `BaseTimeSeriesDataset`.

You need to implement:
1.  `__init__`: Construct the paths to your train/test files and call `super().__init__`.
2.  `get_class_names`: Return a dictionary mapping the internal integer labels (assigned automatically starting from 0) to human-readable string names.

**Example (`src/dataset/my_new_dataset.py`):**

```python
import os
from .base import BaseTimeSeriesDataset
from config import config

class MyNewDataset(BaseTimeSeriesDataset):
    def __init__(self, validation_size=0.2, random_state=42):
        self.dataset_name = "MyNewDataset"
        
        # Define path relative to the configured DATASET_FOLDER
        base_folder = os.path.join(
            config["DATASET_FOLDER"], "Multivariate", self.dataset_name
        )

        train_path = os.path.join(base_folder, f"{self.dataset_name}_TRAIN.ts")
        test_path = os.path.join(base_folder, f"{self.dataset_name}_TEST.ts")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")

        super().__init__(
            train_path=train_path,
            test_path=test_path,
            validation_size=validation_size,
            random_state=random_state,
        )

    def get_class_names(self) -> dict:
        # Map internal labels (0, 1, ...) back to original names if needed.
        return {
            i: f"Class {label}" 
            for i, label in self.inverse_label_map.items()
        }
```

## 3. Register the Dataset

You need to update the main application logic to recognize the new dataset name.

### A. Update `src/main.py`

Import your new class and add a case to the `load_dataset` function.

```python
# ... imports
from dataset.my_new_dataset import MyNewDataset

def load_dataset(dataset_name, ...):
    # ...
    match dataset_name:
        # ... existing cases ...
        case "MyNewDataset":
            dataset = MyNewDataset(
                validation_size=validation_size, 
                random_state=random_state
            )
```

### B. Update `src/config.py`

Add your dataset name to the allowed choices in the argument parser so it can be passed via command line.

```python
# ...
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["ECG200", "ECG5000", "ECGFiveDays", "PenDigits", "Heartbeat", "MyNewDataset"], # Add here
    help="Name of the dataset to load...",
)
```

## 4. Run the Project

You can now run the project with your new dataset:

```bash
python src/main.py --dataset MyNewDataset --mode single_tree
```