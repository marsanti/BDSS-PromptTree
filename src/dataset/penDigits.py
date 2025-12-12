import os
from .base import BaseTimeSeriesDataset
from config import config


class PenDigitsDataset(BaseTimeSeriesDataset):
    """
    Loads the PenDigits multivariate dataset from the UCR/UEA archive .ts format.
    PenDigits contains handwriting samples for digits 0-9, with 2 dimensions (x, y coordinates).
    """

    def __init__(self, validation_size=0.2, random_state=42):

        self.dataset_name = "PenDigits"
        # Construct the base folder path
        base_folder = os.path.join(
            config["DATASET_FOLDER"], "Multivariate", self.dataset_name
        )

        # Construct file paths
        train_path = os.path.join(base_folder, f"{self.dataset_name}_TRAIN.ts")
        test_path = os.path.join(base_folder, f"{self.dataset_name}_TEST.ts")

        print(f"Attempting to load Multivariate dataset: {self.dataset_name}")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")

        super().__init__(
            train_path=train_path,
            test_path=test_path,
            validation_size=validation_size,
            random_state=random_state,
        )

    def get_class_names(self) -> dict:
        """
        Returns the mapping from the internal integer labels (0-9) assigned by the base class
        back to the original digit labels (0-9).
        """
        # The original labels are digits 0 through 9.
        # The base class maps these originals (e.g., 0, 1, ..., 9) to internal integers (0, 1, ..., 9).
        # We want to return a map from the internal integers back to a meaningful name (which is just the digit itself).
        return {
            internal_label: f"Digit {original_label}"
            for internal_label, original_label in self.inverse_label_map.items()
        }
