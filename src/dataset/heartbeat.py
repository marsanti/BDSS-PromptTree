import os
from .base import BaseTimeSeriesDataset # Assuming base.py is in the same directory
from config import config
import numpy as np

class HeartbeatDataset(BaseTimeSeriesDataset):
    """
    Loads the Heartbeat multivariate dataset from the UCR/UEA archive .ts format.
    Class labels are 'normal' and 'abnormal'.
    """
    def __init__(self,
                 validation_size=0.2,
                 random_state=42):

        self.dataset_name = 'Heartbeat'
        base_folder = os.path.join(config['DATASET_FOLDER'], 'Multivariate', self.dataset_name)

        # Construct file paths
        train_path = os.path.join(base_folder, f'{self.dataset_name}_TRAIN.ts')
        test_path = os.path.join(base_folder, f'{self.dataset_name}_TEST.ts')

        print(f"Attempting to load Multivariate dataset: {self.dataset_name}")
        if not os.path.exists(train_path):
             raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
             raise FileNotFoundError(f"Test file not found: {test_path}")

        super().__init__(
            train_path=train_path,
            test_path=test_path,
            validation_size=validation_size,
            random_state=random_state
        )

    def get_class_names(self) -> dict:
        """
        Returns the mapping from the internal integer labels (0, 1) assigned by the base class
        back to the meaningful names 'Normal' and 'Abnormal'.
        """
        original_names = {
            "normal": "Normal",     
            "abnormal": "Abnormal" 
        }
        # Use the inverse_label_map from the base class to link 0/1 back to original strings
        # Then use original_names to get the display name
        return {
            internal_label: original_names.get(original_label, f"Unknown Label {original_label}")
            for internal_label, original_label in self.inverse_label_map.items()
        }