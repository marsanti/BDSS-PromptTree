import os
from .base import BaseTimeSeriesDataset
from config import config
from typing import Literal

ALLOWED_ECG_DATASETS = Literal['ECG200', 'ECG5000', 'ECGFiveDays']

class ECGDataset(BaseTimeSeriesDataset):
    def __init__(self, ecg_dataset_name: ALLOWED_ECG_DATASETS = 'ECG200', validation_size=0.2, random_state=42):
        self.dataset_name = ecg_dataset_name 
        
        # Check if the provided name is valid (although Literal helps)
        if self.dataset_name not in ALLOWED_ECG_DATASETS.__args__:
            raise ValueError(f"Invalid ecg_dataset_name: {self.dataset_name}. "
                             f"Must be one of {ALLOWED_ECG_DATASETS.__args__}")

        # Assuming all datasets are in the 'Univariate' subfolder
        base_folder = os.path.join(config['DATASET_FOLDER'], 'Univariate', self.dataset_name)
        
        train_path = os.path.join(base_folder, f'{self.dataset_name}_TRAIN.ts')
        test_path = os.path.join(base_folder, f'{self.dataset_name}_TEST.ts')
        
        super().__init__(
            train_path=train_path,
            test_path=test_path,
            validation_size=validation_size,
            random_state=random_state
        )

    def _original_class_names(self) -> dict:
        """
        Returns the mapping from original labels found in the dataset files
        to their string names.
        """
        match(self.dataset_name):
            case 'ECG200':
                return {
                    -1: "Myocardial Infarction",
                    1: "Normal Heartbeat"
                }
            case 'ECG5000':
                return {
                    1: "Normal Heartbeat",
                    2: "R-on-T PVC", # R-on-T Premature Ventricular Contraction
                    3: "PVC", # Premature Ventricular Contraction
                    4: "SP", # Supraventricular Premature or Ectopic Beat
                    5: "UB" # Unclassified Beat
                }
            case 'ECGFiveDays':
                return {
                    1: "Normal Heartbeat",
                    2: "Abnormal Heartbeat"
                }
            case _:
                raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def get_class_names(self) -> dict:
        """
        Returns the mapping from integer labels (0, 1) back to 
        meaningful names for the ECG dataset.
        Assumes label_map correctly mapped {-1: 0, 1: 1} or {1: 0, -1: 1}.
        """
        # We need to use the actual mapping created in the base class
        # The base class maps original labels (-1, 1) to (0, 1) or (1, 0)
        # Let's map 0 and 1 back to meaningful names based on the original label
        original_names = self._original_class_names()
        
        # Use the inverse_label_map from the base class to link 0/1 to -1/1
        # Then use original_names to get the string name
        return { 
            integer_label: original_names[original_label] 
            for integer_label, original_label in self.inverse_label_map.items() 
        }