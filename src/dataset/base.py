import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod 

class BaseTimeSeriesDataset(ABC):
    def __init__(self, train_path, test_path, validation_size=0.2, random_state=42):
        
        # Determine loading function based on file extension
        load_func_train = self._get_load_func(train_path)
        load_func_test = self._get_load_func(test_path)

        # Load data
        X_full_train, self.Y_train_raw = load_func_train(train_path)
        self.X_test_raw, self.Y_test_raw = load_func_test(test_path)
        
        # Convert labels to consistent integer type if possible
        # Find unique labels across train and test
        unique_labels = np.unique(np.concatenate((self.Y_train_raw, self.Y_test_raw)))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.inverse_label_map = {i: label for label, i in self.label_map.items()}
        print(f"Detected labels: {unique_labels}. Mapping to: {self.label_map}")

        # Apply mapping
        self.Y_train_full = np.array([self.label_map[y] for y in self.Y_train_raw])
        self.Y_test = np.array([self.label_map[y] for y in self.Y_test_raw])

        # Split the full training set into a new training set and a validation set
        if validation_size > 0.0 and validation_size < 1.0:
            # Perform the split only if validation_size is valid
            self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
                X_full_train,
                self.Y_train_full,
                test_size=validation_size,
                random_state=random_state,
                stratify=self.Y_train_full
            )
            print("Split training data into train and validation sets.")
        elif validation_size == 0.0:
            # If validation_size is 0, use the full training set for training
            # and create empty validation sets.
            print("Validation size is 0. Using full training set for training, validation set is empty.")
            self.X_train = X_full_train
            self.Y_train = self.Y_train_full
            # Create empty arrays with the correct number of dimensions for consistency
            val_channels = self.X_train.shape[1] if self.X_train.ndim > 1 else 0
            val_timesteps = self.X_train.shape[2] if self.X_train.ndim > 2 else 0
            self.X_val = np.empty((0, val_channels, val_timesteps), dtype=X_full_train.dtype)
            self.Y_val = np.empty((0,), dtype=self.Y_train_full.dtype)
        else:
            # Raise an error for invalid validation_size values
            raise ValueError(f"validation_size must be >= 0.0 and < 1.0. Got: {validation_size}")

        # Reshape to (n_samples, n_channels, n_timesteps)
        # Apply to train, val, and test splits
        self.X_train = self._reshape_data(self.X_train)
        self.X_val = self._reshape_data(self.X_val)
        self.X_test = self._reshape_data(self.X_test_raw) # Reshape raw test X


    def _get_load_func(self, file_path):
        """Returns the appropriate loading function based on file extension."""
        if file_path.endswith('.ts'):
            return self._load_and_process_ts
        elif file_path.endswith('.arff'):
            return self._load_and_process_arff
        else:
             raise ValueError(f"Unsupported file format: {file_path}")

    def _load_and_process_arff(self, file_path):
        """Loads data from an ARFF file."""
        try:
            from scipy.io import arff 
        except ImportError:
             raise ImportError("Loading .arff files requires scipy. Please install it (`pip install scipy`).")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data, meta = arff.loadarff(f)
        df = pd.DataFrame(data)

        # Infer attribute columns (excluding the target)
        target_col = meta.names()[-1] # Assume target is the last column
        att_cols = [col for col in meta.names() if col != target_col]
        
        X = df[att_cols].astype(float).values
        # Handle string labels in ARFF by decoding if necessary
        Y_raw = df[target_col]
        if pd.api.types.is_object_dtype(Y_raw):
            Y = Y_raw.str.decode('utf-8').values
        else:
            Y = Y_raw.values # Keep as is if numeric

        print(f"Loaded {len(X)} samples from ARFF: {file_path}")
        return X, Y

    def _load_and_process_ts(self, file_path):
        """Loads data from a .ts file (handles both univariate and multivariate)."""
        all_samples_data = []
        all_labels = []
        is_data_section = False
        labels_are_numeric = True
        num_dimensions = 1

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Metadata parsing
                if line.lower().startswith('@data'):
                    is_data_section = True
                    continue
                elif line.lower().startswith('@dimensions'): 
                    try:
                        num_dimensions = int(line.split()[-1])
                    except:
                        print(f"Warning: Could not parse @dimension in {file_path}. Assuming univariate or checking first data line.")
                elif line.startswith('@'):
                     continue # Skip other metadata

                if is_data_section:
                    parts = line.split(':')
                    if len(parts) < 2:
                        print(f"Warning: Invalid data line format in {file_path}, line {line_num+1}. Skipping row.")
                        continue

                    label_str = parts[-1]
                    dimensions_data_str = parts[:-1] # List of strings, each holding one dimension's data

                    # Dynamically check dimensions if not specified in metadata
                    if line_num == 0 and num_dimensions == 1 and len(dimensions_data_str) > 1:
                         print(f"Info: Detected {len(dimensions_data_str)} dimensions from first data line in {file_path}.")
                         num_dimensions = len(dimensions_data_str)

                    if len(dimensions_data_str) != num_dimensions:
                        print(f"Warning: Mismatched dimensions in {file_path}, line {line_num+1}. Expected {num_dimensions}, found {len(dimensions_data_str)}. Skipping row.")
                        continue

                    sample_channels = []
                    valid_sample = True
                    for dim_str in dimensions_data_str:
                        try:
                            # Values usually comma-separated per dimension
                            channel_values = np.array([float(v) for v in dim_str.split(',')])
                            sample_channels.append(channel_values)
                        except ValueError as e:
                            print(f"Warning: Could not parse series values in {file_path}, line {line_num+1}, dimension data: '{dim_str}'. Error: {e}. Skipping row.")
                            valid_sample = False
                            break # Skip this sample

                    if valid_sample:
                        # Stack channels together for this sample -> shape (num_channels, num_timesteps)
                        all_samples_data.append(np.stack(sample_channels))
                        all_labels.append(label_str)
                        if labels_are_numeric:
                            try: float(label_str)
                            except ValueError: labels_are_numeric = False

        if not all_samples_data:
             raise ValueError(f"No data parsed from {file_path}.")

        # Stack samples together -> shape (num_samples, num_channels, num_timesteps)
        try:
             X = np.stack(all_samples_data).astype(float)
        except ValueError:
             raise ValueError(f"Could not stack samples. Check for inconsistent time series lengths or dimensions in {file_path}.")

        # Convert labels
        Y = np.array(all_labels)
        if labels_are_numeric:
            try:
                Y_float = Y.astype(float)
                if np.all(Y_float == Y_float.astype(int)): Y = Y_float.astype(int)
                else: Y = Y_float
            except ValueError: pass

        print(f"Loaded {len(X)} samples ({X.shape[1]} channels, {X.shape[2]} timesteps) from TS: {file_path}")
        return X, Y


    def _reshape_data(self, X):
        """Ensures data is (n_samples, n_channels, n_timesteps)."""
        if X.ndim == 3:
            return X
        elif X.ndim == 2:
             print("Warning: Reshaping 2D data to 3D (assuming univariate).")
             return X[:, np.newaxis, :]
        else:
            raise ValueError(f"Input data X has unexpected dimension: {X.ndim}")

    @abstractmethod
    def get_class_names(self) -> dict:
        pass

    def get_original_class_names(self) -> dict:
        return self.inverse_label_map