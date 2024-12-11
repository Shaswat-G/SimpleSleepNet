import os
import numpy as np
import glob
import logging
from typing import Dict, Optional
import random

logger = logging.getLogger(__name__)

def load_eeg_data(dataset_path: str, num_files_to_process: Optional[int] = None) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Loads and organizes EEG data from .npz files in the specified dataset path, splitting them into train and test sets.

    Parameters:
    - dataset_path (str): Path to the directory containing .npz files.
    - num_files_to_process (int, optional): Number of files to process. If None, processes all files.

    Returns:
    - eeg_data (dict): EEG epochs organized by set ('train', 'test') and label (0-4).
    """
    eeg_data = {
        'train': {label: [] for label in range(5)},
        'test': {label: [] for label in range(5)},
    }

    try:
        npz_files = sorted(glob.glob(os.path.join(dataset_path, '*.npz')))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {dataset_path}.")
        if num_files_to_process is not None:
            npz_files = npz_files[:num_files_to_process]
        logger.info(f"Processing {len(npz_files)} npz files from {dataset_path}.")

        # Extract subject indices from filenames
        subject_indices = []
        for npz_file in npz_files:
            basename = os.path.basename(npz_file)
            subject_idx = int(basename[3:5])
            subject_indices.append(subject_idx)
        unique_subject_indices = list(set(subject_indices))

        # Shuffle the subject indices
        random.shuffle(unique_subject_indices)
        
        # Compute split sizes based on the total number of unique subjects
        total_subjects = len(unique_subject_indices)
        train_size = int(total_subjects * 0.85)
        test_size = total_subjects - train_size

        # Partition the list into train and test subjects
        train_subjects = unique_subject_indices[:train_size]
        test_subjects = unique_subject_indices[train_size:]

        logger.info(f"Subjects split into train ({len(train_subjects)}), test ({len(test_subjects)}).")

        # Process files and assign to corresponding sets
        for idx, npz_file in enumerate(npz_files, 1):
            try:
                basename = os.path.basename(npz_file)
                subject_idx = int(basename[3:5])

                # Determine the set for the current subject
                if subject_idx in train_subjects:
                    set_name = 'train'
                elif subject_idx in test_subjects:
                    set_name = 'test'
                else:
                    logger.warning(f"Subject index {subject_idx} not found in any set.")
                    continue

                with np.load(npz_file) as data:
                    eeg_epochs, labels = data['x'], data['y']
                    for label in range(5):
                        eeg_data[set_name][label].extend(eeg_epochs[labels == label])
            except Exception as e:
                logger.error(f"Error processing {npz_file}: {e}")
            if idx % 10 == 0 or idx == len(npz_files):
                logger.info(f"Processed {idx}/{len(npz_files)} files.")

        # Convert lists to numpy arrays
        for set_name in eeg_data.keys():
            for label in eeg_data[set_name].keys():
                eeg_data[set_name][label] = np.array(eeg_data[set_name][label])

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    return eeg_data