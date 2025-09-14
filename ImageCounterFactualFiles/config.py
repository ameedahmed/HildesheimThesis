"""
Configuration file for counterfactual analysis script
"""
from dataclasses import dataclass
from typing import Callable

@dataclass
class ModelConfig:
    """Configuration for different model types"""
    model_path: str
    num_classes: int
    labels_path: str
    test_dir: str
    label_processing_func: Callable[[str], str]

# Base directory for all paths
BASE_DIR = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"

# Model configurations
MODEL_CONFIGS = {
    'artstyle': ModelConfig(
        model_path=f"{BASE_DIR}/dataset_resized_train_val_test_combi/TrainValTestModelResult/finetune/20250726-093555-CSWin_96_24322_base_384-384/model_best.pth.tar",
        num_classes=6,
        labels_path=f"{BASE_DIR}/dataset_resized_train_val_test_combi/trainvaltestcombi_artstyle_class_index.json",
        test_dir=f"{BASE_DIR}/dataset_resized_train_val_test_combi/test",
        label_processing_func=lambda x: x.replace('train_', '').replace('_preprocessed_384_dataset', '')
    ),
    'century': ModelConfig(
        model_path=f"{BASE_DIR}/PortraitDataset/CenturyWiseLabelsInsteadofArtists/ResultModelTrainedCenturyWise/finetune/20250727-105348-CSWin_96_24322_base_384-384/model_best.pth.tar",
        num_classes=3,
        labels_path=f"{BASE_DIR}/PortraitDataset/CenturyWiseLabelsInsteadofArtists/century_class_index.json",
        test_dir=f"{BASE_DIR}/PortraitDataset/CenturyWiseLabelsInsteadofArtists/test_century",
        label_processing_func=lambda x: x.replace('_century_resized_dataset', '')
    ),
    'portrait': ModelConfig(
        model_path=f"{BASE_DIR}/outputSmoothingLossPLScheduler/NPortrait32B/finetune/20250713-125113-CSWin_96_24322_base_384-384/model_best.pth.tar",
        num_classes=9,
        labels_path=f"{BASE_DIR}/PortraitDataset/artist_class_index.json",
        test_dir=f"{BASE_DIR}/PortraitDataset/test",
        label_processing_func=lambda x: x.replace('_resizeddataset', '')
    )
}

# Output directories
OUTPUT_DIRS = {
    'correct_classification': f"{BASE_DIR}/ImageCounterfactualExplanations/correct_classification_exp&newimg",
    'wrong_classification': f"{BASE_DIR}/ImageCounterfactualExplanations/wrong_classification_exp&newimg",
    'csv_correct': f"{BASE_DIR}/ImageCounterfactualExplanations/Counterfactual_log.csv",
    'csv_wrong': f"{BASE_DIR}/ImageCounterfactualExplanations/WrongClassificationCounterfactual_log.csv"
}

# DataLoader settings
DATALOADER_CONFIG = {
    'batch_size': 4,  # Increased from 1 for better GPU utilization
    'num_workers': 4,
    'pin_memory': True,
    'shuffle': False
}

# Image processing settings
IMAGE_CONFIG = {
    'size': 384,
    'dpi': 150,
    'figsize': (8, 8)
}

# SEDC settings
SEDC_CONFIG = {
    'kernel_size': 4,
    'max_dist': 200,
    'ratio': 0.2,
    'perturbation_type': 'blur'
} 