import sys
import os
import argparse
import json
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import ContrastiveEEGDataset, SupervisedEEGDataset
from utils import load_eeg_data, validate_config, set_seed, setup_logging, setup_tensorboard, get_tensorboard_logger, close_tensorboard
from models import SimpleSleepNet, SleepStageClassifier
from training import train_contrastive_model, train_fully_finetuned_classifier
from evaluation import LatentSpaceEvaluator, get_predictions, ResultsSaver
from augmentations import load_augmentations_from_config

def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

suppress_warnings()

NUM_CLASSES = 5

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Sleep Stage Classification')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default/config.json',
        help='Path to the config file. Example: configs/experiment1/config1.json'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configuration files and exit.'
    )
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    if not os.path.isfile(config_path):
        logging.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from the config file: {e}")
        sys.exit(1)

def list_available_configs(configs_dir='configs'):
    """
    List all available configuration files.

    Args:
        configs_dir (str): Directory containing configuration files.
    """
    print("Available configuration files:")
    for root, dirs, files in os.walk(configs_dir):
        for file in files:
            if file.endswith('.json'):
                config_path = os.path.join(root, file)
                print(config_path)

def setup_environment(config):
    """
    Set up logging, seed, determine the device, and initialize TensorBoard.
    """
    setup_logging(log_level=logging.INFO, log_file=f'logs/experiment_{config["experiment_num"]}.log')
    logger = logging.getLogger(__name__)
    logger.info("Starting the EEG Project")
    
    set_seed(config["seed"])
    logger.info(f"Random seed set to {config['seed']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    setup_tensorboard(log_dir=f'runs/experiment_{config["experiment_num"]}')
    tensorboard_logger = get_tensorboard_logger()
    logger.info(f"TensorBoard logging initialized at: runs/experiment_{config['experiment_num']}")
    
    return logger, device, tensorboard_logger

def prepare_datasets(config, logger):
    """
    Prepare data for training and evaluation.

    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Logger instance.

    Returns:
        tuple: EEG data, training DataLoader, and test DataLoader.
    """
    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    NUM_WORKERS = config["num_workers"]
    eeg_data = load_eeg_data(dataset_path=config['dataset']['dset_path'], num_files_to_process=config['dataset']['max_files'])
    logger.info("Loaded train and test sets of EEG data")

    # Create datasets and dataloaders
    train_dataset = SupervisedEEGDataset(eeg_data['train'])
    test_dataset = SupervisedEEGDataset(eeg_data['test'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    logger.info("Supervised datasets and dataloaders created.")
    return eeg_data, train_loader, test_loader

def pretrain_contrastive_model(config, eeg_data, device, logger, tensorboard_logger):
    """
    Pretrain the contrastive model.

    Args:
        config (dict): Configuration dictionary.
        eeg_data (dict): EEG data.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger instance.

    Returns:
        SimpleSleepNet: Pretrained encoder model.
    """
    BATCH_SIZE = config["pretraining_params"]["batch_size"]
    LATENT_DIM = config["pretraining_params"]["latent_dim"]
    DROP_PROB = config["pretraining_params"]["dropout_rate"]
    NUM_WORKERS = config["num_workers"]
    TEMP = config["pretraining_params"]["temperature"]
    augmentations = load_augmentations_from_config(config=config)

    train_contrastive_dataset = ContrastiveEEGDataset(eeg_data['train'], augmentations=augmentations)
    train_contrastive_loader = DataLoader(train_contrastive_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
    logger.info(f"Contrastive train dataset created with {len(train_contrastive_dataset)} samples")

    # Create validation dataset and dataloader
    val_contrastive_dataset = ContrastiveEEGDataset(eeg_data['test'], augmentations=augmentations)
    val_contrastive_loader = DataLoader(val_contrastive_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    logger.info(f"Contrastive test dataset created with {len(val_contrastive_dataset)} samples")

    encoder = SimpleSleepNet(latent_dim=LATENT_DIM, dropout=DROP_PROB).to(device)
    
    # Log the encoder model architecture to TensorBoard
    sample_input = torch.zeros(1, 1, 3000).to(device)
    tensorboard_logger.add_graph(encoder, sample_input)
    
    logger.info(f"Model created with {sum(p.numel() for p in encoder.parameters() if p.requires_grad)} trainable parameters")

    contrastive_optimizer = optim.Adam(encoder.parameters(), lr=config["pretraining_params"]["learning_rate"])
    best_encoder_pth = f"{config['pretraining_params']['best_model_pth']}{config['experiment_num']}.pth"

    train_contrastive_model(
        model=encoder,
        dataloader=train_contrastive_loader,
        optimizer=contrastive_optimizer,
        device=device,
        num_epochs=config["pretraining_params"]["max_epochs"],
        temperature=TEMP,
        val_dataloader=val_contrastive_loader,
        check_interval=config["pretraining_params"]["check_interval"],
        min_improvement=config["pretraining_params"]["min_improvement"],
        best_model_path=best_encoder_pth
    )
    logger.info("Contrastive training complete")
    
    # Remove the model saving code, as it's handled inside train_contrastive_model
    # try:
    #     torch.save(encoder.state_dict(), best_encoder_pth)
    #     logger.info("Saved best encoder to %s", best_encoder_pth)
    # except Exception as e:
    #     logger.error("Error saving model: %s", str(e))
    #     raise

    # Load the best encoder weights after training
    try:
        encoder.load_state_dict(torch.load(best_encoder_pth))
        logger.info("Loaded best encoder from %s", best_encoder_pth)
    except Exception as e:
        logger.error("Error loading best encoder: %s", str(e))
        raise

    return encoder

# Comment out latent space evaluation
# def evaluate_latent_space(config, encoder, eeg_data, device, logger):
#     # ...existing code...

def train_full_model(config, encoder, train_loader, test_loader, device, logger, tensorboard_logger):
    """
    Train the encoder and classifier together (full fine-tuning).

    Args:
        config (dict): Configuration dictionary.
        encoder (SimpleSleepNet): Pretrained encoder model.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger instance.

    Returns:
        tuple: Trained classifier model and paths to the best model checkpoints.
    """
    LATENT_DIM = config["pretraining_params"]["latent_dim"]
    DROP_PROB = config["sup_training_params"]["dropout_rate"]
    classifier = SleepStageClassifier(input_dim=LATENT_DIM, num_classes=NUM_CLASSES, dropout_probs=DROP_PROB).to(device)
    
    # Log the classifier model architecture to TensorBoard
    sample_input = torch.zeros(1, LATENT_DIM).to(device)
    tensorboard_logger.add_graph(classifier, sample_input)
    
    # Compute class weights to handle class imbalance
    labels = []
    for _, label in train_loader.dataset:
        labels.append(label)
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # Normalize
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Unfreeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = True
    logger.info("Encoder parameters unfrozen for full fine-tuning")
    
    # Combine parameters
    combined_params = list(encoder.parameters()) + list(classifier.parameters())
    supervised_optimizer = optim.Adam(combined_params, lr=config["sup_training_params"]["learning_rate"])
    logger.info(f"Combined model parameters prepared.")
    
    best_encoder_pth = f"{config['sup_training_params']['best_model_pth']}encoder_{config['experiment_num']}.pth"
    best_classifier_pth = f"{config['sup_training_params']['best_model_pth']}classifier_{config['experiment_num']}.pth"
    train_fully_finetuned_classifier(
        encoder=encoder,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=test_loader,  # Use test_loader as validation loader
        criterion=criterion,
        optimizer=supervised_optimizer,
        num_epochs=config["sup_training_params"]["max_epochs"],
        device=device,
        save_encoder_path=best_encoder_pth,
        save_classifier_path=best_classifier_pth,
        check_interval=config["sup_training_params"]["check_interval"],
        min_improvement=config["sup_training_params"]["min_improvement"]
    )
    logger.info("Full fine-tuning of encoder and classifier complete")
    return classifier, (best_encoder_pth, best_classifier_pth)

# Comment out the old classifier training function
# def train_supervised_classifier( ... ):
#     # ...existing code...

def test_and_save_results(config, encoder, classifier, test_loader, device, logger):
    """
    Test the model and save the classification results.
    """
    # Load the best encoder and classifier weights
    best_encoder_pth = f"{config['sup_training_params']['best_model_pth']}encoder_{config['experiment_num']}.pth"
    best_classifier_pth = f"{config['sup_training_params']['best_model_pth']}classifier_{config['experiment_num']}.pth"
    encoder.load_state_dict(torch.load(best_encoder_pth))
    classifier.load_state_dict(torch.load(best_classifier_pth))
    encoder.to(device)
    classifier.to(device)
    encoder.eval()
    classifier.eval()
    
    # Get predictions and true labels
    predictions, true_labels = get_predictions(encoder, classifier, test_loader, device=device)
    
    # Save the classification results
    results_saver = ResultsSaver(
        results_folder=config["results_folder"],
        experiment_num=config["experiment_num"]
    )
    results_saver.save_classification_results(
        predictions=predictions,
        true_labels=true_labels,
        num_classes=NUM_CLASSES
    )
    logger.info("Classification results saved")

def main():
    """
    Main function to run the entire pipeline.
    """
    args = parse_args()

    if args.list_configs:
        list_available_configs()
        sys.exit(0)

    config = load_config(args.config)
    validate_config(config)

    logger, device, tensorboard_logger = setup_environment(config)
    eeg_data, train_loader, test_loader = prepare_datasets(config, logger)
    encoder = pretrain_contrastive_model(config, eeg_data, device, logger, tensorboard_logger)
    # Comment out the old classifier training call
    # classifier, _ = train_supervised_classifier(config, encoder, train_loader, test_loader, device, logger, tensorboard_logger)
    # Call the new training function
    classifier, _ = train_full_model(config, encoder, train_loader, test_loader, device, logger, tensorboard_logger)
    test_and_save_results(config, encoder, classifier, test_loader, device, logger)
    logger.info("Experiment complete")
    close_tensorboard()

if __name__ == "__main__":
    main()