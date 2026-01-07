"""
Training Script for Azure AutoML Model

This script provides a high-level interface for training models using Azure AutoML.

Usage:
    python scripts/train_model.py --config config.json
    python scripts/train_model.py --experiment-name my-experiment --task regression
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from azureml.core import Workspace, Experiment, Dataset
from azureml.train.automl import AutoMLConfig


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the training script."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/training.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def connect_workspace(config: Optional[Dict[str, Any]] = None) -> Workspace:
    """
    Connect to Azure ML Workspace.
    
    Args:
        config: Optional workspace configuration dictionary
        
    Returns:
        Connected Workspace object
    """
    if config:
        ws = Workspace(
            subscription_id=config['subscription_id'],
            resource_group=config['resource_group'],
            workspace_name=config['workspace_name']
        )
    else:
        # Try to load from config file
        ws = Workspace.from_config()
    
    return ws


def create_automl_config(
    task: str,
    dataset: Dataset,
    label_column: str,
    compute_target: str,
    **kwargs
) -> AutoMLConfig:
    """
    Create AutoML configuration.
    
    Args:
        task: ML task type ('classification', 'regression', 'forecasting')
        dataset: Training dataset
        label_column: Name of the label column
        compute_target: Name of the compute target
        **kwargs: Additional AutoML configuration parameters
        
    Returns:
        Configured AutoMLConfig object
    """
    default_config = {
        'task': task,
        'training_data': dataset,
        'label_column_name': label_column,
        'compute_target': compute_target,
        'enable_early_stopping': True,
        'enable_onnx_compatible_models': False,
        'iteration_timeout_minutes': 30,
        'max_concurrent_iterations': 4,
        'max_cores_per_iteration': -1,
        'experiment_timeout_minutes': 30,
        'primary_metric': get_default_metric(task),
        'enable_stack_ensembling': True,
        'enable_voting_ensemble': True,
    }
    
    # Merge with provided kwargs
    default_config.update(kwargs)
    
    return AutoMLConfig(**default_config)


def get_default_metric(task: str) -> str:
    """Get the default primary metric for a given task."""
    metrics = {
        'classification': 'AUC_weighted',
        'regression': 'normalized_root_mean_squared_error',
        'forecasting': 'normalized_root_mean_squared_error'
    }
    return metrics.get(task, 'AUC_weighted')


def train_model(
    workspace: Workspace,
    experiment_name: str,
    automl_config: AutoMLConfig,
    logger: logging.Logger
) -> Any:
    """
    Train a model using AutoML.
    
    Args:
        workspace: Azure ML Workspace
        experiment_name: Name of the experiment
        automl_config: AutoML configuration
        logger: Logger instance
        
    Returns:
        Best run from the AutoML experiment
    """
    logger.info(f"Starting experiment: {experiment_name}")
    
    experiment = Experiment(workspace, experiment_name)
    
    logger.info("Submitting AutoML run...")
    run = experiment.submit(automl_config, show_output=True)
    
    logger.info("Waiting for run to complete...")
    run.wait_for_completion(show_output=True)
    
    # Get best run
    best_run, fitted_model = run.get_output()
    
    logger.info(f"Best run ID: {best_run.id}")
    logger.info(f"Best run metrics: {best_run.get_metrics()}")
    
    return best_run, fitted_model


def save_model_info(run_id: str, metrics: Dict[str, Any], output_dir: str = "outputs") -> None:
    """Save model information to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    info = {
        'run_id': run_id,
        'metrics': metrics,
        'timestamp': str(Path.ctime(Path.cwd()))
    }
    
    output_file = os.path.join(output_dir, 'model_info.json')
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train Azure AutoML Model')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='automl-experiment',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression', 'forecasting'],
        default='regression',
        help='ML task type'
    )
    parser.add_argument(
        '--dataset-id',
        type=str,
        help='Dataset ID in Azure ML workspace'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default='target',
        help='Name of the label column'
    )
    parser.add_argument(
        '--compute-target',
        type=str,
        default='cpu-cluster',
        help='Name of the compute target'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("=" * 80)
    logger.info("Azure AutoML Training Script")
    logger.info("=" * 80)
    
    try:
        # Load configuration if provided
        config = load_config(args.config) if args.config else {}
        
        # Connect to workspace
        logger.info("Connecting to Azure ML Workspace...")
        workspace = connect_workspace(config.get('workspace'))
        logger.info(f"Connected to workspace: {workspace.name}")
        
        # Get dataset
        if args.dataset_id:
            logger.info(f"Loading dataset: {args.dataset_id}")
            dataset = Dataset.get_by_id(workspace, id=args.dataset_id)
        elif 'dataset_id' in config:
            dataset = Dataset.get_by_id(workspace, id=config['dataset_id'])
        else:
            raise ValueError("Dataset ID must be provided via --dataset-id or in config file")
        
        # Create AutoML configuration
        logger.info("Creating AutoML configuration...")
        automl_config = create_automl_config(
            task=args.task,
            dataset=dataset,
            label_column=args.label_column,
            compute_target=args.compute_target,
            **config.get('automl_settings', {})
        )
        
        # Train model
        best_run, fitted_model = train_model(
            workspace=workspace,
            experiment_name=args.experiment_name,
            automl_config=automl_config,
            logger=logger
        )
        
        # Save model info
        logger.info("Saving model information...")
        save_model_info(
            run_id=best_run.id,
            metrics=best_run.get_metrics()
        )
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Best model run ID: {best_run.id}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
