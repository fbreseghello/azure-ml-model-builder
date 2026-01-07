"""
Model Deployment Script for Azure ML

This script handles deploying trained models to Azure endpoints.

Usage:
    python scripts/deploy_model.py --model-name my-model --service-name my-service
"""

import argparse
import json
import logging
import sys
from typing import Dict, Any, Optional

from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, AksWebservice, Webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the deployment script."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def connect_workspace() -> Workspace:
    """Connect to Azure ML Workspace."""
    return Workspace.from_config()


def get_model(workspace: Workspace, model_name: str, version: Optional[int] = None) -> Model:
    """
    Retrieve a registered model from the workspace.
    
    Args:
        workspace: Azure ML Workspace
        model_name: Name of the registered model
        version: Optional version number (latest if not specified)
        
    Returns:
        Model object
    """
    if version:
        return Model(workspace, name=model_name, version=version)
    else:
        return Model(workspace, name=model_name)


def create_inference_config(
    entry_script: str = "outputs/scoring_file_v_2_0_0.py",
    environment_name: str = "automl-env"
) -> InferenceConfig:
    """
    Create inference configuration for deployment.
    
    Args:
        entry_script: Path to the scoring script
        environment_name: Name of the environment
        
    Returns:
        InferenceConfig object
    """
    # Create or load environment
    env = Environment(name=environment_name)
    env.python.conda_dependencies_file = "outputs/conda_env_v_1_0_0.yml"
    
    inference_config = InferenceConfig(
        entry_script=entry_script,
        environment=env
    )
    
    return inference_config


def deploy_to_aci(
    workspace: Workspace,
    model: Model,
    service_name: str,
    inference_config: InferenceConfig,
    cpu_cores: int = 1,
    memory_gb: int = 1,
    logger: logging.Logger = None
) -> Webservice:
    """
    Deploy model to Azure Container Instance (ACI).
    
    Args:
        workspace: Azure ML Workspace
        model: Model to deploy
        service_name: Name for the web service
        inference_config: Inference configuration
        cpu_cores: Number of CPU cores
        memory_gb: Memory in GB
        logger: Logger instance
        
    Returns:
        Deployed Webservice object
    """
    if logger:
        logger.info(f"Deploying to ACI with {cpu_cores} CPU cores and {memory_gb}GB memory")
    
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        auth_enabled=True,
        enable_app_insights=True,
        collect_model_data=True
    )
    
    service = Model.deploy(
        workspace=workspace,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True
    )
    
    if logger:
        logger.info("Waiting for deployment to complete...")
    
    service.wait_for_deployment(show_output=True)
    
    return service


def deploy_to_aks(
    workspace: Workspace,
    model: Model,
    service_name: str,
    inference_config: InferenceConfig,
    compute_target_name: str,
    logger: logging.Logger = None
) -> Webservice:
    """
    Deploy model to Azure Kubernetes Service (AKS).
    
    Args:
        workspace: Azure ML Workspace
        model: Model to deploy
        service_name: Name for the web service
        inference_config: Inference configuration
        compute_target_name: Name of the AKS compute target
        logger: Logger instance
        
    Returns:
        Deployed Webservice object
    """
    if logger:
        logger.info(f"Deploying to AKS cluster: {compute_target_name}")
    
    deployment_config = AksWebservice.deploy_configuration(
        autoscale_enabled=True,
        autoscale_min_replicas=1,
        autoscale_max_replicas=3,
        cpu_cores=1,
        memory_gb=2,
        auth_enabled=True,
        enable_app_insights=True,
        collect_model_data=True
    )
    
    service = Model.deploy(
        workspace=workspace,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        deployment_target=workspace.compute_targets[compute_target_name],
        overwrite=True
    )
    
    if logger:
        logger.info("Waiting for deployment to complete...")
    
    service.wait_for_deployment(show_output=True)
    
    return service


def test_service(service: Webservice, test_data: Dict[str, Any], logger: logging.Logger) -> Any:
    """
    Test the deployed service with sample data.
    
    Args:
        service: Deployed web service
        test_data: Test data dictionary
        logger: Logger instance
        
    Returns:
        Prediction results
    """
    logger.info("Testing deployed service...")
    
    input_data = json.dumps(test_data)
    result = service.run(input_data)
    
    logger.info(f"Test prediction result: {result}")
    
    return result


def save_service_info(service: Webservice, output_file: str = "outputs/service_info.json") -> None:
    """Save service information to a JSON file."""
    info = {
        'name': service.name,
        'scoring_uri': service.scoring_uri,
        'swagger_uri': service.swagger_uri,
        'state': service.state
    }
    
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Deploy Azure AutoML Model')
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the registered model'
    )
    parser.add_argument(
        '--service-name',
        type=str,
        required=True,
        help='Name for the web service'
    )
    parser.add_argument(
        '--deployment-target',
        type=str,
        choices=['aci', 'aks'],
        default='aci',
        help='Deployment target (ACI or AKS)'
    )
    parser.add_argument(
        '--aks-cluster',
        type=str,
        help='Name of AKS cluster (required for AKS deployment)'
    )
    parser.add_argument(
        '--cpu-cores',
        type=int,
        default=1,
        help='Number of CPU cores (for ACI)'
    )
    parser.add_argument(
        '--memory-gb',
        type=int,
        default=1,
        help='Memory in GB (for ACI)'
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
    logger.info("Azure ML Model Deployment Script")
    logger.info("=" * 80)
    
    try:
        # Connect to workspace
        logger.info("Connecting to Azure ML Workspace...")
        workspace = connect_workspace()
        logger.info(f"Connected to workspace: {workspace.name}")
        
        # Get model
        logger.info(f"Retrieving model: {args.model_name}")
        model = get_model(workspace, args.model_name)
        logger.info(f"Model version: {model.version}")
        
        # Create inference config
        logger.info("Creating inference configuration...")
        inference_config = create_inference_config()
        
        # Deploy model
        if args.deployment_target == 'aci':
            service = deploy_to_aci(
                workspace=workspace,
                model=model,
                service_name=args.service_name,
                inference_config=inference_config,
                cpu_cores=args.cpu_cores,
                memory_gb=args.memory_gb,
                logger=logger
            )
        else:  # AKS
            if not args.aks_cluster:
                raise ValueError("--aks-cluster is required for AKS deployment")
            
            service = deploy_to_aks(
                workspace=workspace,
                model=model,
                service_name=args.service_name,
                inference_config=inference_config,
                compute_target_name=args.aks_cluster,
                logger=logger
            )
        
        # Save service info
        logger.info("Saving service information...")
        save_service_info(service)
        
        logger.info("=" * 80)
        logger.info("Deployment completed successfully!")
        logger.info(f"Service name: {service.name}")
        logger.info(f"Scoring URI: {service.scoring_uri}")
        logger.info(f"Swagger URI: {service.swagger_uri}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
