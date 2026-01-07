"""
Model Testing and Evaluation Script

This script tests deployed models and evaluates their performance.

Usage:
    python scripts/test_model.py --service-name my-service --test-data test_data.json
"""

import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from azureml.core import Workspace
from azureml.core.webservice import Webservice


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_test_data(file_path: str) -> Dict[str, Any]:
    """Load test data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_sample_data() -> Dict[str, Any]:
    """Create sample test data for bike rental prediction."""
    return {
        'data': [{
            'day': 1,
            'mnth': 1,
            'year': 1,
            'season': 1,
            'holiday': 0,
            'weekday': 1,
            'workingday': 1,
            'weathersit': 1,
            'temp': 0.3,
            'atemp': 0.3,
            'hum': 0.6,
            'windspeed': 0.15
        }]
    }


def test_endpoint(
    service: Webservice,
    test_data: Dict[str, Any],
    logger: logging.Logger
) -> Any:
    """
    Test a deployed endpoint with data.
    
    Args:
        service: Deployed web service
        test_data: Test data dictionary
        logger: Logger instance
        
    Returns:
        Prediction results
    """
    logger.info(f"Testing service: {service.name}")
    logger.info(f"Scoring URI: {service.scoring_uri}")
    
    input_data = json.dumps(test_data)
    
    try:
        result = service.run(input_data)
        logger.info(f"Prediction successful!")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def batch_test(
    service: Webservice,
    test_data_list: List[Dict[str, Any]],
    logger: logging.Logger
) -> List[Any]:
    """
    Test endpoint with multiple data points.
    
    Args:
        service: Deployed web service
        test_data_list: List of test data dictionaries
        logger: Logger instance
        
    Returns:
        List of prediction results
    """
    results = []
    
    for i, test_data in enumerate(test_data_list):
        logger.info(f"Testing sample {i+1}/{len(test_data_list)}")
        result = test_endpoint(service, test_data, logger)
        results.append(result)
    
    return results


def save_results(results: List[Any], output_file: str = "outputs/test_results.json") -> None:
    """Save test results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Test Azure ML Model Endpoint')
    parser.add_argument(
        '--service-name',
        type=str,
        required=True,
        help='Name of the deployed web service'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data JSON file'
    )
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Use sample data for testing'
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
    logger.info("Azure ML Model Testing Script")
    logger.info("=" * 80)
    
    try:
        # Connect to workspace
        logger.info("Connecting to Azure ML Workspace...")
        workspace = Workspace.from_config()
        logger.info(f"Connected to workspace: {workspace.name}")
        
        # Get service
        logger.info(f"Retrieving service: {args.service_name}")
        service = Webservice(workspace, name=args.service_name)
        logger.info(f"Service state: {service.state}")
        
        # Load or create test data
        if args.use_sample:
            logger.info("Using sample test data")
            test_data = create_sample_data()
        elif args.test_data:
            logger.info(f"Loading test data from: {args.test_data}")
            test_data = load_test_data(args.test_data)
        else:
            raise ValueError("Either --test-data or --use-sample must be specified")
        
        # Test endpoint
        logger.info("Testing endpoint...")
        result = test_endpoint(service, test_data, logger)
        
        # Display results
        logger.info("=" * 80)
        logger.info("Test Results:")
        logger.info(f"Prediction: {result}")
        logger.info("=" * 80)
        
        # Save results
        logger.info("Saving results...")
        save_results([result])
        
        logger.info("Testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
