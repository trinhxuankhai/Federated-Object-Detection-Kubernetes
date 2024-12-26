from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import sys
import numpy as np

# Set logging level for flower
logging.getLogger('flwr').setLevel(logging.DEBUG)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average of metrics from clients."""
    if not metrics:
        return {}
    
    # Extract values and weights
    values = []
    weights = []
    keys = metrics[0][1].keys()
    
    # Organize metrics by key
    metrics_by_key = {key: [] for key in keys}
    for weight, metric_dict in metrics:
        weights.append(weight)
        for key, value in metric_dict.items():
            metrics_by_key[key].append(value)
    
    # Calculate weighted average for each metric
    weighted_metrics = {}
    for key, values in metrics_by_key.items():
        weighted_metrics[key] = np.average(values, weights=weights)
    
    return weighted_metrics

def setup_server_logging():
    """Set up logging for the server"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs/server")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up log files
    log_file = log_dir / f"fl_server_{timestamp}.log"
    metrics_file = log_dir / f"server_metrics_{timestamp}.json"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG level
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return metrics_file

class YOLOFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, metrics_file: Path, **kwargs):
        super().__init__(**kwargs)
        self.metrics_file = metrics_file
        self.current_round = 0
        logging.debug("Initialized YOLOFedAvg strategy")

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation metrics and save them"""
        self.current_round = server_round
        logging.debug(f"Round {server_round}: Aggregating evaluation results")
        logging.debug(f"Number of results: {len(results)}, Number of failures: {len(failures)}")
        
        if results:
            try:
                # Aggregate metrics using parent class method
                aggregated = super().aggregate_evaluate(server_round, results, failures)
                
                if aggregated:
                    loss, metrics = aggregated
                    metrics_dict = {
                        "loss": loss,
                        **metrics
                    }
                    
                    # Save metrics
                    # save_server_metrics(self.metrics_file, server_round, metrics_dict)
                    logging.info(f"Round {server_round} completed. Metrics: {metrics_dict}")
                    
                    return aggregated
                else:
                    logging.warning(f"Round {server_round}: No metrics were aggregated")
            except Exception as e:
                logging.error(f"Error in aggregate_evaluate: {str(e)}")
                raise
        else:
            logging.warning(f"Round {server_round}: No results to aggregate")
        
        return None

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate model updates from clients"""
        logging.debug(f"Round {server_round}: Aggregating fit results")
        logging.debug(f"Number of results: {len(results)}, Number of failures: {len(failures)}")
        
        try:
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated:
                logging.debug(f"Round {server_round}: Successfully aggregated parameters")
            else:
                logging.warning(f"Round {server_round}: No parameters were aggregated")
            return aggregated
        except Exception as e:
            logging.error(f"Error in aggregate_fit: {str(e)}")
            raise

# Parse inputs
parser = argparse.ArgumentParser(description="Launches FL server for YOLO.")
parser.add_argument('-clients', "--clients", type=int, default=2, 
                    help="Define the number of clients to be part of the FL process")
parser.add_argument('-min', "--min", type=int, default=2, 
                    help="Minimum number of available clients")
parser.add_argument('-rounds', "--rounds", type=int, default=5, 
                    help="Number of FL rounds")
parser.add_argument('-port', "--port", type=str, default="8080", 
                    help="Port for the server")
args = vars(parser.parse_args())

num_clients = args['clients']
min_clients = args['min']
rounds = args['rounds']
port = args['port']

# Setup logging
metrics_file = setup_server_logging()
logging.info(f"Starting FL server with {num_clients} clients, {min_clients} minimum clients, {rounds} rounds")

# Define strategy
strategy = YOLOFedAvg(
    metrics_file=metrics_file,
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=num_clients,
    min_evaluate_clients=num_clients,
    min_available_clients=min_clients,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=None,  # Allow server to request parameters from client
)

# Start Flower server
logging.info(f"Starting server on port {port}")
try:
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
    logging.info("Server completed all rounds successfully")
except Exception as e:
    logging.error(f"Server error: {str(e)}")
    raise