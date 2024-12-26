import warnings
from collections import OrderedDict
import flwr as fl
import torch
import argparse
from ultralytics import YOLO
from pathlib import Path
import logging
import json
from datetime import datetime
import os
import sys
from ultralytics import settings

# Update a setting
# settings.update({"datasets_dir": "/app"})
settings.update({"runs_dir": "./runs"})

# Set logging level for flower
logging.getLogger('flwr').setLevel(logging.DEBUG)

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(cfg, model_name="yolo11n.pt"):
    """Load YOLO model
    
    Args:
        model_name (str): Name of the model to load. Defaults to "yolov8n.pt"
        
    Returns:
        YOLO: Loaded YOLO model
    """
    logging.info(f"Loading YOLO model: {model_name}")
    try:
        model = YOLO(model_name)
        model.to(DEVICE)
        logging.info(f"Model loaded successfully and moved to device: {DEVICE}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def train(model, data_yaml, epochs=1):
    """Train the YOLO model"""
    logging.info(f"Starting training for {epochs} epochs")
    logging.info(f"Using data config from: {data_yaml}")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            device=DEVICE,
            save=False  # Don't save checkpoints during FL training
        )
        logging.info("Training completed successfully")
        return results
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

def validate(model, data_yaml):
    """Validate the YOLO model"""
    logging.info("Starting validation")
    
    try:
        results = model.val(data=data_yaml)
        metrics = {
            "metrics/precision": float(results.results_dict['metrics/precision(B)']),
            "metrics/recall": float(results.results_dict['metrics/recall(B)']),
            "metrics/mAP50": float(results.results_dict['metrics/mAP50(B)']),
            "metrics/mAP50-95": float(results.results_dict['metrics/mAP50-95(B)'])
        }
        logging.info(f"Validation metrics: {metrics}")
        return float(results.results_dict['metrics/mAP50-95(B)']), metrics
    except Exception as e:
        logging.error(f"Error during validation: {str(e)}")
        raise

def setup_logging(cid):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create client-specific log directory
    client_log_dir = log_dir / f"client_{cid}"
    client_log_dir.mkdir(exist_ok=True)
    
    # Set up timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up file handler for general logs
    log_file = client_log_dir / f"fl_training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Changed to DEBUG level
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set up metrics log file
    metrics_file = client_log_dir / f"metrics_{timestamp}.json"
    
    return metrics_file

def save_metrics(metrics_file: Path, round_number: str, metrics: dict):
    """Save metrics to JSON file"""
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    data[f"round_{round_number}"] = metrics
    
    with open(metrics_file, 'w') as f:
        json.dump(data, f, indent=4)

def verify_data_yaml(data_yaml):
    """Verify that the data.yaml file exists and is valid"""
    try:
        yaml_path = Path(data_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
        logging.info(f"data.yaml verified at {data_yaml}")
        return True
    except Exception as e:
        logging.error(f"Error verifying data.yaml: {str(e)}")
        return False

class YOLOFlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.round = 0
        logging.debug("Initializing YOLOFlowerClient")
        
    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        logging.debug("Getting model parameters")
        try:
            params = [val.cpu().numpy() for _, val in MODEL.model.state_dict().items()]
            logging.debug(f"Successfully extracted {len(params)} parameter tensors")
            return params
        except Exception as e:
            logging.error(f"Error in get_parameters: {str(e)}")
            raise

    def set_parameters(self, parameters, model):
        """Set model parameters from a list of NumPy arrays."""
        logging.debug(f"Setting {len(parameters)} parameter tensors")
        try:
            params_dict = zip(model.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.model.load_state_dict(state_dict, strict=True)
            logging.debug("Parameters successfully set")
        except Exception as e:
            logging.error(f"Error in set_parameters: {str(e)}")
            raise

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        self.round += 1
        logging.debug(f"Starting fit() for round {self.round}")
        logging.debug(f"Received config: {config}")
        
        try:
            self.set_parameters(parameters, MODEL)
            
            # Get epochs from config, default to 1
            epochs = config.get("epochs", 1)
            logging.info(f"Training for {epochs} epochs")
            
            # Verify data.yaml before training
            if not verify_data_yaml(data_yaml):
                raise ValueError(f"Invalid data.yaml path: {data_yaml}")
            
            # Train the model
            results = train(MODEL, data_yaml, epochs=epochs)
            parameters = self.get_parameters(config)

            MODEL_COPY.model = copy.deepcopy(MODEL.model)
            
            # Log training metrics
            training_metrics = {
                "training_loss": float(results.results_dict.get('metrics/loss', 0)),
                "box_loss": float(results.results_dict.get('metrics/box_loss', 0)),
                "cls_loss": float(results.results_dict.get('metrics/cls_loss', 0)),
                "dfl_loss": float(results.results_dict.get('metrics/dfl_loss', 0))
            }
            save_metrics(metrics_file, f"{self.round}_train", training_metrics)
            logging.info(f"Training metrics for round {self.round}: {training_metrics}")
            
            return parameters, 1, training_metrics
        
        except Exception as e:
            logging.error(f"Error in fit() round {self.round}: {str(e)}")
            raise

    def evaluate(self, parameters, config):
        """Evaluate the model on the local dataset."""
        logging.debug(f"Starting evaluate() for round {self.round}")
        logging.debug(f"Received config: {config}")
        
        try:

            self.set_parameters(parameters, MODEL)
            self.set_parameters(parameters, MODEL_COPY)
            
            # Verify data.yaml before validation
            if not verify_data_yaml(data_yaml):
                raise ValueError(f"Invalid data.yaml path: {data_yaml}")
            
            loss, metrics = validate(MODEL_COPY, data_yaml)
            
            # Save validation metrics
            save_metrics(metrics_file, f"{self.round}_val", metrics)
            logging.info(f"Validation metrics for round {self.round}: {metrics}")
            
            return float(loss), 1, metrics
        
        except Exception as e:
            logging.error(f"Error in evaluate() round {self.round}: {str(e)}")
            raise

# Parse arguments
parser = argparse.ArgumentParser(description="Launches FL clients for YOLO.")
parser.add_argument('-cid', "--cid", type=int, default=0, help="Client ID")
parser.add_argument('-server', "--server", default="localhost", help="Server Address")
parser.add_argument('-port', "--port", default="30051", help="Server Port")
parser.add_argument('-data', "--data", default="./data/data.yaml", help="Path to data.yaml")
parser.add_argument('-model', "--model", default="yolo11n.pt", help="Path to YOLO model")
args = vars(parser.parse_args())

# Initialize global variables
cid = args['cid']
server = args['server']
port = args['port']
data_yaml = args['data']
model_path = args['model']

# Setup logging
metrics_file = setup_logging(cid)
logging.info(f"Client {cid} initialized")
logging.info(f"Device being used: {DEVICE}")

# Verify data.yaml exists before proceeding
if not verify_data_yaml(data_yaml):
    logging.error(f"data.yaml not found at {data_yaml}")
    sys.exit(1)

# Load model
try:
    import copy
    MODEL = load_model(model_path)
    MODEL_COPY = load_model(model_path)
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    sys.exit(1)

# Start Flower client
logging.info(f"Starting YOLO Federated Learning client {cid}")
logging.info(f"Connecting to FL server {server} on port {port}...")

try:
    fl.client.start_numpy_client(
        server_address=f"{server}:{port}",
        client=YOLOFlowerClient(),
    )
    logging.info("Federated learning session completed successfully")
except Exception as e:
    logging.error(f"Error in federated learning session: {str(e)}")
    raise