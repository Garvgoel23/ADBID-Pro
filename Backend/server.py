from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
from model import BidModel
import os
from datetime import datetime
import numpy as np
import logging
import json
from dataclasses import dataclass

@dataclass
class BidRequest:
    adSlotWidth: int = 300
    adSlotHeight: int = 250
    adSlotFloorPrice: float = 0.0
    advertiserId: int = 0
    region: str = "Unknown"
    city: str = "Unknown"
    adexchange: str = "Unknown"

# Configure logging with both file and console handlers
def setup_logging(log_dir: str = "backend/logs") -> None:
    """Set up logging configuration with both file and console output."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"server_{datetime.now().strftime('%Y%m%d')}.log")
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

def log_message(level: str, message: str) -> None:
    """Log a message with the specified level."""
    level = level.upper()
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    else:
        logging.warning(f"Unknown log level '{level}'. Message: {message}")

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value between min and max values."""
    try:
        if np.isnan(value) or np.isnan(min_val) or np.isnan(max_val):
            return 0.0
        if max_val - min_val == 0:
            return 0.0
        return float((value - min_val) / (max_val - min_val))
    except Exception as e:
        log_message("ERROR", f"Normalization failed: {str(e)}")
        return 0.0

def extract_features(bid_request: BidRequest) -> Optional[Dict[str, Any]]:
    """Extract features from a bid request."""
    try:
        features = {
            "Region": str(bid_request.region),
            "City": str(bid_request.city),
            "Adexchange": str(bid_request.adexchange),
            "Adslotfloorprice": float(bid_request.adSlotFloorPrice)
        }
        
        # Validate extracted features
        if any(v is None for v in features.values()):
            raise ValueError("Missing required features in bid request")
            
        return features
    except Exception as e:
        log_message("ERROR", f"Feature extraction failed: {str(e)}")
        return None

# Initialize Flask app
app = Flask(__name__)

# Set up logging
setup_logging()

# Initialize bidding system and model
bid_model = BidModel()

# Load the trained model at startup
if not bid_model.load_model():
    log_message("ERROR", "Failed to load model at startup!")

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    model_status = "loaded" if bid_model.model is not None else "not loaded"
    log_message("INFO", f"Health check - Model status: {model_status}")
    return jsonify({
        "status": "Server is running",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/predict_bid", methods=["POST"])
def predict_bid():
    """Receives a bid request in JSON format and returns the bid price."""
    try:
        # Parse JSON request
        data = request.json
        if not data:
            log_message("ERROR", "No data provided in request")
            return jsonify({"error": "No data provided"}), 400

        log_message("INFO", f"Received bid request: {data}")

        # Create BidRequest object
        bid_request = BidRequest(
            adSlotWidth=data.get("adSlotWidth", 300),
            adSlotHeight=data.get("adSlotHeight", 250),
            adSlotFloorPrice=float(data.get("adSlotFloorPrice", 0.0)),
            advertiserId=int(data.get("advertiserId", 0)),
            region=str(data.get("region", "Unknown")),
            city=str(data.get("city", "Unknown")),
            adexchange=str(data.get("adexchange", "Unknown"))
        )

        # Extract features
        features = extract_features(bid_request)
        if features is None:
            return jsonify({"error": "Failed to extract features from request"}), 400

        # Get ML-based bid prediction
        try:
            bid_price_ml = bid_model.predict_bid(features)
            
            # Create response
            response = {
                "predicted_bid": round(bid_price_ml, 2),
                "currency": "USD",
                "timestamp": datetime.now().isoformat(),
                "request_features": {
                    k: str(v) if isinstance(v, (np.int64, np.float64)) else v 
                    for k, v in features.items()
                }
            }

            log_message("INFO", f"Prediction successful: {response}")
            
            # Save prediction to log file
            log_file = os.path.join("backend/logs", f"predictions_{datetime.now().strftime('%Y%m%d')}.json")
            try:
                existing_data = []
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        existing_data = json.load(f)
                existing_data.append(response)
                with open(log_file, 'w') as f:
                    json.dump(existing_data, f, indent=4)
            except Exception as e:
                log_message("WARNING", f"Failed to save prediction to log file: {str(e)}")

            return jsonify(response), 200

        except Exception as e:
            log_message("ERROR", f"Prediction error: {str(e)}")
            return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    except Exception as e:
        log_message("ERROR", f"Request processing error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/model/reload", methods=["POST"])
def reload_model():
    """Endpoint to reload the model from disk."""
    try:
        success = bid_model.load_model()
        msg = "Model reloaded successfully" if success else "Failed to reload model"
        log_message("INFO" if success else "ERROR", msg)
        
        if success:
            return jsonify({"message": msg}), 200
        else:
            return jsonify({"error": msg}), 500
    except Exception as e:
        log_message("ERROR", f"Error reloading model: {str(e)}")
        return jsonify({"error": f"Error reloading model: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    log_message("WARNING", f"Route not found: {request.url}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    log_message("WARNING", f"Method not allowed: {request.method} {request.url}")
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_server_error(error):
    log_message("ERROR", f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
