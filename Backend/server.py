from flask import Flask, request, jsonify
from typing import Dict, Any
from model import BidModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Initialize bidding system and model
bid_model = BidModel()

# Load the trained model at startup
if not bid_model.load_model():
    logging.error("Failed to load model at startup!")

def transform_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform the incoming request data to match model features."""
    return {
        "Region": str(data.get("region", "Unknown")),
        "City": str(data.get("city", "Unknown")),
        "Adexchange": str(data.get("adexchange", "Unknown")),
        "Adslotfloorprice": float(data.get("adSlotFloorPrice", 0.0))
    }

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    model_status = "loaded" if bid_model.model is not None else "not loaded"
    return jsonify({
        "status": "Server is running",
        "model_status": model_status
    }), 200

@app.route("/predict_bid", methods=["POST"])
def predict_bid():
    """Receives a bid request in JSON format and returns the bid price."""
    try:
        # Parse JSON request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        logging.info(f"Received bid request: {data}")

        # Transform request data to match model features
        try:
            model_input = transform_request(data)
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

        # Get ML-based bid prediction
        try:
            bid_price_ml = bid_model.predict_bid(model_input)
            
            response = {
                "predicted_bid": round(bid_price_ml, 2),
                "currency": "USD",
                "model_version": "xgboost_v1"
            }

            logging.info(f"Prediction successful: {response}")
            return jsonify(response), 200

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    except Exception as e:
        logging.error(f"Request processing error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/model/reload", methods=["POST"])
def reload_model():
    """Endpoint to reload the model from disk."""
    try:
        success = bid_model.load_model()
        if success:
            return jsonify({"message": "Model reloaded successfully"}), 200
        else:
            return jsonify({"error": "Failed to reload model"}), 500
    except Exception as e:
        return jsonify({"error": f"Error reloading model: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
