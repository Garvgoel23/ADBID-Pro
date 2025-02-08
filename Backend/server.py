from flask import Flask, request, jsonify
from BidRequest import BidRequest
from Bid import Bid
from model import BidModel
from data_processor import load_preprocessed_data

app = Flask(__name__)

# Initialize bidding system and model
bidder = Bid()
bid_model = BidModel()

# Load the trained model at startup
bid_model.load_model()


@app.route("/health", methods=["GET"])
def health_check():
    #Simple health check endpoint.

    return jsonify({"status": "Server is running"}), 200


@app.route("/predict_bid", methods=["POST"])
def predict_bid():
    #Receives a bid request in JSON format and returns the bid price.

    try:
        # Parse JSON request
        data = request.json

        # Create a BidRequest object from JSON
        bid_request = BidRequest()
        bid_request.adSlotWidth = data.get("adSlotWidth", 300)
        bid_request.adSlotHeight = data.get("adSlotHeight", 250)
        bid_request.adSlotFloorPrice = data.get("adSlotFloorPrice", 200)
        bid_request.advertiserId = data.get("advertiserId", 1458)
        bid_request.region = data.get("region", 2)

        # Get bid price (Rule-based or ML-based)
        bid_price_rule_based = bidder.getBidPrice(bid_request)
        bid_price_ml = bid_model.predict_bid(bid_request)

        response = {
            "rule_based_bid": bid_price_rule_based,
            "ml_predicted_bid": bid_price_ml
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process_data", methods=["GET"])
def process_data():
    #Loads and processes historical bid logs.

    try:
        processed_data = load_preprocessed_data()
        return jsonify({"message": "Data processed successfully", "records": len(processed_data)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
