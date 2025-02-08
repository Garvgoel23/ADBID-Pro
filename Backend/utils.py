import json
import logging
import os
import numpy as np

# Configure logging
logging.basicConfig(
    filename="backend/logs/server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_message(level, message):
    #Logs a message to the server.log file.

    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)

def normalize(value, min_val, max_val):
    #Normalizes a value to the range [0,1].
    
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

def extract_features(bid_request):
    
    try:
        features = {
            "ad_slot_width": bid_request.adSlotWidth,
            "ad_slot_height": bid_request.adSlotHeight,
            "ad_slot_floor_price": bid_request.adSlotFloorPrice,
            "advertiser_id": bid_request.advertiserId,
            "region": bid_request.region,
        }
        return features
    except Exception as e:
        log_message("ERROR", f"Feature extraction failed: {str(e)}")
        return None

def save_json(data, filename):
    
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        log_message("INFO", f"Successfully saved data to {filename}")
    except Exception as e:
        log_message("ERROR", f"Failed to save JSON: {str(e)}")

def load_json(filename):
    
    try:
        if not os.path.exists(filename):
            log_message("WARNING", f"File {filename} not found.")
            return None
        
        with open(filename, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        log_message("ERROR", f"Failed to load JSON: {str(e)}")
        return None
