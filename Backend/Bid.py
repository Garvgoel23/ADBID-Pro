import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .BidRequest import BidRequest
from .Bidder import Bidder
from .model import BidModel
import random

class Bid(Bidder):
    def __init__(self, initial_budget=100000, model_path="bid_model.pkl"):
        self.base_bid_price = 300  
        self.bid_ratio = 50  
        self.advertiser_bidding_multiplier = {
            1458: 1.0,
            3358: 1.2,
            3386: 1.0,
            3427: 0.9,
            3476: 1.5
        }
        self.budget = initial_budget
        self.model = BidModel(model_path=model_path)
        self.bid_logs = []

    # Your existing methods here...

def main():
    print("Initializing Bid system...")
    bidder = Bid()
    
    # Create a test request
    test_request = BidRequest(
        adSlotWidth=300,
        adSlotHeight=250,
        adSlotFloorPrice=10,
        advertiserId=1458,
        region="NA"
    )
    
    # Test the bidding
    print("Testing bid generation...")
    bid_price = bidder.getBidPrice(test_request)
    print(f"Generated bid price: {bid_price}")

if __name__ == "__main__":
    main()
