from BidRequest import BidRequest
from Bidder import Bidder
from model import ClickPredictionModel  # Import ML model
import random

class Bid(Bidder):
    def __init__(self, initial_budget=100000):  # Set total budget
        self.base_bid_price = 300  
        self.bid_ratio = 50  
        self.advertiser_bidding_multiplier = {
            1458: 1.0,
            3358: 1.2,
            3386: 1.0,
            3427: 0.9,
            3476: 1.5
        }
        self.budget = initial_budget  # Initialize budget
        self.model = ClickPredictionModel()  # ML Model for predictions
        self.bid_logs = []  # Track bid responses

    def getBidPrice(self, bidRequest: BidRequest) -> int:
        if self.budget <= 0:
            return -1  # Stop bidding when budget is exhausted

        if random.randint(0, 99) >= self.bid_ratio:
            return -1  # No bid decision

        advertiser_id = bidRequest.advertiserID
        floor_price = bidRequest.adslotfloorprice

        bid_multiplier = self.advertiser_bidding_multiplier.get(advertiser_id, 1.0)
        base_bid = int(self.base_bid_price * bid_multiplier)

        # ML-based click probability prediction
        click_prob = self.model.predict_click(bidRequest)
        conversion_prob = self.model.predict_conversion(bidRequest)
        adjusted_bid = int(base_bid * (1 + 2 * click_prob + 5 * conversion_prob))

        final_bid = max(adjusted_bid, floor_price)  # Ensure it's above floor price

        if final_bid > self.budget:
            return -1  # Skip if bid exceeds remaining budget

        self.budget -= final_bid  # Deduct bid amount from budget

        # Log bid details
        self.bid_logs.append({
            "bidID": bidRequest.bidID,
            "advertiserID": advertiser_id,
            "bid_price": final_bid,
            "floor_price": floor_price,
            "remaining_budget": self.budget
        })

        return final_bid
