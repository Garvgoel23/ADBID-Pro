from abc import ABC, abstractmethod
from BidRequest import BidRequest
from typing import Union, Dict, Any

class Bidder(ABC):
    
    @abstractmethod
    def getBidPrice(self, bid_request: BidRequest) -> int:
        
        pass
    
    def get_bid_price(self, bid_request: BidRequest) -> int:
        
        return self.getBidPrice(bid_request)
