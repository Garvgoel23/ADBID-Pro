from abc import ABC, abstractmethod
from BidRequest import BidRequest

class Bidder(ABC):
   # Abstract base class for bidders. Any custom bidder must implement the get_bid_price method.
    
    @abstractmethod
    def get_bid_price(self, bid_request: BidRequest) -> int:
       #:return: The bid price as an integer.

        pass
