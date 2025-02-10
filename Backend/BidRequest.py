from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class BidRequest:
    """
    Represents a bid request with all necessary auction information.
    Uses dataclass for cleaner initialization and serialization.
    """
    # Required fields (matches your existing server validation)
    adSlotWidth: int
    adSlotHeight: int
    adSlotFloorPrice: float
    advertiserId: int
    region: str
    
    # Optional fields with default values
    bidID: Optional[str] = None
    timestamp: Optional[datetime] = None
    visitorID: Optional[str] = None
    userAgent: Optional[str] = None
    ipAddress: Optional[str] = None
    city: Optional[str] = None
    adExchange: Optional[str] = None
    domain: Optional[str] = None
    url: Optional[str] = None
    anonymousURLID: Optional[str] = None
    adSlotID: Optional[str] = None
    adSlotVisibility: Optional[str] = None
    adSlotFormat: Optional[str] = None
    creativeID: Optional[str] = None
    userTags: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and convert types after initialization"""
        # Convert string numbers to proper types if needed
        if isinstance(self.adSlotWidth, str):
            self.adSlotWidth = int(self.adSlotWidth)
        if isinstance(self.adSlotHeight, str):
            self.adSlotHeight = int(self.adSlotHeight)
        if isinstance(self.adSlotFloorPrice, str):
            self.adSlotFloorPrice = float(self.adSlotFloorPrice)
        if isinstance(self.advertiserId, str):
            self.advertiserId = int(self.advertiserId)
            
        # Convert timestamp string to datetime if needed
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.strptime(self.timestamp, "%Y%m%d%H%M%S%f")
            except ValueError:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary for JSON serialization.
        Handles datetime conversion.
        """
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.strftime("%Y%m%d%H%M%S%f")
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BidRequest':
        """
        Creates a BidRequest instance from a dictionary.
        Handles both snake_case and camelCase keys for compatibility.
        """
        # Map snake_case to camelCase for compatibility
        key_mapping = {
            'bid_id': 'bidID',
            'visitor_id': 'visitorID',
            'user_agent': 'userAgent',
            'ip_address': 'ipAddress',
            'ad_exchange': 'adExchange',
            'anonymous_url_id': 'anonymousURLID',
            'ad_slot_id': 'adSlotID',
            'ad_slot_width': 'adSlotWidth',
            'ad_slot_height': 'adSlotHeight',
            'ad_slot_visibility': 'adSlotVisibility',
            'ad_slot_format': 'adSlotFormat',
            'ad_slot_floor_price': 'adSlotFloorPrice',
            'creative_id': 'creativeID',
            'advertiser_id': 'advertiserId',
            'user_tags': 'userTags'
        }
        
        # Convert keys to camelCase
        converted_data = {}
        for key, value in data.items():
            new_key = key_mapping.get(key, key)
            converted_data[new_key] = value
            
        return cls(**converted_data)

    # Backwards compatibility properties
    @property
    def bid_id(self) -> Optional[str]:
        return self.bidID
        
    @property
    def ad_slot_width(self) -> int:
        return self.adSlotWidth
        
    @property
    def ad_slot_height(self) -> int:
        return self.adSlotHeight
        
    @property
    def ad_slot_floor_price(self) -> float:
        return self.adSlotFloorPrice
        
    @property
    def advertiser_id(self) -> int:
        return self.advertiserId
