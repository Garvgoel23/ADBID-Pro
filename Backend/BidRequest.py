class BidRequest:
    def __init__(self, bid_id=None, timestamp=None, visitor_id=None, user_agent=None, ip_address=None,
                 region=None, city=None, ad_exchange=None, domain=None, url=None, anonymous_url_id=None,
                 ad_slot_id=None, ad_slot_width=None, ad_slot_height=None, ad_slot_visibility=None,
                 ad_slot_format=None, ad_slot_floor_price=None, creative_id=None, advertiser_id=None,
                 user_tags=None):
        self.bid_id = bid_id
        self.timestamp = timestamp
        self.visitor_id = visitor_id
        self.user_agent = user_agent
        self.ip_address = ip_address
        self.region = region
        self.city = city
        self.ad_exchange = ad_exchange
        self.domain = domain
        self.url = url
        self.anonymous_url_id = anonymous_url_id
        self.ad_slot_id = ad_slot_id
        self.ad_slot_width = ad_slot_width
        self.ad_slot_height = ad_slot_height
        self.ad_slot_visibility = ad_slot_visibility
        self.ad_slot_format = ad_slot_format
        self.ad_slot_floor_price = ad_slot_floor_price
        self.creative_id = creative_id
        self.advertiser_id = advertiser_id
        self.user_tags = user_tags

    def to_dict(self):
        #Converts the object to a dictionary for easy JSON serialization.
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        #Creates a BidRequest instance from a dictionary.
        return cls(**data)
