import json

def extract_attributes(restaurant: dict) -> dict:
    """Extract and normalize restaurant attributes."""
    attrs = restaurant.get('attributes') or {}
    
    # Helper to parse string booleans
    def parse_bool(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value_lower = value.lower().strip("'\"u ")
            if value_lower in ('true', '1'):
                return True
            if value_lower in ('false', '0', 'none'):
                return False
        return None
    
    # Helper to parse price range
    def parse_price(value):
        if value is None:
            return None
        try:
            return int(value)
        except:
            return None
    
    # Helper to parse parking (often a JSON string)
    def parse_parking(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                # Remove u' prefix if present
                value = value.replace("u'", "'")
                return eval(value)  # Safe here since it's from trusted dataset
            except:
                return None
        return None
    
    # Helper to clean string values
    def clean_string(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip("'\"u ")
        return value
    
    return {
        'price_range': parse_price(attrs.get('RestaurantsPriceRange2')),
        'takes_reservations': parse_bool(attrs.get('RestaurantsReservations')),
        'delivery': parse_bool(attrs.get('RestaurantsDelivery')),
        'takeout': parse_bool(attrs.get('RestaurantsTakeOut')),
        'outdoor_seating': parse_bool(attrs.get('OutdoorSeating')),
        'good_for_kids': parse_bool(attrs.get('GoodForKids')),
        'wifi': clean_string(attrs.get('WiFi')),
        'alcohol': clean_string(attrs.get('Alcohol')),
        'parking': parse_parking(attrs.get('BusinessParking')),
        'wheelchair_accessible': parse_bool(attrs.get('WheelchairAccessible')),
        'caters': parse_bool(attrs.get('Caters')),
        'has_tv': parse_bool(attrs.get('HasTV')),
        'noise_level': clean_string(attrs.get('NoiseLevel')),
        'attire': clean_string(attrs.get('RestaurantsAttire')),
        'good_for_groups': parse_bool(attrs.get('RestaurantsGoodForGroups')),
    }

def create_text_metadata(restaurant: dict) -> dict:
    """Create comprehensive metadata for text index."""
    attrs = extract_attributes(restaurant)
    
    return {
        # Core
        'id': restaurant['business_id'],
        'name': restaurant['name'],
        'categories': restaurant.get('categories', ''),
        'rating': restaurant.get('stars', 0),
        'review_count': restaurant.get('review_count', 0),
        'is_open': restaurant.get('is_open', 0),
        
        # Location
        'latitude': restaurant.get('latitude'),
        'longitude': restaurant.get('longitude'),
        'address': restaurant.get('address', ''),
        'city': restaurant.get('city', ''),
        'state': restaurant.get('state', ''),
        'postal_code': restaurant.get('postal_code', ''),
        
        # Attributes
        **attrs,
        
        # Hours
        'hours': restaurant.get('hours', {})
    }

def create_image_metadata(restaurant: dict, photo: dict) -> dict:
    """Create metadata for image index (no location data - use text search for location filtering)."""
    attrs = extract_attributes(restaurant)
    
    return {
        # Core
        'id': restaurant['business_id'],
        'name': restaurant['name'],
        'photo_id': photo['photo_id'],
        'label': photo.get('label', ''),
        'categories': restaurant.get('categories', ''),
        'rating': restaurant.get('stars', 0),
        'review_count': restaurant.get('review_count', 0),
        'is_open': restaurant.get('is_open', 0),
        
        # Attributes
        **attrs,
        
        # Hours
        'hours': restaurant.get('hours', {})
    }
