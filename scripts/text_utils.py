"""
Utility functions for creating rich text representations of restaurants
"""

def create_rich_text(restaurant: dict) -> str:
    """
    Create comprehensive text representation for better semantic search
    
    Includes:
    - Name, categories, location
    - Rating, review count, price
    - Key features (outdoor seating, delivery, etc.)
    - Ambiance and meal types
    """
    parts = []
    
    # 1. Name (most important)
    parts.append(f"Restaurant: {restaurant['name']}")
    
    # 2. Categories/Cuisine
    if restaurant.get('categories'):
        parts.append(f"Cuisine: {restaurant['categories']}")
    
    # 3. Location
    location_parts = []
    if restaurant.get('address'):
        location_parts.append(restaurant['address'])
    if restaurant.get('city'):
        location_parts.append(restaurant['city'])
    if restaurant.get('state'):
        location_parts.append(restaurant['state'])
    if location_parts:
        parts.append(f"Location: {', '.join(location_parts)}")
    
    # 4. Rating & Reviews (high priority)
    if restaurant.get('stars'):
        parts.append(f"Rating: {restaurant['stars']} stars")
    if restaurant.get('review_count'):
        parts.append(f"{restaurant['review_count']} reviews")
    
    # 5. Price Range (high priority)
    attributes = restaurant.get('attributes') or {}  # Handle None attributes
    if attributes.get('RestaurantsPriceRange2'):
        try:
            price_level = int(attributes['RestaurantsPriceRange2'])
            price_text = '$' * price_level
            parts.append(f"Price: {price_text}")
        except (ValueError, TypeError):
            pass
    
    # 6. Key Features (medium priority)
    features = []
    
    if attributes.get('OutdoorSeating') == 'True' or attributes.get('OutdoorSeating') is True:
        features.append("outdoor seating")
    if attributes.get('GoodForKids') == 'True' or attributes.get('GoodForKids') is True:
        features.append("kid-friendly")
    if attributes.get('RestaurantsReservations') == 'True' or attributes.get('RestaurantsReservations') is True:
        features.append("accepts reservations")
    if attributes.get('RestaurantsDelivery') == 'True' or attributes.get('RestaurantsDelivery') is True:
        features.append("delivery available")
    if attributes.get('RestaurantsTakeOut') == 'True' or attributes.get('RestaurantsTakeOut') is True:
        features.append("takeout available")
    if attributes.get('WiFi') and attributes['WiFi'] != 'no':
        features.append("has WiFi")
    
    # Alcohol
    if attributes.get('Alcohol'):
        alcohol = attributes['Alcohol']
        if alcohol and alcohol != 'none':
            features.append(f"serves alcohol")
    
    if features:
        parts.append(f"Features: {', '.join(features)}")
    
    # 7. Good For (meal types) - medium priority
    good_for = attributes.get('GoodForMeal', {})
    if isinstance(good_for, dict):
        meals = [k.lower() for k, v in good_for.items() if v == 'True' or v is True]
        if meals:
            parts.append(f"Good for: {', '.join(meals)}")
    elif isinstance(good_for, str):
        # Sometimes it's a string representation
        if 'breakfast' in good_for.lower():
            parts.append("Good for: breakfast")
        elif 'lunch' in good_for.lower():
            parts.append("Good for: lunch")
        elif 'dinner' in good_for.lower():
            parts.append("Good for: dinner")
    
    # 8. Ambiance - medium priority
    ambience = attributes.get('Ambience', {})
    if isinstance(ambience, dict):
        ambience_types = [k.lower() for k, v in ambience.items() if v == 'True' or v is True]
        if ambience_types:
            parts.append(f"Ambiance: {', '.join(ambience_types)}")
    
    # Join all parts with periods
    return ". ".join(parts) + "."


def get_price_text(restaurant: dict) -> str:
    """Extract price range as text"""
    attributes = restaurant.get('attributes') or {}  # Handle None attributes
    if attributes.get('RestaurantsPriceRange2'):
        try:
            price_level = int(attributes['RestaurantsPriceRange2'])
            return '$' * price_level
        except (ValueError, TypeError):
            return 'Unknown'
    return 'Unknown'
