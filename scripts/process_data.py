import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from tqdm import tqdm
from collections import defaultdict

def load_businesses(raw_path: str = "data/raw/yelp_academic_dataset_business.json"):
    businesses = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading businesses"):
            businesses.append(json.loads(line))
    return businesses

def load_photos(raw_path: str = "data/raw/photos.json"):
    photos = []
    if not Path(raw_path).exists():
        print(f"⚠️  Photos file not found: {raw_path}")
        return photos
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading photos"):
            photos.append(json.loads(line))
    return photos

def filter_restaurants(businesses: list):
    restaurant_keywords = [
        'restaurant', 'food', 'cafe', 'bar', 'pizza', 'burger',
        'sushi', 'mexican', 'italian', 'chinese', 'japanese',
        'thai', 'indian', 'korean', 'vietnamese', 'american',
        'breakfast', 'brunch', 'lunch', 'dinner', 'bakery',
        'coffee', 'tea', 'sandwiches', 'deli', 'grill'
    ]
    
    restaurants = []
    for business in tqdm(businesses, desc="Filtering restaurants"):
        categories = business.get('categories', '')
        if not categories:
            continue
        
        categories_lower = categories.lower()
        if any(keyword in categories_lower for keyword in restaurant_keywords):
            if business.get('is_open', 0) == 1:
                restaurants.append(business)
    
    return restaurants

def create_photo_mapping(photos: list):
    business_photos = defaultdict(list)
    
    for photo in tqdm(photos, desc="Mapping photos"):
        business_id = photo['business_id']
        photo_path = f"data/raw/photos/{photo['photo_id']}.jpg"
        
        if Path(photo_path).exists():
            business_photos[business_id].append({
                'photo_id': photo['photo_id'],
                'path': photo_path,
                'label': photo.get('label', ''),
                'caption': photo.get('caption', '')
            })
    
    return business_photos

def process_restaurants(restaurants: list, business_photos: dict):
    processed = []
    
    for restaurant in tqdm(restaurants, desc="Processing restaurants"):
        processed_restaurant = {
            'business_id': restaurant['business_id'],
            'name': restaurant['name'],
            'address': restaurant.get('address', ''),
            'city': restaurant.get('city', ''),
            'state': restaurant.get('state', ''),
            'postal_code': restaurant.get('postal_code', ''),
            'latitude': restaurant.get('latitude'),
            'longitude': restaurant.get('longitude'),
            'stars': restaurant.get('stars', 0),
            'review_count': restaurant.get('review_count', 0),
            'categories': restaurant.get('categories', ''),
            'attributes': restaurant.get('attributes', {}),
            'hours': restaurant.get('hours', {}),
            'photos': business_photos.get(restaurant['business_id'], [])
        }
        
        processed.append(processed_restaurant)
    
    return processed

def save_processed_data(restaurants: list, output_path: str = "data/processed/restaurants.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(restaurants, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(restaurants)} restaurants to {output_path}")

def create_sample_dataset(restaurants: list, sample_size: int = 500):
    import random
    
    sample = random.sample(restaurants, min(sample_size, len(restaurants)))
    save_processed_data(sample, "data/processed/restaurants_sample.json")
    print(f"✓ Created sample dataset with {len(sample)} restaurants")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Yelp dataset')
    parser.add_argument('--sample', type=int, help='Create sample dataset with N restaurants')
    args = parser.parse_args()
    
    print("="*60)
    print("Yelp Dataset Processing")
    print("="*60 + "\n")
    
    print("Step 1: Loading raw data...")
    businesses = load_businesses()
    photos = load_photos()
    
    print(f"\nLoaded:")
    print(f"  - Businesses: {len(businesses):,}")
    print(f"  - Photos: {len(photos):,}")
    
    print("\nStep 2: Filtering restaurants...")
    restaurants = filter_restaurants(businesses)
    print(f"  - Restaurants: {len(restaurants):,}")
    
    print("\nStep 3: Mapping photos to businesses...")
    business_photos = create_photo_mapping(photos)
    restaurants_with_photos = sum(1 for r in restaurants if r['business_id'] in business_photos)
    print(f"  - Restaurants with photos: {restaurants_with_photos:,}")
    
    print("\nStep 4: Processing restaurant data...")
    processed_restaurants = process_restaurants(restaurants, business_photos)
    
    print("\nStep 5: Saving processed data...")
    save_processed_data(processed_restaurants)
    
    if args.sample:
        print(f"\nStep 6: Creating sample dataset ({args.sample} restaurants)...")
        create_sample_dataset(processed_restaurants, args.sample)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Total restaurants: {len(processed_restaurants):,}")
    print(f"  With photos: {restaurants_with_photos:,}")
    print(f"  Average rating: {sum(r['stars'] for r in processed_restaurants) / len(processed_restaurants):.2f}")
    print(f"\nOutput:")
    print(f"  - data/processed/restaurants.json")
    if args.sample:
        print(f"  - data/processed/restaurants_sample.json")

if __name__ == "__main__":
    main()
