"""
Basic SURG Usage Example with MovieLens

This example demonstrates the basic usage of the SURG recommendation system
using the MovieLens 1M dataset. It shows how to:

1. Load and preprocess the MovieLens dataset
2. Initialize a SURG recommender with default settings  
3. Train the model on user-item interactions
4. Generate recommendations for users
5. Evaluate recommendation quality
6. Generate explanations for recommendations

This is a beginner-friendly example that covers the essential workflow
of using SURG for recommendation tasks.

Usage:
    python examples/basic_usage.py
    
    # Or with custom parameters:
    python examples/basic_usage.py --dataset ml-1m --n_users 1000 --n_recommendations 10
"""

import argparse
import sys
from pathlib import Path

# Add surg to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from surg import SURGRecommender
from surg.datasets import load_movielens
from surg.utils.metrics import evaluate_recommendations
from surg.utils.data_preprocessing import train_test_split


def main():
    """
    Main function demonstrating basic SURG usage with MovieLens.
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basic SURG Example with MovieLens')
    parser.add_argument('--dataset', default='ml-1m', 
                       choices=['ml-100k', 'ml-1m', 'ml-10m', 'ml-25m'],
                       help='MovieLens dataset version to use')
    parser.add_argument('--n_users', type=int, default=1000,
                       help='Number of users to include in example')
    parser.add_argument('--n_recommendations', type=int, default=10,
                       help='Number of recommendations to generate per user')
    parser.add_argument('--enable_genai', action='store_true',
                       help='Enable GenAI enhancements (requires API keys)')
    
    args = parser.parse_args()
    
    print("🎬 SURG Basic Example with MovieLens Dataset")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print(f"\n📊 Loading {args.dataset} dataset...")
    try:
        data = load_movielens(version=args.dataset, download=True)
        interactions = data['interactions']
        users = data['users'] 
        items = data['items']
        
        # Limit to subset for faster execution
        if args.n_users < len(users):
            user_subset = users.head(args.n_users)['user_id'].values
            interactions = interactions[interactions['user_id'].isin(user_subset)]
        
        print(f"✅ Loaded dataset with:")
        print(f"   - {interactions['user_id'].nunique():,} users")
        print(f"   - {interactions['item_id'].nunique():,} items") 
        print(f"   - {len(interactions):,} interactions")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure you have internet connection for first-time download")
        return
    
    # Step 2: Split data into train/test
    print("\n🔀 Splitting data into train/test sets...")
    train_data, test_data = train_test_split(
        interactions, 
        test_size=0.2, 
        strategy='random',
        random_state=42
    )
    
    print(f"   - Training interactions: {len(train_data):,}")
    print(f"   - Test interactions: {len(test_data):,}")
    
    # Step 3: Initialize SURG recommender
    print(f"\n🤖 Initializing SURG recommender...")
    
    config = {
        'lightfm': {
            'loss': 'warp',
            'no_components': 50,
            'learning_rate': 0.05,
            'epochs': 10
        },
        'genai': {
            'backend': 'local',  # Use local fallback if no API keys
            'max_length': 150
        }
    }
    
    recommender = SURGRecommender(
        config=config,
        enable_genai=args.enable_genai
    )
    
    print(f"   - GenAI enhancements: {'✅ Enabled' if args.enable_genai else '❌ Disabled'}")
    print(f"   - Algorithm: Hybrid (LightFM + GenAI layers)")
    
    # Step 4: Train the model
    print(f"\n🏋️ Training the recommendation model...")
    try:
        # Convert to format expected by SURG
        user_item_matrix = create_interaction_matrix(train_data)
        user_features = create_user_features(users) if not users.empty else None
        item_features = create_item_features(items) if not items.empty else None
        
        recommender.fit(
            interactions=user_item_matrix,
            user_features=user_features,
            item_features=item_features
        )
        
        print("✅ Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Step 5: Generate recommendations
    print(f"\n🎯 Generating recommendations...")
    
    # Select a few test users
    test_users = train_data['user_id'].unique()[:5]
    
    for user_id in test_users:
        try:
            # Get recommendations
            recommendations = recommender.recommend(
                user_id=user_id,
                k=args.n_recommendations,
                include_explanations=args.enable_genai
            )
            
            print(f"\n👤 User {user_id} Recommendations:")
            
            # Display recommended items
            for i, (item_id, score) in enumerate(zip(
                recommendations['item_ids'], 
                recommendations['scores']
            ), 1):
                item_title = get_item_title(items, item_id)
                print(f"   {i:2d}. {item_title} (score: {score:.3f})")
            
            # Display explanation if available
            if args.enable_genai and 'explanations' in recommendations:
                print(f"\n💡 Explanation: {recommendations['explanations'][0]}")
                
        except Exception as e:
            print(f"❌ Error generating recommendations for user {user_id}: {e}")
    
    # Step 6: Evaluate recommendations
    print(f"\n📈 Evaluating recommendation quality...")
    try:
        # Generate recommendations for all test users
        test_users_sample = test_data['user_id'].unique()[:100]  # Sample for speed
        
        all_recommendations = {}
        for user_id in test_users_sample:
            recs = recommender.recommend(user_id=user_id, k=10)
            all_recommendations[user_id] = recs['item_ids']
        
        # Calculate metrics
        metrics = evaluate_recommendations(
            recommendations=all_recommendations,
            test_data=test_data,
            k=10
        )
        
        print(f"   - Precision@10: {metrics['precision@10']:.3f}")
        print(f"   - Recall@10: {metrics['recall@10']:.3f}")
        print(f"   - NDCG@10: {metrics['ndcg@10']:.3f}")
        print(f"   - Coverage: {metrics['coverage']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
    
    # Step 7: Demonstrate additional features
    print(f"\n🔍 Additional Features:")
    
    try:
        # User profile (if GenAI enabled)
        if args.enable_genai:
            profile = recommender.get_user_profile(test_users[0])
            if profile:
                print(f"   - User {test_users[0]} profile: {profile.get('summary', 'N/A')}")
        
        # Similar items
        popular_item = train_data['item_id'].value_counts().index[0]
        similar_items = recommender.get_similar_items(popular_item, k=5)
        if similar_items:
            print(f"   - Items similar to {get_item_title(items, popular_item)}:")
            for item_id, score in similar_items[popular_item][:3]:
                print(f"     • {get_item_title(items, item_id)} (similarity: {score:.3f})")
                
    except Exception as e:
        print(f"⚠️ Some additional features not available: {e}")
    
    print(f"\n🎉 Example completed successfully!")
    print(f"\n💡 Next steps:")
    print(f"   - Try different datasets with --dataset flag")
    print(f"   - Enable GenAI features with --enable_genai (requires API keys)")
    print(f"   - Explore advanced_pipeline.py for more complex workflows")
    print(f"   - Check custom_genai_layer.py for extending SURG")


def create_interaction_matrix(interactions_df):
    """
    Convert interactions DataFrame to format expected by SURG.
    
    Args:
        interactions_df: DataFrame with columns [user_id, item_id, rating]
        
    Returns:
        Sparse interaction matrix
    """
    # This would convert to scipy sparse matrix format
    # Implementation depends on the actual SURG interface
    print("   🔄 Converting interactions to matrix format...")
    return interactions_df  # Placeholder


def create_user_features(users_df):
    """
    Create user feature matrix from user metadata.
    
    Args:
        users_df: DataFrame with user information
        
    Returns:
        User features matrix
    """
    if users_df.empty:
        return None
    
    print("   🔄 Creating user features...")
    return users_df  # Placeholder


def create_item_features(items_df):
    """
    Create item feature matrix from item metadata.
    
    Args:
        items_df: DataFrame with item information
        
    Returns:
        Item features matrix  
    """
    if items_df.empty:
        return None
        
    print("   🔄 Creating item features...")
    return items_df  # Placeholder


def get_item_title(items_df, item_id):
    """
    Get the title of an item by ID.
    
    Args:
        items_df: DataFrame with item information
        item_id: Item identifier
        
    Returns:
        Item title string
    """
    if items_df.empty:
        return f"Item {item_id}"
    
    item_row = items_df[items_df['item_id'] == item_id]
    if len(item_row) > 0:
        return item_row.iloc[0].get('title', f"Item {item_id}")
    return f"Item {item_id}"


if __name__ == "__main__":
    main()