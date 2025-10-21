#!/usr/bin/env python3
"""
SURG Quick Start Example
=======================

This example demonstrates how to get started with SURG in just a few minutes.
Perfect for first-time users who want to see the system in action quickly.

Flow Diagram:
```
[Load Data] → [Configure SURG] → [Generate Recommendations] → [View Results]
```

Time to complete: ~5 minutes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Import SURG components
from surg import SURG, SURGConfig
from surg.data import UserData, ItemData, InteractionData


def create_sample_data():
    """
    Create sample e-commerce data for demonstration.
    
    Data Structure:
    - Users: demographics and preferences
    - Items: products with features
    - Interactions: purchase/view history
    """
    print("📊 Creating sample data...")
    
    # Create users data
    users_data = {
        'user_id': [f'user_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 100),
        'signup_date': [
            datetime.now() - timedelta(days=np.random.randint(1, 365))
            for _ in range(100)
        ]
    }
    users_df = pd.DataFrame(users_data)
    
    # Create items data (products)
    items_data = {
        'item_id': [f'item_{i}' for i in range(1, 201)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Sports', 'Home'], 200),
        'price': np.random.uniform(10, 500, 200),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD'], 200),
        'rating': np.random.uniform(3.0, 5.0, 200),
        'description': [f'Great product {i} with amazing features' for i in range(1, 201)]
    }
    items_df = pd.DataFrame(items_data)
    
    # Create interactions data
    interactions_data = []
    for _ in range(1000):  # 1000 interactions
        user_id = f"user_{np.random.randint(1, 101)}"
        item_id = f"item_{np.random.randint(1, 201)}"
        interaction_type = np.random.choice(['view', 'purchase', 'cart_add'], p=[0.7, 0.2, 0.1])
        rating = np.random.randint(1, 6) if interaction_type == 'purchase' else None
        timestamp = datetime.now() - timedelta(days=np.random.randint(1, 90))
        
        interactions_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'timestamp': timestamp
        })
    
    interactions_df = pd.DataFrame(interactions_data)
    
    print(f"✅ Created sample data:")
    print(f"   - {len(users_df)} users")
    print(f"   - {len(items_df)} items")
    print(f"   - {len(interactions_df)} interactions")
    
    return users_df, items_df, interactions_df


def basic_recommendations_example():
    """
    Example 1: Basic recommendations without GenAI
    Perfect for testing the core recommendation engine
    """
    print("\n🎯 Example 1: Basic Recommendations")
    print("=" * 50)
    
    # Create sample data
    users_df, items_df, interactions_df = create_sample_data()
    
    # Configure SURG (basic setup)
    config = SURGConfig(
        algorithm="collaborative_filtering",
        min_interactions=2,
        max_recommendations=10
    )
    
    # Initialize SURG
    surg = SURG(config)
    print("✅ SURG initialized with basic configuration")
    
    # Load data
    surg.load_data(
        users_df=users_df,
        items_df=items_df,
        interactions_df=interactions_df
    )
    print("✅ Data loaded successfully")
    
    # Get recommendations for a user
    user_id = "user_1"
    recommendations = surg.recommend(
        user_id=user_id,
        num_recommendations=5
    )
    
    print(f"\n📋 Recommendations for {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['item_id']} (score: {rec['score']:.3f})")
    
    return recommendations


def genai_enhanced_example():
    """
    Example 2: GenAI-Enhanced Recommendations
    Shows how AI can improve recommendation quality and explanations
    """
    print("\n🤖 Example 2: GenAI-Enhanced Recommendations")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️  OpenAI API key not found. Skipping GenAI example.")
        print("   Set OPENAI_API_KEY environment variable to try this feature.")
        return None
    
    # Create sample data
    users_df, items_df, interactions_df = create_sample_data()
    
    # Configure SURG with GenAI
    config = SURGConfig(
        algorithm="hybrid",
        genai_provider="openai",
        openai_api_key=openai_key,
        genai_model="gpt-3.5-turbo",
        enable_explanations=True
    )
    
    # Initialize SURG
    surg = SURG(config)
    print("✅ SURG initialized with GenAI configuration")
    
    # Load data
    surg.load_data(
        users_df=users_df,
        items_df=items_df,
        interactions_df=interactions_df
    )
    print("✅ Data loaded successfully")
    
    # Get AI-enhanced recommendations
    user_id = "user_5"
    recommendations = surg.recommend(
        user_id=user_id,
        num_recommendations=5,
        enhance_with_genai=True,
        include_explanations=True
    )
    
    print(f"\n🎯 AI-Enhanced Recommendations for {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['item_id']} (score: {rec['score']:.3f})")
        if 'explanation' in rec:
            print(f"      💡 {rec['explanation']}")
    
    return recommendations


def personalized_search_example():
    """
    Example 3: Personalized Search with Context
    Shows how to use SURG for search result personalization
    """
    print("\n🔍 Example 3: Personalized Search")
    print("=" * 50)
    
    # Create sample data
    users_df, items_df, interactions_df = create_sample_data()
    
    # Configure SURG
    config = SURGConfig(
        algorithm="content_based",
        use_embeddings=True
    )
    
    surg = SURG(config)
    surg.load_data(users_df, items_df, interactions_df)
    
    # Personalized search
    user_id = "user_10"
    search_query = "electronics smartphone"
    
    results = surg.search(
        user_id=user_id,
        query=search_query,
        personalize=True,
        num_results=5
    )
    
    print(f"🔍 Search results for '{search_query}' (personalized for {user_id}):")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['item_id']} (relevance: {result['relevance_score']:.3f})")
    
    return results


def batch_recommendations_example():
    """
    Example 4: Batch Recommendations for Multiple Users
    Shows how to efficiently generate recommendations for many users
    """
    print("\n📦 Example 4: Batch Recommendations")
    print("=" * 50)
    
    # Create sample data
    users_df, items_df, interactions_df = create_sample_data()
    
    # Configure SURG
    config = SURGConfig(
        algorithm="collaborative_filtering",
        batch_size=50
    )
    
    surg = SURG(config)
    surg.load_data(users_df, items_df, interactions_df)
    
    # Generate batch recommendations
    user_ids = ["user_1", "user_2", "user_3", "user_4", "user_5"]
    
    batch_recommendations = surg.recommend_batch(
        user_ids=user_ids,
        num_recommendations=3
    )
    
    print("📊 Batch recommendations generated:")
    for user_id, recs in batch_recommendations.items():
        print(f"   {user_id}: {len(recs)} recommendations")
        for rec in recs[:2]:  # Show first 2
            print(f"     - {rec['item_id']} (score: {rec['score']:.3f})")
    
    return batch_recommendations


def evaluation_example():
    """
    Example 5: Quick Model Evaluation
    Shows how to evaluate recommendation quality
    """
    print("\n📊 Example 5: Model Evaluation")
    print("=" * 50)
    
    # Create sample data with train/test split
    users_df, items_df, interactions_df = create_sample_data()
    
    # Split data for evaluation
    train_size = int(0.8 * len(interactions_df))
    train_df = interactions_df.iloc[:train_size]
    test_df = interactions_df.iloc[train_size:]
    
    # Configure and train SURG
    config = SURGConfig(algorithm="collaborative_filtering")
    surg = SURG(config)
    surg.load_data(users_df, items_df, train_df)
    
    # Evaluate on test set
    metrics = surg.evaluate(
        test_interactions=test_df,
        k_values=[5, 10],
        metrics=['precision', 'recall', 'ndcg']
    )
    
    print("📈 Evaluation Results:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    return metrics


def main():
    """
    Run all examples to demonstrate SURG capabilities
    """
    print("🚀 SURG Quick Start Examples")
    print("=" * 60)
    print("This script demonstrates core SURG functionality with sample data.")
    print("Each example showcases different features and use cases.\n")
    
    try:
        # Run examples
        basic_recommendations_example()
        genai_enhanced_example()
        personalized_search_example()
        batch_recommendations_example()
        evaluation_example()
        
        print("\n✅ All examples completed successfully!")
        print("\n🎉 Next Steps:")
        print("   1. Try with your own data")
        print("   2. Explore advanced examples in other directories")
        print("   3. Check out production deployment examples")
        print("   4. Read the full documentation")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have installed SURG correctly:")
        print("   pip install surg[all]")


if __name__ == "__main__":
    main()