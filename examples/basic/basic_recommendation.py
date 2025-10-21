#!/usr/bin/env python3
"""
Basic Recommendation Example
===========================

This example demonstrates the core recommendation functionality of SURG
without GenAI enhancement. Perfect for understanding the fundamental
recommendation algorithms and data flow.

Learning Objectives:
- Load and prepare data for SURG
- Configure basic recommendation algorithms
- Generate and interpret recommendations
- Understand recommendation scoring

Time to complete: ~10 minutes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Import SURG components
from surg import SURG, SURGConfig
from surg.data import UserData, ItemData, InteractionData


class BasicRecommendationDemo:
    """
    Comprehensive demonstration of basic recommendation functionality.
    
    This class walks through different recommendation algorithms and 
    shows how to interpret results.
    """
    
    def __init__(self):
        self.users_df = None
        self.items_df = None
        self.interactions_df = None
        self.surg = None
        
    def load_sample_data(self) -> None:
        """
        Load sample movie recommendation data.
        
        Data Schema:
        - Users: user_id, age, gender, occupation
        - Items: item_id, title, genre, year, director
        - Interactions: user_id, item_id, rating, timestamp
        """
        print("📊 Loading sample movie recommendation data...")
        
        # Create users (movie watchers)
        np.random.seed(42)  # For reproducible results
        
        users_data = {
            'user_id': [f'user_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 65, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'occupation': np.random.choice([
                'student', 'engineer', 'teacher', 'doctor', 'artist', 
                'manager', 'retired', 'other'
            ], 100),
            'zip_code': np.random.randint(10000, 99999, 100)
        }
        self.users_df = pd.DataFrame(users_data)
        
        # Create movies (items)
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        directors = ['Spielberg', 'Nolan', 'Tarantino', 'Scorsese', 'Hitchcock', 'Lucas']
        
        items_data = {
            'item_id': [f'movie_{i}' for i in range(1, 201)],
            'title': [f'Movie Title {i}' for i in range(1, 201)],
            'genre': np.random.choice(genres, 200),
            'year': np.random.randint(1990, 2024, 200),
            'director': np.random.choice(directors, 200),
            'duration': np.random.randint(90, 180, 200),  # minutes
            'budget': np.random.uniform(1, 200, 200),  # millions
        }
        self.items_df = pd.DataFrame(items_data)
        
        # Create realistic interactions (ratings)
        interactions_data = []
        
        # Generate interactions with some patterns
        for user_idx in range(100):
            user_id = f"user_{user_idx + 1}"
            user_age = self.users_df.iloc[user_idx]['age']
            user_gender = self.users_df.iloc[user_idx]['gender']
            
            # Number of movies rated by this user (some users more active)
            num_ratings = np.random.poisson(15) + 5  # 5-30 ratings per user
            
            # Select random movies to rate
            movie_indices = np.random.choice(200, min(num_ratings, 200), replace=False)
            
            for movie_idx in movie_indices:
                movie_id = f"movie_{movie_idx + 1}"
                movie_genre = self.items_df.iloc[movie_idx]['genre']
                movie_year = self.items_df.iloc[movie_idx]['year']
                
                # Generate realistic rating based on user preferences
                base_rating = 3.0
                
                # Age preferences
                if user_age < 30 and movie_genre in ['Action', 'Sci-Fi']:
                    base_rating += 0.5
                elif user_age > 50 and movie_genre in ['Drama', 'Romance']:
                    base_rating += 0.5
                
                # Gender preferences (stereotypical for demo)
                if user_gender == 'M' and movie_genre in ['Action', 'Sci-Fi']:
                    base_rating += 0.3
                elif user_gender == 'F' and movie_genre in ['Romance', 'Drama']:
                    base_rating += 0.3
                
                # Recent movies slightly preferred
                if movie_year > 2015:
                    base_rating += 0.2
                
                # Add noise and clamp to 1-5 range
                rating = base_rating + np.random.normal(0, 0.8)
                rating = max(1, min(5, round(rating)))
                
                # Generate realistic timestamp
                timestamp = datetime.now() - timedelta(
                    days=np.random.randint(1, 365)
                )
                
                interactions_data.append({
                    'user_id': user_id,
                    'item_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        self.interactions_df = pd.DataFrame(interactions_data)
        
        print(f"✅ Sample data created:")
        print(f"   - {len(self.users_df)} users")
        print(f"   - {len(self.items_df)} movies")
        print(f"   - {len(self.interactions_df)} ratings")
        print(f"   - Average ratings per user: {len(self.interactions_df) / len(self.users_df):.1f}")
        print(f"   - Rating distribution: {self.interactions_df['rating'].value_counts().sort_index().to_dict()}")
    
    def demonstrate_collaborative_filtering(self) -> Dict[str, Any]:
        """
        Demonstrate collaborative filtering recommendations.
        
        Collaborative filtering finds users with similar preferences
        and recommends items liked by similar users.
        """
        print("\n🤝 Collaborative Filtering Example")
        print("=" * 50)
        print("Finds users with similar tastes and recommends items they liked.")
        
        # Configure SURG for collaborative filtering
        config = SURGConfig(
            algorithm="collaborative_filtering",
            min_interactions=3,  # Minimum interactions to consider a user
            similarity_metric="cosine",  # cosine, pearson, jaccard
            num_neighbors=10,  # Number of similar users to consider
            max_recommendations=10
        )
        
        self.surg = SURG(config)
        
        # Load data
        self.surg.load_data(
            users_df=self.users_df,
            items_df=self.items_df,
            interactions_df=self.interactions_df
        )
        
        print("✅ Data loaded for collaborative filtering")
        
        # Get recommendations for multiple users
        test_users = ["user_1", "user_5", "user_10"]
        all_recommendations = {}
        
        for user_id in test_users:
            print(f"\n📋 Collaborative Filtering Recommendations for {user_id}:")
            
            # Get user's rating history
            user_history = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ].sort_values('rating', ascending=False)
            
            print(f"   User's top-rated movies:")
            for _, interaction in user_history.head(3).iterrows():
                movie_info = self.items_df[
                    self.items_df['item_id'] == interaction['item_id']
                ].iloc[0]
                print(f"     - {movie_info['title']} ({movie_info['genre']}) - Rating: {interaction['rating']}")
            
            # Generate recommendations
            recommendations = self.surg.recommend(
                user_id=user_id,
                num_recommendations=5,
                exclude_seen=True  # Don't recommend already seen movies
            )
            
            print(f"   Recommended movies:")
            for i, rec in enumerate(recommendations, 1):
                movie_info = self.items_df[
                    self.items_df['item_id'] == rec['item_id']
                ].iloc[0]
                print(f"     {i}. {movie_info['title']} ({movie_info['genre']}) - Score: {rec['score']:.3f}")
            
            all_recommendations[user_id] = recommendations
        
        return all_recommendations
    
    def demonstrate_content_based_filtering(self) -> Dict[str, Any]:
        """
        Demonstrate content-based filtering recommendations.
        
        Content-based filtering recommends items similar to those
        the user has liked based on item features.
        """
        print("\n🎬 Content-Based Filtering Example")
        print("=" * 50)
        print("Recommends movies similar to those you've liked based on genre, director, etc.")
        
        # Configure SURG for content-based filtering
        config = SURGConfig(
            algorithm="content_based",
            content_features=["genre", "director", "year"],
            similarity_metric="cosine",
            feature_weights={
                "genre": 0.5,
                "director": 0.3,
                "year": 0.2
            },
            max_recommendations=10
        )
        
        self.surg = SURG(config)
        
        # Load data
        self.surg.load_data(
            users_df=self.users_df,
            items_df=self.items_df,
            interactions_df=self.interactions_df
        )
        
        print("✅ Data loaded for content-based filtering")
        
        # Get recommendations
        test_users = ["user_2", "user_7", "user_15"]
        all_recommendations = {}
        
        for user_id in test_users:
            print(f"\n📋 Content-Based Recommendations for {user_id}:")
            
            # Analyze user's preferences
            user_history = self.interactions_df[
                (self.interactions_df['user_id'] == user_id) & 
                (self.interactions_df['rating'] >= 4)  # Only high ratings
            ]
            
            # Get preferred genres
            user_movies = user_history.merge(self.items_df, on='item_id')
            genre_preferences = user_movies['genre'].value_counts()
            director_preferences = user_movies['director'].value_counts()
            
            print(f"   User's preferred genres: {dict(genre_preferences.head(3))}")
            print(f"   User's preferred directors: {dict(director_preferences.head(2))}")
            
            # Generate recommendations
            recommendations = self.surg.recommend(
                user_id=user_id,
                num_recommendations=5,
                exclude_seen=True
            )
            
            print(f"   Recommended movies:")
            for i, rec in enumerate(recommendations, 1):
                movie_info = self.items_df[
                    self.items_df['item_id'] == rec['item_id']
                ].iloc[0]
                print(f"     {i}. {movie_info['title']} ({movie_info['genre']}, {movie_info['director']}) - Score: {rec['score']:.3f}")
            
            all_recommendations[user_id] = recommendations
        
        return all_recommendations
    
    def demonstrate_hybrid_recommendations(self) -> Dict[str, Any]:
        """
        Demonstrate hybrid recommendations combining multiple algorithms.
        
        Hybrid approaches combine collaborative and content-based filtering
        to get the best of both worlds.
        """
        print("\n🔀 Hybrid Recommendations Example")
        print("=" * 50)
        print("Combines collaborative and content-based filtering for better recommendations.")
        
        # Configure SURG for hybrid approach
        config = SURGConfig(
            algorithm="hybrid",
            hybrid_weights={
                "collaborative_filtering": 0.6,
                "content_based": 0.4
            },
            min_interactions=2,
            content_features=["genre", "director"],
            max_recommendations=10
        )
        
        self.surg = SURG(config)
        
        # Load data
        self.surg.load_data(
            users_df=self.users_df,
            items_df=self.items_df,
            interactions_df=self.interactions_df
        )
        
        print("✅ Data loaded for hybrid recommendations")
        
        # Compare different approaches for the same user
        test_user = "user_3"
        print(f"\n📊 Comparing recommendation approaches for {test_user}:")
        
        # Get hybrid recommendations
        recommendations = self.surg.recommend(
            user_id=test_user,
            num_recommendations=8,
            exclude_seen=True,
            return_explanations=True  # Get explanation of why recommended
        )
        
        print(f"   Hybrid recommendations:")
        for i, rec in enumerate(recommendations, 1):
            movie_info = self.items_df[
                self.items_df['item_id'] == rec['item_id']
            ].iloc[0]
            print(f"     {i}. {movie_info['title']} ({movie_info['genre']}) - Score: {rec['score']:.3f}")
            if 'explanation' in rec:
                print(f"        💡 {rec['explanation']}")
        
        return {test_user: recommendations}
    
    def analyze_recommendation_quality(self) -> Dict[str, float]:
        """
        Analyze the quality of recommendations using various metrics.
        
        Computes diversity, novelty, and coverage metrics to understand
        recommendation quality beyond just accuracy.
        """
        print("\n📈 Recommendation Quality Analysis")
        print("=" * 50)
        
        if self.surg is None:
            print("❌ No SURG instance available. Run a recommendation example first.")
            return {}
        
        # Generate recommendations for multiple users
        test_users = [f"user_{i}" for i in range(1, 21)]  # 20 users
        all_recommendations = []
        
        for user_id in test_users:
            try:
                recs = self.surg.recommend(
                    user_id=user_id,
                    num_recommendations=10,
                    exclude_seen=True
                )
                all_recommendations.extend([rec['item_id'] for rec in recs])
            except Exception as e:
                print(f"   ⚠️  Could not generate recommendations for {user_id}: {e}")
        
        if not all_recommendations:
            print("❌ No recommendations generated for analysis.")
            return {}
        
        # Calculate metrics
        metrics = {}
        
        # 1. Coverage: What percentage of items are recommended?
        unique_recommended = set(all_recommendations)
        total_items = set(self.items_df['item_id'])
        coverage = len(unique_recommended) / len(total_items)
        metrics['coverage'] = coverage
        
        # 2. Diversity: How diverse are the recommendations?
        # Genre diversity
        recommended_movies = self.items_df[
            self.items_df['item_id'].isin(unique_recommended)
        ]
        genre_distribution = recommended_movies['genre'].value_counts(normalize=True)
        genre_entropy = -sum(p * np.log2(p) for p in genre_distribution if p > 0)
        max_entropy = np.log2(len(genre_distribution))
        diversity = genre_entropy / max_entropy if max_entropy > 0 else 0
        metrics['diversity'] = diversity
        
        # 3. Novelty: How often are popular items recommended?
        item_popularity = self.interactions_df['item_id'].value_counts()
        popular_items = set(item_popularity.head(20).index)  # Top 20 popular items
        popular_recommended = len(unique_recommended.intersection(popular_items))
        novelty = 1 - (popular_recommended / len(unique_recommended))
        metrics['novelty'] = novelty
        
        # 4. Average recommendation score
        avg_score = np.mean([
            np.mean([rec['score'] for rec in self.surg.recommend(f"user_{i}", 5, exclude_seen=True)])
            for i in range(1, 11)
        ])
        metrics['avg_score'] = avg_score
        
        print("📊 Recommendation Quality Metrics:")
        print(f"   📊 Coverage: {coverage:.3f} ({len(unique_recommended)}/{len(total_items)} items recommended)")
        print(f"   🎨 Diversity: {diversity:.3f} (0=all same genre, 1=perfectly diverse)")
        print(f"   ✨ Novelty: {novelty:.3f} (0=only popular items, 1=only niche items)")
        print(f"   ⭐ Average Score: {avg_score:.3f}")
        
        return metrics
    
    def demonstrate_cold_start_handling(self) -> Dict[str, Any]:
        """
        Demonstrate how SURG handles cold start problems.
        
        Cold start occurs when:
        1. New users with no interaction history
        2. New items with no ratings
        """
        print("\n🆕 Cold Start Problem Handling")
        print("=" * 50)
        
        # Create a new user with no history
        new_user_id = "new_user_001"
        print(f"🆕 Generating recommendations for new user: {new_user_id}")
        
        # Configure SURG with cold start handling
        config = SURGConfig(
            algorithm="hybrid",
            cold_start_strategy="popularity_based",  # fallback strategy
            min_interactions=1,
            max_recommendations=10
        )
        
        self.surg = SURG(config)
        self.surg.load_data(
            users_df=self.users_df,
            items_df=self.items_df,
            interactions_df=self.interactions_df
        )
        
        try:
            # Try to get recommendations for new user
            recommendations = self.surg.recommend(
                user_id=new_user_id,
                num_recommendations=5,
                cold_start_fallback=True
            )
            
            print(f"   Cold start recommendations (popularity-based):")
            for i, rec in enumerate(recommendations, 1):
                movie_info = self.items_df[
                    self.items_df['item_id'] == rec['item_id']
                ].iloc[0]
                print(f"     {i}. {movie_info['title']} ({movie_info['genre']}) - Score: {rec['score']:.3f}")
            
            return {new_user_id: recommendations}
            
        except Exception as e:
            print(f"   ❌ Cold start handling failed: {e}")
            print("   💡 This is expected in the demo - real implementation would handle this gracefully")
            return {}


def main():
    """
    Run comprehensive basic recommendation demonstrations.
    """
    print("🎬 SURG Basic Recommendations Demo")
    print("=" * 60)
    print("This demo shows core recommendation algorithms without GenAI enhancement.")
    print("Perfect for understanding fundamental recommendation concepts.\n")
    
    # Initialize demo
    demo = BasicRecommendationDemo()
    
    try:
        # Load sample data
        demo.load_sample_data()
        
        # Demonstrate different algorithms
        cf_results = demo.demonstrate_collaborative_filtering()
        cb_results = demo.demonstrate_content_based_filtering()
        hybrid_results = demo.demonstrate_hybrid_recommendations()
        
        # Analyze recommendation quality
        quality_metrics = demo.analyze_recommendation_quality()
        
        # Demonstrate cold start handling
        cold_start_results = demo.demonstrate_cold_start_handling()
        
        print("\n✅ All basic recommendation examples completed!")
        print("\n🎯 Key Takeaways:")
        print("   1. Collaborative filtering works well for users with rich interaction history")
        print("   2. Content-based filtering handles new users better")
        print("   3. Hybrid approaches combine strengths of both methods")
        print("   4. Quality metrics help evaluate recommendation systems")
        print("   5. Cold start problems need special handling strategies")
        
        print("\n🚀 Next Steps:")
        print("   - Try with your own data")
        print("   - Experiment with different algorithm parameters")
        print("   - Explore GenAI-enhanced recommendations")
        print("   - Set up evaluation pipelines")
        
    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        print("Make sure you have installed SURG correctly:")
        print("   pip install surg[all]")


if __name__ == "__main__":
    main()