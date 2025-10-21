#!/usr/bin/env python3
"""
GenAI-Enhanced Recommendations Example
=====================================

This example demonstrates how GenAI (Generative AI) can enhance traditional
recommendation systems by providing explanations, context-aware suggestions,
and intelligent reasoning about user preferences.

Key Features Demonstrated:
- AI-powered recommendation explanations
- Context-aware personalization
- Multi-modal recommendation enhancement
- Intelligent filtering and ranking

Time to complete: ~15 minutes
Requirements: OpenAI API key
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import SURG components
from surg import SURG, SURGConfig
from surg.genai import OpenAIProvider
from surg.data import UserData, ItemData, InteractionData


class GenAIRecommendationDemo:
    """
    Comprehensive demonstration of GenAI-enhanced recommendations.
    
    This class shows how AI can make recommendations more intelligent,
    explainable, and contextually relevant.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.users_df = None
        self.items_df = None
        self.interactions_df = None
        self.surg = None
        
        if not self.openai_api_key:
            print("⚠️  Warning: OpenAI API key not found.")
            print("   Set OPENAI_API_KEY environment variable or pass it to the constructor.")
            print("   Some examples will be skipped.\n")
    
    def create_rich_sample_data(self) -> None:
        """
        Create rich sample data with detailed item descriptions and user context.
        This data is designed to showcase GenAI capabilities.
        """
        print("📚 Creating rich sample data for GenAI demonstrations...")
        
        np.random.seed(42)
        
        # Create diverse users with rich profiles
        users_data = {
            'user_id': [f'user_{i}' for i in range(1, 51)],
            'age': np.random.randint(22, 55, 50),
            'profession': np.random.choice([
                'software_engineer', 'teacher', 'doctor', 'artist', 'entrepreneur',
                'student', 'researcher', 'designer', 'writer', 'consultant'
            ], 50),
            'interests': [
                np.random.choice([
                    'technology', 'health', 'arts', 'business', 'science',
                    'sports', 'travel', 'cooking', 'music', 'photography'
                ], size=np.random.randint(2, 4), replace=False).tolist()
                for _ in range(50)
            ],
            'personality': np.random.choice([
                'analytical', 'creative', 'practical', 'adventurous', 'social',
                'introverted', 'detail_oriented', 'big_picture', 'methodical'
            ], 50),
            'current_goals': [
                np.random.choice([
                    'career_growth', 'skill_development', 'health_improvement',
                    'creative_expression', 'knowledge_expansion', 'networking',
                    'work_life_balance', 'financial_growth', 'personal_growth'
                ], size=np.random.randint(1, 3), replace=False).tolist()
                for _ in range(50)
            ]
        }
        self.users_df = pd.DataFrame(users_data)
        
        # Create detailed books with rich descriptions
        book_categories = ['Technology', 'Business', 'Science', 'Arts', 'Health', 'Biography']
        
        books_data = []
        for i in range(1, 101):
            category = np.random.choice(book_categories)
            
            # Generate context-appropriate titles and descriptions
            if category == 'Technology':
                titles = [
                    'The Future of AI and Machine Learning',
                    'Quantum Computing Fundamentals',
                    'Cybersecurity in the Digital Age',
                    'Building Scalable Web Applications',
                    'Data Science for Business Leaders'
                ]
                descriptions = [
                    'Comprehensive guide to understanding artificial intelligence and its applications in modern business.',
                    'Deep dive into quantum computing principles and their potential to revolutionize technology.',
                    'Essential cybersecurity practices for protecting digital assets in an interconnected world.',
                    'Practical approach to building web applications that scale with your business needs.',
                    'Bridge the gap between data science theory and practical business implementation.'
                ]
            elif category == 'Business':
                titles = [
                    'Strategic Leadership in the Modern Era',
                    'Startup Success Stories and Lessons',
                    'The Art of Negotiation',
                    'Digital Marketing Mastery',
                    'Financial Planning for Entrepreneurs'
                ]
                descriptions = [
                    'Learn how successful leaders navigate complex business challenges in today\'s fast-paced environment.',
                    'Real stories from successful startups and the key lessons that led to their success.',
                    'Master the psychology and tactics of effective negotiation in business and life.',
                    'Complete guide to leveraging digital channels for marketing success.',
                    'Essential financial strategies for building and growing a successful business.'
                ]
            elif category == 'Science':
                titles = [
                    'Climate Change and Environmental Solutions',
                    'The Mysteries of Space Exploration',
                    'Advances in Medical Research',
                    'The Psychology of Human Behavior',
                    'Renewable Energy Technologies'
                ]
                descriptions = [
                    'Explore the latest research on climate change and innovative solutions for a sustainable future.',
                    'Journey through the cosmos and discover the latest findings in space exploration.',
                    'Cutting-edge medical research that\'s changing how we understand and treat diseases.',
                    'Deep insights into human psychology and what drives our behavior and decision-making.',
                    'Comprehensive overview of renewable energy technologies and their implementation.'
                ]
            else:  # Arts, Health, Biography
                titles = [
                    'Creative Expression in the Digital Age',
                    'Mindfulness and Mental Wellness',
                    'Lives That Changed the World'
                ]
                descriptions = [
                    'How digital tools are transforming creative expression and artistic collaboration.',
                    'Practical approaches to mental wellness and mindfulness in everyday life.',
                    'Inspiring biographies of individuals who made significant impacts on society.'
                ]
            
            title = np.random.choice(titles)
            description = np.random.choice(descriptions)
            
            books_data.append({
                'item_id': f'book_{i}',
                'title': f'{title} (Volume {i})',
                'category': category,
                'author': f'Author {i}',
                'publication_year': np.random.randint(2015, 2024),
                'pages': np.random.randint(200, 500),
                'rating': round(np.random.uniform(3.5, 4.8), 1),
                'description': description,
                'tags': np.random.choice([
                    'bestseller', 'award_winning', 'practical', 'theoretical',
                    'beginner_friendly', 'advanced', 'comprehensive', 'concise'
                ], size=np.random.randint(2, 4), replace=False).tolist(),
                'price': round(np.random.uniform(15, 45), 2)
            })
        
        self.items_df = pd.DataFrame(books_data)
        
        # Create meaningful interactions
        interactions_data = []
        for user_idx in range(50):
            user_id = f"user_{user_idx + 1}"
            user_info = self.users_df.iloc[user_idx]
            
            # Users read books related to their interests and profession
            relevant_books = self.items_df[
                self.items_df['category'].str.lower().isin([
                    interest.lower() for interest in user_info['interests']
                ]) | 
                (self.items_df['category'] == 'Business') if user_info['profession'] in ['entrepreneur', 'consultant'] else False
            ]
            
            # Add some random books too
            random_books = self.items_df.sample(n=min(10, len(self.items_df)))
            potential_books = pd.concat([relevant_books, random_books]).drop_duplicates()
            
            # Generate interactions
            num_interactions = np.random.randint(5, 15)
            selected_books = potential_books.sample(n=min(num_interactions, len(potential_books)))
            
            for _, book in selected_books.iterrows():
                # Rating based on relevance to user
                base_rating = 3.0
                
                # Higher rating for relevant books
                if any(interest.lower() in book['category'].lower() for interest in user_info['interests']):
                    base_rating += 1.0
                
                # Adjust for user personality
                if user_info['personality'] in ['analytical', 'detail_oriented'] and 'comprehensive' in book['tags']:
                    base_rating += 0.5
                elif user_info['personality'] in ['creative', 'big_picture'] and 'theoretical' in book['tags']:
                    base_rating += 0.5
                
                rating = max(1, min(5, round(base_rating + np.random.normal(0, 0.7))))
                
                interactions_data.append({
                    'user_id': user_id,
                    'item_id': book['item_id'],
                    'rating': rating,
                    'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 180)),
                    'review_text': self._generate_review_text(rating, book['title'], user_info['personality'])
                })
        
        self.interactions_df = pd.DataFrame(interactions_data)
        
        print(f"✅ Rich sample data created:")
        print(f"   - {len(self.users_df)} users with detailed profiles")
        print(f"   - {len(self.items_df)} books with rich descriptions")
        print(f"   - {len(self.interactions_df)} interactions with reviews")
    
    def _generate_review_text(self, rating: int, title: str, personality: str) -> str:
        """Generate realistic review text based on rating and personality."""
        if rating >= 4:
            if personality == 'analytical':
                return f"Excellent insights in '{title}'. Well-researched and data-driven approach."
            elif personality == 'creative':
                return f"'{title}' sparked many creative ideas. Inspiring and thought-provoking."
            else:
                return f"Really enjoyed '{title}'. Practical and well-written."
        elif rating == 3:
            return f"'{title}' was decent. Some good points but could be more engaging."
        else:
            return f"'{title}' didn't meet my expectations. Could be improved."
    
    def demonstrate_ai_explanations(self) -> Dict[str, Any]:
        """
        Demonstrate AI-generated explanations for recommendations.
        
        Shows how GenAI can provide intelligent explanations for why
        specific items are recommended to users.
        """
        print("\n🤖 AI-Generated Recommendation Explanations")
        print("=" * 60)
        
        if not self.openai_api_key:
            print("⚠️  Skipping - OpenAI API key required")
            return {}
        
        # Configure SURG with GenAI explanations
        config = SURGConfig(
            algorithm="hybrid",
            genai_provider="openai",
            openai_api_key=self.openai_api_key,
            genai_model="gpt-3.5-turbo",
            enable_explanations=True,
            explanation_style="detailed",  # brief, detailed, technical
            max_recommendations=5
        )
        
        self.surg = SURG(config)
        self.surg.load_data(
            users_df=self.users_df,
            items_df=self.items_df,
            interactions_df=self.interactions_df
        )
        
        print("✅ SURG configured with AI explanations")
        
        # Get recommendations with explanations
        test_user = "user_5"
        user_info = self.users_df[self.users_df['user_id'] == test_user].iloc[0]
        
        print(f"\n👤 User Profile for {test_user}:")
        print(f"   Profession: {user_info['profession']}")
        print(f"   Interests: {', '.join(user_info['interests'])}")
        print(f"   Personality: {user_info['personality']}")
        print(f"   Goals: {', '.join(user_info['current_goals'])}")
        
        # Show user's reading history
        user_history = self.interactions_df[
            (self.interactions_df['user_id'] == test_user) & 
            (self.interactions_df['rating'] >= 4)
        ].merge(self.items_df, on='item_id')
        
        print(f"\n📚 Recent highly-rated books:")
        for _, book in user_history.head(3).iterrows():
            print(f"   - {book['title']} ({book['category']}) - {book['rating']}⭐")
        
        # Generate AI-enhanced recommendations
        recommendations = self.surg.recommend(
            user_id=test_user,
            num_recommendations=5,
            enhance_with_genai=True,
            include_explanations=True,
            context={
                'user_profile': user_info.to_dict(),
                'recent_activity': 'active_reader',
                'session_context': 'browsing_for_professional_development'
            }
        )
        
        print(f"\n🎯 AI-Enhanced Recommendations with Explanations:")
        for i, rec in enumerate(recommendations, 1):
            book_info = self.items_df[self.items_df['item_id'] == rec['item_id']].iloc[0]
            print(f"\n   {i}. {book_info['title']}")
            print(f"      Category: {book_info['category']} | Score: {rec['score']:.3f}")
            print(f"      💡 AI Explanation: {rec.get('explanation', 'No explanation available')}")
            
            if 'reasoning' in rec:
                print(f"      🧠 AI Reasoning: {rec['reasoning']}")
        
        return {test_user: recommendations}
    
    def demonstrate_contextual_recommendations(self) -> Dict[str, Any]:
        """
        Demonstrate context-aware recommendations using GenAI.
        
        Shows how AI can adapt recommendations based on current context,
        mood, situation, or specific goals.
        """
        print("\n🎯 Context-Aware AI Recommendations")
        print("=" * 60)
        
        if not self.openai_api_key:
            print("⚠️  Skipping - OpenAI API key required")
            return {}
        
        config = SURGConfig(
            algorithm="content_based",
            genai_provider="openai",
            openai_api_key=self.openai_api_key,
            use_context=True,
            context_weight=0.3
        )
        
        self.surg = SURG(config)
        self.surg.load_data(
            users_df=self.users_df,
            items_df=self.items_df,
            interactions_df=self.interactions_df
        )
        
        test_user = "user_10"
        user_info = self.users_df[self.users_df['user_id'] == test_user].iloc[0]
        
        # Different contextual scenarios
        contexts = [
            {
                'name': 'Work Commute',
                'context': {
                    'situation': 'commuting_to_work',
                    'time_available': '30_minutes',
                    'mood': 'focused',
                    'goal': 'professional_development'
                }
            },
            {
                'name': 'Weekend Relaxation',
                'context': {
                    'situation': 'weekend_leisure',
                    'time_available': '2_hours',
                    'mood': 'relaxed',
                    'goal': 'entertainment_and_learning'
                }
            },
            {
                'name': 'Before Important Meeting',
                'context': {
                    'situation': 'preparing_for_meeting',
                    'time_available': '15_minutes',
                    'mood': 'anxious',
                    'goal': 'confidence_building'
                }
            }
        ]
        
        all_recommendations = {}
        
        for scenario in contexts:
            print(f"\n📍 Scenario: {scenario['name']}")
            print(f"   Context: {scenario['context']}")
            
            recommendations = self.surg.recommend(
                user_id=test_user,
                num_recommendations=3,
                enhance_with_genai=True,
                context=scenario['context'],
                include_explanations=True
            )
            
            print(f"   🎯 Contextual Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                book_info = self.items_df[self.items_df['item_id'] == rec['item_id']].iloc[0]
                print(f"     {i}. {book_info['title']} ({book_info['category']})")
                print(f"        💡 Why for this context: {rec.get('explanation', 'N/A')}")
            
            all_recommendations[scenario['name']] = recommendations
        
        return all_recommendations
    
    def demonstrate_intelligent_filtering(self) -> Dict[str, Any]:
        """
        Demonstrate AI-powered intelligent filtering and curation.
        
        Shows how GenAI can filter recommendations based on complex criteria
        that would be difficult to encode in traditional rules.
        """
        print("\n🧠 Intelligent AI Filtering and Curation")
        print("=" * 60)
        
        if not self.openai_api_key:
            print("⚠️  Skipping - OpenAI API key required")
            return {}
        
        config = SURGConfig(
            algorithm="collaborative_filtering",
            genai_provider="openai",
            openai_api_key=self.openai_api_key,
            ai_filtering=True,
            content_filters=[
                "avoid_overly_technical",
                "prioritize_actionable_content",
                "match_current_skill_level"
            ]
        )
        
        self.surg = SURG(config)
        self.surg.load_data(
            users_df=self.users_df,
            items_df=self.items_df,
            interactions_df=self.interactions_df
        )
        
        test_user = "user_15"
        user_info = self.users_df[self.users_df['user_id'] == test_user].iloc[0]
        
        print(f"\n👤 User: {test_user}")
        print(f"   Profession: {user_info['profession']}")
        print(f"   Current Goals: {', '.join(user_info['current_goals'])}")
        
        # Demonstrate different filtering scenarios
        filtering_scenarios = [
            {
                'name': 'Beginner-Friendly Only',
                'filters': ['beginner_friendly', 'practical_focus', 'step_by_step_guidance'],
                'prompt': 'Filter for books suitable for someone just starting in their field'
            },
            {
                'name': 'Advanced Deep-Dive',
                'filters': ['advanced_level', 'comprehensive_coverage', 'expert_insights'],
                'prompt': 'Filter for advanced, in-depth books for an experienced professional'
            },
            {
                'name': 'Quick Wins',
                'filters': ['actionable_tips', 'immediate_application', 'time_efficient'],
                'prompt': 'Filter for books that provide quick, actionable insights'
            }
        ]
        
        all_filtered_recommendations = {}
        
        for scenario in filtering_scenarios:
            print(f"\n🔍 Filtering Scenario: {scenario['name']}")
            print(f"   AI Filter Prompt: {scenario['prompt']}")
            
            recommendations = self.surg.recommend(
                user_id=test_user,
                num_recommendations=4,
                enhance_with_genai=True,
                ai_filters=scenario['filters'],
                filter_prompt=scenario['prompt'],
                include_explanations=True
            )
            
            print(f"   📚 Filtered Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                book_info = self.items_df[self.items_df['item_id'] == rec['item_id']].iloc[0]
                print(f"     {i}. {book_info['title']}")
                print(f"        Tags: {', '.join(book_info['tags'])}")
                print(f"        💡 Why it matches filter: {rec.get('filter_explanation', 'N/A')}")
            
            all_filtered_recommendations[scenario['name']] = recommendations
        
        return all_filtered_recommendations
    
    def demonstrate_multi_modal_enhancement(self) -> Dict[str, Any]:
        """
        Demonstrate multi-modal GenAI enhancement.
        
        Shows how AI can integrate multiple data sources and types
        to create richer recommendations.
        """
        print("\n🎨 Multi-Modal AI Enhancement")
        print("=" * 60)
        
        if not self.openai_api_key:
            print("⚠️  Skipping - OpenAI API key required")
            return {}
        
        # This demonstrates the concept - in a real implementation,
        # you would integrate actual multi-modal data
        print("🔮 Simulating multi-modal AI enhancement...")
        print("   In production, this would integrate:")
        print("   - Text analysis of book descriptions")
        print("   - User behavior patterns")
        print("   - Social signals and trends")
        print("   - Temporal factors")
        print("   - Cross-platform data")
        
        config = SURGConfig(
            algorithm="hybrid",
            genai_provider="openai",
            openai_api_key=self.openai_api_key,
            multi_modal=True,
            modalities=["text", "behavioral", "temporal", "social"]
        )
        
        test_user = "user_20"
        
        # Simulate multi-modal analysis
        multi_modal_insights = {
            'text_analysis': {
                'preferred_writing_style': 'practical_and_conversational',
                'complexity_level': 'intermediate',
                'topics_of_interest': ['technology', 'business_strategy']
            },
            'behavioral_patterns': {
                'reading_time': 'evening_reader',
                'completion_rate': 'high',
                'review_tendency': 'detailed_reviewer'
            },
            'temporal_factors': {
                'seasonal_preferences': 'prefers_new_releases',
                'trending_topics': 'ai_and_automation'
            },
            'social_signals': {
                'peer_recommendations': 'follows_tech_leaders',
                'discussion_participation': 'active_in_forums'
            }
        }
        
        print(f"\n🧩 Multi-Modal Analysis for {test_user}:")
        for modality, insights in multi_modal_insights.items():
            print(f"   {modality.title()}: {insights}")
        
        # Simulate enhanced recommendations
        enhanced_recommendations = [
            {
                'item_id': 'book_42',
                'score': 0.95,
                'multi_modal_explanation': 'High match across all modalities: writing style matches preference, trending topic, discussed in user\'s forums, optimal length for evening reading',
                'confidence': 'very_high'
            },
            {
                'item_id': 'book_73',
                'score': 0.88,
                'multi_modal_explanation': 'Strong text and behavioral match: intermediate complexity, practical approach, similar to highly-rated books',
                'confidence': 'high'
            }
        ]
        
        print(f"\n🎯 Multi-Modal Enhanced Recommendations:")
        for i, rec in enumerate(enhanced_recommendations, 1):
            book_info = self.items_df[self.items_df['item_id'] == rec['item_id']].iloc[0]
            print(f"   {i}. {book_info['title']}")
            print(f"      Score: {rec['score']:.3f} | Confidence: {rec['confidence']}")
            print(f"      🧩 Multi-modal insight: {rec['multi_modal_explanation']}")
        
        return {test_user: enhanced_recommendations}


def main():
    """
    Run comprehensive GenAI recommendation demonstrations.
    """
    print("🤖 SURG GenAI-Enhanced Recommendations Demo")
    print("=" * 70)
    print("This demo showcases how GenAI can make recommendations more intelligent,")
    print("explainable, and contextually relevant.\n")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OPENAI_API_KEY environment variable not set.")
        print("   Some examples will be skipped.")
        print("   To run all examples, set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'\n")
    
    # Initialize demo
    demo = GenAIRecommendationDemo(api_key)
    
    try:
        # Create rich sample data
        demo.create_rich_sample_data()
        
        # Run GenAI demonstrations
        explanation_results = demo.demonstrate_ai_explanations()
        contextual_results = demo.demonstrate_contextual_recommendations()
        filtering_results = demo.demonstrate_intelligent_filtering()
        multimodal_results = demo.demonstrate_multi_modal_enhancement()
        
        print("\n✅ All GenAI recommendation examples completed!")
        print("\n🎯 Key GenAI Capabilities Demonstrated:")
        print("   1. 🤖 AI-generated explanations for recommendations")
        print("   2. 🎯 Context-aware personalization")
        print("   3. 🧠 Intelligent filtering and curation")
        print("   4. 🎨 Multi-modal data integration")
        print("   5. 📊 Enhanced reasoning and decision-making")
        
        print("\n🚀 Advanced Applications:")
        print("   - Real-time context adaptation")
        print("   - Cross-platform recommendation consistency")
        print("   - Emotion and sentiment-aware recommendations")
        print("   - Conversational recommendation interfaces")
        print("   - Automated A/B testing of recommendation strategies")
        
        print("\n💡 Next Steps:")
        print("   - Integrate with your own GenAI models")
        print("   - Experiment with custom prompts and reasoning")
        print("   - Set up evaluation metrics for AI-enhanced recommendations")
        print("   - Explore conversation-driven recommendation flows")
        
    except Exception as e:
        print(f"\n❌ Error in GenAI demo: {e}")
        print("Make sure you have:")
        print("   1. Installed SURG with GenAI support: pip install surg[genai]")
        print("   2. Set up your OpenAI API key")
        print("   3. Sufficient API quota for OpenAI requests")


if __name__ == "__main__":
    main()