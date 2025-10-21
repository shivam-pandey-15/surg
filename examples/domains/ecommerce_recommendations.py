#!/usr/bin/env python3
"""
E-commerce Recommendations Example
=================================

This example demonstrates how to use SURG for e-commerce product recommendations
with real-world scenarios including cart abandonment, seasonal trends, and
cross-selling strategies.

Key E-commerce Features:
- Product similarity recommendations
- Customer segmentation-based recommendations
- Cart abandonment recovery
- Cross-selling and upselling
- Seasonal and trending product recommendations
- Price-sensitive recommendations

Time to complete: ~15 minutes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import SURG components
from surg import SURG, SURGConfig
from surg.data import UserData, ItemData, InteractionData


class EcommerceRecommendationSystem:
    """
    Complete e-commerce recommendation system demonstration.
    
    This class shows how to implement various e-commerce recommendation
    strategies using SURG, including personalization, business rules,
    and revenue optimization.
    """
    
    def __init__(self):
        self.customers_df = None
        self.products_df = None
        self.orders_df = None
        self.cart_events_df = None
        self.surg = None
        
    def create_ecommerce_data(self) -> None:
        """
        Create realistic e-commerce data including customers, products,
        orders, and shopping behavior.
        """
        print("🛒 Creating realistic e-commerce data...")
        
        np.random.seed(42)
        
        # Create customer profiles
        customers_data = {
            'customer_id': [f'cust_{i}' for i in range(1, 1001)],
            'age': np.random.randint(18, 70, 1000),
            'gender': np.random.choice(['M', 'F'], 1000),
            'location': np.random.choice([
                'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'
            ], 1000),
            'income_level': np.random.choice(['low', 'medium', 'high'], 1000, p=[0.3, 0.5, 0.2]),
            'customer_segment': np.random.choice([
                'budget_conscious', 'premium', 'brand_loyal', 'bargain_hunter', 'early_adopter'
            ], 1000),
            'signup_date': [
                datetime.now() - timedelta(days=np.random.randint(1, 730))
                for _ in range(1000)
            ],
            'total_spent': np.random.exponential(200, 1000),  # Exponential distribution for spending
            'order_frequency': np.random.choice(['low', 'medium', 'high'], 1000, p=[0.4, 0.4, 0.2])
        }
        self.customers_df = pd.DataFrame(customers_data)
        
        # Create product catalog
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Toys']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'Premium', 'Budget']
        
        products_data = []
        for i in range(1, 2001):  # 2000 products
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            
            # Generate category-appropriate pricing
            if category == 'Electronics':
                base_price = np.random.uniform(50, 1000)
            elif category == 'Clothing':
                base_price = np.random.uniform(20, 200)
            elif category == 'Home & Garden':
                base_price = np.random.uniform(15, 500)
            else:
                base_price = np.random.uniform(10, 100)
            
            # Adjust price based on brand
            if brand == 'Premium':
                price = base_price * np.random.uniform(1.2, 2.0)
            elif brand == 'Budget':
                price = base_price * np.random.uniform(0.5, 0.8)
            else:
                price = base_price * np.random.uniform(0.8, 1.2)
            
            products_data.append({
                'product_id': f'prod_{i}',
                'name': f'{category} Product {i}',
                'category': category,
                'subcategory': f'{category}_Sub_{np.random.randint(1, 5)}',
                'brand': brand,
                'price': round(price, 2),
                'cost': round(price * np.random.uniform(0.4, 0.7), 2),  # Cost for margin calculation
                'rating': round(np.random.uniform(3.0, 5.0), 1),
                'review_count': np.random.randint(0, 1000),
                'stock_quantity': np.random.randint(0, 500),
                'weight': np.random.uniform(0.1, 10.0),  # kg
                'dimensions': f'{np.random.randint(5, 50)}x{np.random.randint(5, 50)}x{np.random.randint(5, 50)}',
                'launch_date': datetime.now() - timedelta(days=np.random.randint(1, 1095)),
                'seasonal': np.random.choice([True, False], p=[0.3, 0.7]),
                'promotion_eligible': np.random.choice([True, False], p=[0.4, 0.6])
            })
        
        self.products_df = pd.DataFrame(products_data)
        
        # Create order history
        orders_data = []
        order_id = 1
        
        for customer_idx in range(1000):
            customer_id = f'cust_{customer_idx + 1}'
            customer_info = self.customers_df.iloc[customer_idx]
            
            # Number of orders based on customer frequency
            if customer_info['order_frequency'] == 'high':
                num_orders = np.random.randint(10, 30)
            elif customer_info['order_frequency'] == 'medium':
                num_orders = np.random.randint(3, 10)
            else:
                num_orders = np.random.randint(1, 5)
            
            for _ in range(num_orders):
                order_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
                
                # Items per order
                items_in_order = np.random.randint(1, 8)
                
                for _ in range(items_in_order):
                    # Select product based on customer preferences
                    if customer_info['customer_segment'] == 'premium':
                        # Premium customers prefer higher-priced items
                        available_products = self.products_df[self.products_df['price'] > 100]
                    elif customer_info['customer_segment'] == 'budget_conscious':
                        # Budget customers prefer lower-priced items
                        available_products = self.products_df[self.products_df['price'] < 50]
                    else:
                        available_products = self.products_df
                    
                    if len(available_products) > 0:
                        product = available_products.sample(1).iloc[0]
                        
                        orders_data.append({
                            'order_id': f'order_{order_id}',
                            'customer_id': customer_id,
                            'product_id': product['product_id'],
                            'quantity': np.random.randint(1, 4),
                            'unit_price': product['price'],
                            'total_price': product['price'] * np.random.randint(1, 4),
                            'order_date': order_date,
                            'rating': np.random.randint(1, 6) if np.random.random() < 0.7 else None,
                            'review_text': None  # Could be populated with review text
                        })
                
                order_id += 1
        
        self.orders_df = pd.DataFrame(orders_data)
        
        # Create cart abandonment data
        cart_events_data = []
        for _ in range(5000):  # 5000 cart events
            customer_id = f"cust_{np.random.randint(1, 1001)}"
            product_id = f"prod_{np.random.randint(1, 2001)}"
            
            cart_events_data.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'event_type': np.random.choice(['add_to_cart', 'remove_from_cart', 'abandoned_cart'], p=[0.5, 0.2, 0.3]),
                'event_date': datetime.now() - timedelta(days=np.random.randint(1, 90)),
                'session_id': f'session_{np.random.randint(1, 10000)}'
            })
        
        self.cart_events_df = pd.DataFrame(cart_events_data)
        
        print(f"✅ E-commerce data created:")
        print(f"   - {len(self.customers_df)} customers")
        print(f"   - {len(self.products_df)} products")
        print(f"   - {len(self.orders_df)} order items")
        print(f"   - {len(self.cart_events_df)} cart events")
        
        # Calculate some statistics
        avg_order_value = self.orders_df.groupby('order_id')['total_price'].sum().mean()
        print(f"   - Average order value: ${avg_order_value:.2f}")
        
        category_distribution = self.products_df['category'].value_counts()
        print(f"   - Product categories: {dict(category_distribution)}")
    
    def demonstrate_product_similarity_recommendations(self) -> Dict[str, Any]:
        """
        Demonstrate product similarity recommendations.
        
        Shows "customers who viewed this item also viewed" and
        "similar products" recommendations.
        """
        print("\n🔍 Product Similarity Recommendations")
        print("=" * 60)
        
        # Configure SURG for product similarity
        config = SURGConfig(
            algorithm="content_based",
            content_features=["category", "subcategory", "brand", "price"],
            similarity_metric="cosine",
            feature_weights={
                "category": 0.4,
                "subcategory": 0.3,
                "brand": 0.2,
                "price": 0.1
            }
        )
        
        self.surg = SURG(config)
        
        # Prepare data for SURG
        # Transform orders to interactions format
        interactions_df = self.orders_df.rename(columns={
            'customer_id': 'user_id',
            'product_id': 'item_id',
            'order_date': 'timestamp'
        })
        interactions_df['interaction_type'] = 'purchase'
        
        # Load data
        self.surg.load_data(
            users_df=self.customers_df.rename(columns={'customer_id': 'user_id'}),
            items_df=self.products_df.rename(columns={'product_id': 'item_id'}),
            interactions_df=interactions_df
        )
        
        print("✅ Data loaded for product similarity analysis")
        
        # Demonstrate similar products for different categories
        test_products = [
            'prod_1',   # Electronics
            'prod_500', # Clothing
            'prod_1000' # Home & Garden
        ]
        
        all_similar_products = {}
        
        for product_id in test_products:
            product_info = self.products_df[self.products_df['product_id'] == product_id].iloc[0]
            
            print(f"\n📱 Similar products to: {product_info['name']}")
            print(f"   Category: {product_info['category']} | Brand: {product_info['brand']} | Price: ${product_info['price']}")
            
            # Get similar products
            similar_products = self.surg.get_similar_items(
                item_id=product_id,
                num_similar=5,
                similarity_threshold=0.3
            )
            
            print(f"   🔗 Similar products:")
            for i, similar in enumerate(similar_products, 1):
                similar_info = self.products_df[
                    self.products_df['product_id'] == similar['item_id']
                ].iloc[0]
                print(f"     {i}. {similar_info['name']} (${similar_info['price']}) - Similarity: {similar['similarity']:.3f}")
            
            all_similar_products[product_id] = similar_products
        
        return all_similar_products
    
    def demonstrate_customer_segmentation_recommendations(self) -> Dict[str, Any]:
        """
        Demonstrate recommendations based on customer segmentation.
        
        Shows how different customer segments get personalized recommendations
        based on their behavior and preferences.
        """
        print("\n👥 Customer Segmentation-Based Recommendations")
        print("=" * 60)
        
        # Configure SURG for collaborative filtering
        config = SURGConfig(
            algorithm="collaborative_filtering",
            user_features=["age", "income_level", "customer_segment"],
            min_interactions=3,
            num_neighbors=20
        )
        
        self.surg = SURG(config)
        
        # Prepare data
        interactions_df = self.orders_df.rename(columns={
            'customer_id': 'user_id',
            'product_id': 'item_id',
            'order_date': 'timestamp'
        })
        interactions_df['interaction_type'] = 'purchase'
        
        self.surg.load_data(
            users_df=self.customers_df.rename(columns={'customer_id': 'user_id'}),
            items_df=self.products_df.rename(columns={'product_id': 'item_id'}),
            interactions_df=interactions_df
        )
        
        print("✅ Data loaded for customer segmentation analysis")
        
        # Test different customer segments
        segment_examples = {
            'premium': 'cust_1',
            'budget_conscious': 'cust_2',
            'brand_loyal': 'cust_3',
            'early_adopter': 'cust_4'
        }
        
        all_segment_recommendations = {}
        
        for segment, customer_id in segment_examples.items():
            customer_info = self.customers_df[
                self.customers_df['customer_id'] == customer_id
            ].iloc[0]
            
            print(f"\n🎯 Recommendations for {segment.upper()} customer:")
            print(f"   Customer: {customer_id}")
            print(f"   Age: {customer_info['age']} | Income: {customer_info['income_level']}")
            print(f"   Total spent: ${customer_info['total_spent']:.2f}")
            
            # Get purchase history
            purchase_history = self.orders_df[
                self.orders_df['customer_id'] == customer_id
            ].merge(self.products_df, on='product_id')
            
            if len(purchase_history) > 0:
                print(f"   Recent purchases:")
                for _, purchase in purchase_history.head(3).iterrows():
                    print(f"     - {purchase['name']} (${purchase['unit_price']})")
            
            # Generate recommendations
            recommendations = self.surg.recommend(
                user_id=customer_id,
                num_recommendations=5,
                exclude_seen=True
            )
            
            print(f"   🛍️ Personalized recommendations:")
            for i, rec in enumerate(recommendations, 1):
                product_info = self.products_df[
                    self.products_df['product_id'] == rec['item_id']
                ].iloc[0]
                print(f"     {i}. {product_info['name']} (${product_info['price']}) - Score: {rec['score']:.3f}")
            
            all_segment_recommendations[segment] = recommendations
        
        return all_segment_recommendations
    
    def demonstrate_cart_abandonment_recovery(self) -> Dict[str, Any]:
        """
        Demonstrate cart abandonment recovery recommendations.
        
        Shows how to create recommendations to bring customers back
        to complete their purchases.
        """
        print("\n🛒 Cart Abandonment Recovery Recommendations")
        print("=" * 60)
        
        # Find customers with abandoned carts
        abandoned_carts = self.cart_events_df[
            self.cart_events_df['event_type'] == 'abandoned_cart'
        ].groupby('customer_id').agg({
            'product_id': 'count',
            'event_date': 'max'
        }).reset_index()
        
        # Focus on recent abandonments (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_abandoners = abandoned_carts[
            abandoned_carts['event_date'] > recent_cutoff
        ].head(5)
        
        print(f"📊 Found {len(recent_abandoners)} customers with recent cart abandonments")
        
        recovery_strategies = {}
        
        for _, abandoner in recent_abandoners.iterrows():
            customer_id = abandoner['customer_id']
            customer_info = self.customers_df[
                self.customers_df['customer_id'] == customer_id
            ].iloc[0]
            
            print(f"\n🎯 Recovery strategy for {customer_id}:")
            print(f"   Segment: {customer_info['customer_segment']}")
            print(f"   Abandoned items: {abandoner['product_id']}")
            
            # Get abandoned products
            abandoned_products = self.cart_events_df[
                (self.cart_events_df['customer_id'] == customer_id) &
                (self.cart_events_df['event_type'] == 'abandoned_cart')
            ]['product_id'].tolist()
            
            print(f"   📦 Abandoned products:")
            for prod_id in abandoned_products[:3]:
                product_info = self.products_df[
                    self.products_df['product_id'] == prod_id
                ].iloc[0]
                print(f"     - {product_info['name']} (${product_info['price']})")
            
            # Recovery recommendations strategy
            recovery_recs = []
            
            # 1. Remind about abandoned items (with potential discount)
            for prod_id in abandoned_products[:2]:
                product_info = self.products_df[
                    self.products_df['product_id'] == prod_id
                ].iloc[0]
                discount_price = product_info['price'] * 0.9  # 10% discount
                recovery_recs.append({
                    'item_id': prod_id,
                    'strategy': 'abandoned_item_reminder',
                    'original_price': product_info['price'],
                    'discount_price': discount_price,
                    'message': f'Complete your purchase - Save 10%!'
                })
            
            # 2. Recommend similar items at lower price points
            for prod_id in abandoned_products[:1]:
                similar_items = self.surg.get_similar_items(
                    item_id=prod_id,
                    num_similar=3,
                    price_range=(0, product_info['price'] * 0.8)  # Cheaper alternatives
                )
                
                for similar in similar_items:
                    recovery_recs.append({
                        'item_id': similar['item_id'],
                        'strategy': 'cheaper_alternative',
                        'message': 'Similar item at a better price!'
                    })
            
            # 3. Complementary products
            # (In a real system, this would use association rules)
            category = self.products_df[
                self.products_df['product_id'] == abandoned_products[0]
            ].iloc[0]['category']
            
            complementary_products = self.products_df[
                (self.products_df['category'] == category) &
                (~self.products_df['product_id'].isin(abandoned_products)) &
                (self.products_df['price'] < 50)  # Affordable add-ons
            ].sample(2)
            
            for _, comp_prod in complementary_products.iterrows():
                recovery_recs.append({
                    'item_id': comp_prod['product_id'],
                    'strategy': 'complementary_product',
                    'message': 'Perfect addition to your cart!'
                })
            
            print(f"   💡 Recovery recommendations:")
            for i, rec in enumerate(recovery_recs[:5], 1):
                product_info = self.products_df[
                    self.products_df['product_id'] == rec['item_id']
                ].iloc[0]
                print(f"     {i}. {product_info['name']} - {rec['strategy']}")
                print(f"        💬 {rec['message']}")
            
            recovery_strategies[customer_id] = recovery_recs
        
        return recovery_strategies
    
    def demonstrate_cross_selling_upselling(self) -> Dict[str, Any]:
        """
        Demonstrate cross-selling and upselling recommendations.
        
        Shows how to recommend complementary products (cross-sell)
        and higher-value alternatives (upsell).
        """
        print("\n📈 Cross-Selling and Upselling Recommendations")
        print("=" * 60)
        
        # Analyze purchase patterns for cross-selling opportunities
        print("🔍 Analyzing purchase patterns for cross-sell opportunities...")
        
        # Find frequently bought together items
        order_combinations = self.orders_df.groupby('order_id')['product_id'].apply(list).reset_index()
        order_combinations = order_combinations[order_combinations['product_id'].str.len() > 1]
        
        # Simple market basket analysis
        frequent_combinations = {}
        for _, order in order_combinations.iterrows():
            products = order['product_id']
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    pair = tuple(sorted([products[i], products[j]]))
                    frequent_combinations[pair] = frequent_combinations.get(pair, 0) + 1
        
        # Get top combinations
        top_combinations = sorted(frequent_combinations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"📊 Top product combinations:")
        for (prod1, prod2), count in top_combinations[:5]:
            prod1_info = self.products_df[self.products_df['product_id'] == prod1].iloc[0]
            prod2_info = self.products_df[self.products_df['product_id'] == prod2].iloc[0]
            print(f"   {prod1_info['name']} + {prod2_info['name']} ({count} times)")
        
        # Demonstrate for specific customer
        test_customer = 'cust_10'
        customer_info = self.customers_df[
            self.customers_df['customer_id'] == test_customer
        ].iloc[0]
        
        print(f"\n🎯 Cross-sell/Upsell for {test_customer}:")
        print(f"   Segment: {customer_info['customer_segment']}")
        print(f"   Total spent: ${customer_info['total_spent']:.2f}")
        
        # Get customer's recent purchases
        recent_purchases = self.orders_df[
            (self.orders_df['customer_id'] == test_customer) &
            (self.orders_df['order_date'] > datetime.now() - timedelta(days=30))
        ].merge(self.products_df, on='product_id')
        
        if len(recent_purchases) > 0:
            print(f"   📦 Recent purchases:")
            for _, purchase in recent_purchases.head(3).iterrows():
                print(f"     - {purchase['name']} (${purchase['unit_price']})")
            
            # Cross-selling recommendations
            print(f"\n🔗 Cross-selling opportunities:")
            cross_sell_recs = []
            
            for _, purchase in recent_purchases.head(2).iterrows():
                # Find products frequently bought with this item
                purchase_category = purchase['category']
                
                # Recommend complementary categories
                complementary_categories = {
                    'Electronics': ['Electronics', 'Books'],  # Accessories, manuals
                    'Clothing': ['Beauty', 'Clothing'],       # Matching items, accessories
                    'Home & Garden': ['Home & Garden', 'Books'], # Related items, DIY guides
                    'Sports': ['Sports', 'Clothing'],         # Equipment, apparel
                    'Books': ['Electronics', 'Books'],        # E-readers, related books
                    'Beauty': ['Beauty', 'Clothing'],         # Related products, fashion
                    'Toys': ['Books', 'Electronics']          # Educational, tech toys
                }.get(purchase_category, [purchase_category])
                
                complementary_products = self.products_df[
                    (self.products_df['category'].isin(complementary_categories)) &
                    (self.products_df['product_id'] != purchase['product_id']) &
                    (self.products_df['price'] <= purchase['unit_price'] * 1.5)  # Reasonable price range
                ].sample(min(3, len(self.products_df)))
                
                for _, comp_prod in complementary_products.iterrows():
                    cross_sell_recs.append({
                        'item_id': comp_prod['product_id'],
                        'item_name': comp_prod['name'],
                        'price': comp_prod['price'],
                        'reason': f"Goes well with {purchase['name']}"
                    })
            
            for i, rec in enumerate(cross_sell_recs[:5], 1):
                print(f"     {i}. {rec['item_name']} (${rec['price']:.2f}) - {rec['reason']}")
            
            # Upselling recommendations
            print(f"\n⬆️ Upselling opportunities:")
            upsell_recs = []
            
            for _, purchase in recent_purchases.head(2).iterrows():
                # Find higher-value items in same category
                upsell_products = self.products_df[
                    (self.products_df['category'] == purchase['category']) &
                    (self.products_df['price'] > purchase['unit_price']) &
                    (self.products_df['price'] <= purchase['unit_price'] * 2) &  # Not too expensive
                    (self.products_df['rating'] >= purchase['rating'])  # Better or equal rating
                ].sort_values('rating', ascending=False).head(3)
                
                for _, upsell_prod in upsell_products.iterrows():
                    price_difference = upsell_prod['price'] - purchase['unit_price']
                    upsell_recs.append({
                        'item_id': upsell_prod['product_id'],
                        'item_name': upsell_prod['name'],
                        'price': upsell_prod['price'],
                        'price_difference': price_difference,
                        'reason': f"Premium upgrade from {purchase['name']}"
                    })
            
            for i, rec in enumerate(upsell_recs[:5], 1):
                print(f"     {i}. {rec['item_name']} (${rec['price']:.2f}, +${rec['price_difference']:.2f}) - {rec['reason']}")
            
            return {
                'customer_id': test_customer,
                'cross_sell': cross_sell_recs,
                'upsell': upsell_recs
            }
        
        else:
            print("   No recent purchases found for demonstration")
            return {}
    
    def demonstrate_seasonal_trending_recommendations(self) -> Dict[str, Any]:
        """
        Demonstrate seasonal and trending product recommendations.
        
        Shows how to incorporate time-based factors and popularity trends
        into recommendations.
        """
        print("\n📅 Seasonal and Trending Recommendations")
        print("=" * 60)
        
        # Simulate seasonal patterns
        current_month = datetime.now().month
        seasonal_categories = {
            'winter': ['Clothing', 'Home & Garden'],  # Winter clothes, heating
            'spring': ['Home & Garden', 'Sports'],    # Gardening, outdoor sports
            'summer': ['Sports', 'Beauty'],           # Summer activities, sun care
            'fall': ['Clothing', 'Books']             # Back to school, cozy items
        }
        
        # Determine current season
        if current_month in [12, 1, 2]:
            current_season = 'winter'
        elif current_month in [3, 4, 5]:
            current_season = 'spring'
        elif current_month in [6, 7, 8]:
            current_season = 'summer'
        else:
            current_season = 'fall'
        
        print(f"🌟 Current season: {current_season.upper()}")
        seasonal_cats = seasonal_categories[current_season]
        print(f"   Promoted categories: {', '.join(seasonal_cats)}")
        
        # Trending products (simulate based on recent orders)
        recent_orders = self.orders_df[
            self.orders_df['order_date'] > datetime.now() - timedelta(days=30)
        ]
        
        trending_products = recent_orders.groupby('product_id').agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        trending_products['trend_score'] = (
            trending_products['quantity'] * 0.4 +
            trending_products['order_id'] * 0.6
        )
        
        trending_products = trending_products.sort_values('trend_score', ascending=False)
        
        print(f"\n📈 Trending products (last 30 days):")
        for _, trending in trending_products.head(5).iterrows():
            product_info = self.products_df[
                self.products_df['product_id'] == trending['product_id']
            ].iloc[0]
            print(f"   {product_info['name']} ({product_info['category']}) - Trend Score: {trending['trend_score']:.1f}")
        
        # Generate seasonal recommendations for different customers
        test_customers = ['cust_5', 'cust_15', 'cust_25']
        seasonal_recommendations = {}
        
        for customer_id in test_customers:
            customer_info = self.customers_df[
                self.customers_df['customer_id'] == customer_id
            ].iloc[0]
            
            print(f"\n🎯 Seasonal recommendations for {customer_id}:")
            print(f"   Age: {customer_info['age']} | Segment: {customer_info['customer_segment']}")
            
            # Combine seasonal and trending factors
            seasonal_products = self.products_df[
                self.products_df['category'].isin(seasonal_cats)
            ]
            
            # Filter based on customer segment
            if customer_info['customer_segment'] == 'premium':
                seasonal_products = seasonal_products[seasonal_products['price'] > 100]
            elif customer_info['customer_segment'] == 'budget_conscious':
                seasonal_products = seasonal_products[seasonal_products['price'] < 50]
            
            # Add trending boost
            seasonal_trending = seasonal_products.merge(
                trending_products[['product_id', 'trend_score']],
                on='product_id',
                how='left'
            )
            seasonal_trending['trend_score'] = seasonal_trending['trend_score'].fillna(0)
            
            # Score products (combine rating and trend)
            seasonal_trending['recommendation_score'] = (
                seasonal_trending['rating'] * 0.6 +
                seasonal_trending['trend_score'] * 0.4
            )
            
            # Get top recommendations
            top_seasonal = seasonal_trending.sort_values(
                'recommendation_score', ascending=False
            ).head(5)
            
            print(f"   🌟 Seasonal + Trending recommendations:")
            for i, (_, product) in enumerate(top_seasonal.iterrows(), 1):
                print(f"     {i}. {product['name']} (${product['price']:.2f})")
                print(f"        Rating: {product['rating']}⭐ | Trend: {product['trend_score']:.1f}")
            
            seasonal_recommendations[customer_id] = top_seasonal.to_dict('records')
        
        return seasonal_recommendations


def main():
    """
    Run comprehensive e-commerce recommendation demonstrations.
    """
    print("🛍️  SURG E-commerce Recommendations Demo")
    print("=" * 70)
    print("This demo showcases e-commerce-specific recommendation strategies")
    print("including personalization, cross-selling, and business optimization.\n")
    
    # Initialize e-commerce system
    ecommerce = EcommerceRecommendationSystem()
    
    try:
        # Create realistic e-commerce data
        ecommerce.create_ecommerce_data()
        
        # Demonstrate different recommendation strategies
        similarity_results = ecommerce.demonstrate_product_similarity_recommendations()
        segmentation_results = ecommerce.demonstrate_customer_segmentation_recommendations()
        cart_recovery_results = ecommerce.demonstrate_cart_abandonment_recovery()
        cross_sell_results = ecommerce.demonstrate_cross_selling_upselling()
        seasonal_results = ecommerce.demonstrate_seasonal_trending_recommendations()
        
        print("\n✅ All e-commerce recommendation examples completed!")
        print("\n🎯 E-commerce Strategies Demonstrated:")
        print("   1. 🔍 Product similarity recommendations")
        print("   2. 👥 Customer segmentation-based personalization")
        print("   3. 🛒 Cart abandonment recovery")
        print("   4. 📈 Cross-selling and upselling")
        print("   5. 📅 Seasonal and trending recommendations")
        
        print("\n💰 Business Impact:")
        print("   - Increased average order value through cross-selling")
        print("   - Reduced cart abandonment through targeted recovery")
        print("   - Improved customer satisfaction with relevant recommendations")
        print("   - Higher conversion rates through personalization")
        print("   - Better inventory turnover with trending promotions")
        
        print("\n🚀 Advanced E-commerce Features:")
        print("   - Real-time inventory integration")
        print("   - Price optimization algorithms")
        print("   - A/B testing for recommendation strategies")
        print("   - Multi-channel consistency")
        print("   - Revenue-optimized ranking")
        print("   - Customer lifetime value prediction")
        
        print("\n💡 Next Steps:")
        print("   - Integrate with your product catalog")
        print("   - Set up real-time event tracking")
        print("   - Implement business rule constraints")
        print("   - Add revenue and margin optimization")
        print("   - Deploy A/B testing framework")
        
    except Exception as e:
        print(f"\n❌ Error in e-commerce demo: {e}")
        print("Make sure you have installed SURG correctly:")
        print("   pip install surg[all]")


if __name__ == "__main__":
    main()