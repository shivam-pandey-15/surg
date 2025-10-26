"""
SURG: Smart User Recommendation using GenAI

A hybrid recommendation system that integrates generative AI capabilities
with traditional collaborative filtering approaches built on top of LightFM
and other established recommendation libraries.

Architecture Overview:
┌─────────────────────────────────────────┐
│           User Interface Layer          │
├─────────────────────────────────────────┤
│        GenAI Enhancement Layers         │
│  ┌─────────┬─────────┬─────────────────┐ │
│  │Content  │Context  │ Explanation     │ │
│  │Gen Layer│Enrichmt │ Generation      │ │
│  └─────────┴─────────┴─────────────────┘ │
├─────────────────────────────────────────┤
│      Core Recommendation Algorithms     │
│  ┌─────────┬─────────┬─────────────────┐ │
│  │Traditnl │Matrix   │ Deep Learning   │ │
│  │Methods  │Factorzn │ Approaches      │ │
│  └─────────┴─────────┴─────────────────┘ │
├─────────────────────────────────────────┤
│         Data Processing Layer           │
└─────────────────────────────────────────┘

Key Components:
- Core: Traditional, matrix factorization, and deep learning algorithms
- GenAI: Content generation, context enrichment, explanation layers
- Pipeline: Training, evaluation, and recommendation workflows
- Utils: Configuration, preprocessing, metrics, and utilities
- Adapters: External service integrations (LLMs, vector stores, data sources)

Usage:
    >>> from surg import SURGRecommender
    >>> recommender = SURGRecommender()
    >>> recommender.fit(interactions, user_features, item_features)
    >>> recommendations = recommender.recommend(user_id=123, k=10)
"""

__version__ = "0.1.0"
__author__ = "Shivam Pandey"
__email__ = "shivampandey15199@gmail.com"