"""
Recommendation engine and algorithms for the SURG system.

This module implements the core recommendation algorithms including
candidate generation, neural ranking, and GenAI-assisted re-ranking.
It provides both collaborative filtering and content-based approaches
enhanced with generative AI capabilities.
"""

__all__ = [
    "RecommendationEngine",
    "CandidateGenerator", 
    "NeuralRanker",
    "GenAIReranker",
    "RecommendationStrategy",
]