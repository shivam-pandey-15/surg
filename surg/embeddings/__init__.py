"""
Embedding generation and management for the SURG recommendation system.

This module provides comprehensive embedding generation capabilities using
GenAI backends, including caching, batch processing, and embedding management.
It serves as the bridge between raw data and vector representations used
throughout the recommendation pipeline.
"""

__all__ = [
    "EmbeddingGenerator",
    "EmbeddingCache",
    "EmbeddingManager",
    "EmbeddingConfig",
]