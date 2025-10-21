"""
GenAI backend integration for the SURG recommendation system.

This module provides pluggable backends for integrating with various
Generative AI services, with primary support for OpenAI SDK. The backend
architecture allows for easy extension to other AI providers while
maintaining a consistent interface throughout the system.
"""

__all__ = [
    "GenAIBackend",
    "OpenAIBackend",
    "GenAIResponse",
    "EmbeddingResponse",
]