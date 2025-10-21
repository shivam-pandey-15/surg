"""
SURG: Smart User Recommendation with GenAI

A comprehensive recommendation system library that leverages Generative AI
for enhanced user recommendations across various domains.

This package provides:
- Data ingestion and preprocessing
- GenAI-powered feature enrichment
- Advanced embedding generation
- Scalable recommendation algorithms
- Comprehensive evaluation metrics
- Production-ready deployment tools
"""

__version__ = "0.1.0"
__author__ = "Shivam Pandey"
__email__ = "shivampandey15199@gmail.com"
__license__ = "MIT"

from surg.core.surg import SURG
from surg.core.exceptions import SURGException, DataValidationError, ModelError
from surg.data.models import User, Item, UserItemInteraction
from surg.config.settings import SURGConfig

__all__ = [
    "SURG",
    "SURGException",
    "DataValidationError", 
    "ModelError",
    "User",
    "Item",
    "UserItemInteraction",
    "SURGConfig",
]