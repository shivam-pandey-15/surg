"""
Main SURG class providing the primary interface for the recommendation system.

This module contains the core SURG class that orchestrates the entire recommendation
pipeline, from data ingestion through model training to recommendation generation.
The class serves as the main entry point for users and coordinates interactions
between all system components.

Key responsibilities:
- Coordinate data ingestion and preprocessing
- Manage model training and evaluation pipelines
- Generate recommendations for users
- Handle configuration and system setup
- Provide hooks for customization and extension
"""

from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from surg.core.exceptions import SURGException, ModelError
from surg.config.settings import SURGConfig
from surg.data.ingestion import DataIngestionPipeline
from surg.genai.backend import GenAIBackend
from surg.embeddings.generator import EmbeddingGenerator
from surg.recommender.engine import RecommendationEngine
from surg.evaluation.metrics import MetricsCalculator


class RecommendationPipeline(ABC):
    """
    Abstract base class for recommendation pipelines.
    
    Defines the interface that all recommendation pipelines must implement,
    enabling pluggable pipeline architectures for different use cases.
    """
    
    @abstractmethod
    def fit(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
            interactions_df: pd.DataFrame) -> None:
        """
        Train the recommendation pipeline on provided data.
        
        Args:
            users_df: DataFrame containing user information
            items_df: DataFrame containing item information  
            interactions_df: DataFrame containing user-item interactions
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: Any, k: int = 10, 
                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: Identifier for the target user
            k: Number of recommendations to generate
            filters: Optional filters to apply to recommendations
            
        Returns:
            List of recommended items with scores and metadata
        """
        pass


class ModelRegistry:
    """
    Registry for managing trained models and their metadata.
    
    Provides functionality to save, load, and manage different versions
    of trained recommendation models, enabling model versioning and
    A/B testing capabilities.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            storage_path: Optional path for persistent model storage
        """
        pass
    
    def register_model(self, model: Any, name: str, version: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a trained model in the registry.
        
        Args:
            model: The trained model object
            name: Human-readable model name
            version: Model version identifier
            metadata: Optional metadata about the model
        """
        pass
    
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """
        Load a registered model from the registry.
        
        Args:
            name: Model name to load
            version: Specific version to load (latest if not specified)
            
        Returns:
            The loaded model object
        """
        pass


class SURG:
    """
    Main SURG class providing the primary interface for the recommendation system.
    
    This class orchestrates the entire recommendation pipeline, coordinating
    data ingestion, model training, and recommendation generation. It serves
    as the main entry point for users and provides a simple, unified API
    for building and deploying recommendation systems.
    """
    
    def __init__(self, config: Optional[SURGConfig] = None, 
                 genai_backend: Optional[GenAIBackend] = None,
                 pipeline: Optional[RecommendationPipeline] = None):
        """
        Initialize the SURG recommendation system.
        
        Args:
            config: Configuration object (uses defaults if not provided)
            genai_backend: GenAI backend for embedding generation (uses OpenAI if not provided)
            pipeline: Custom recommendation pipeline (uses default if not provided)
        """
        pass
    
    def fit(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
            interactions_df: pd.DataFrame, **kwargs) -> 'SURG':
        """
        Train the recommendation system on provided data.
        
        Coordinates the entire training pipeline including data validation,
        preprocessing, embedding generation, and model training.
        
        Args:
            users_df: DataFrame containing user information and features
            items_df: DataFrame containing item information and features
            interactions_df: DataFrame containing user-item interactions
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    def recommend(self, user_id: Any, k: int = 10, 
                 filters: Optional[Dict[str, Any]] = None,
                 explain: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: Identifier for the target user
            k: Number of recommendations to generate
            filters: Optional filters to apply (e.g., category, price range)
            explain: Whether to include explanation for recommendations
            
        Returns:
            List of recommended items with scores, or dict with explanations if explain=True
        """
        pass
    
    def batch_recommend(self, user_ids: List[Any], k: int = 10,
                       filters: Optional[Dict[str, Any]] = None) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Generate recommendations for multiple users efficiently.
        
        Args:
            user_ids: List of user identifiers
            k: Number of recommendations per user
            filters: Optional filters to apply
            
        Returns:
            Dictionary mapping user_ids to their recommendations
        """
        pass
    
    def evaluate(self, test_data: pd.DataFrame, 
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the recommendation system on test data.
        
        Args:
            test_data: Test interactions for evaluation
            metrics: List of metrics to compute (uses defaults if not provided)
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the trained model and configuration to disk.
        
        Args:
            path: Directory path to save the model
        """
        pass
    
    @classmethod
    def load(cls, path: str) -> 'SURG':
        """
        Load a previously saved SURG model.
        
        Args:
            path: Directory path containing the saved model
            
        Returns:
            Loaded SURG instance
        """
        pass