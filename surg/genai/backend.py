"""
GenAI backend implementations for various AI service providers.

This module defines the abstract backend interface and concrete implementations
for different GenAI providers. The primary implementation uses OpenAI SDK,
with the architecture designed to support additional providers in the future.
"""

from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio

from surg.core.exceptions import GenAIError


@dataclass
class GenAIResponse:
    """
    Standard response format for GenAI operations.
    
    Provides a consistent interface for responses from different GenAI providers,
    including the generated content, metadata, and usage statistics.
    """
    content: str
    model: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def get_token_count(self) -> int:
        """
        Get the total token count for this response.
        
        Returns:
            Total tokens used in the request/response
        """
        pass


@dataclass 
class EmbeddingResponse:
    """
    Response format for embedding generation operations.
    
    Contains the generated embeddings along with metadata about the
    embedding model and generation process.
    """
    embeddings: List[List[float]]
    model: str
    dimensions: int
    usage: Dict[str, Any]
    
    def get_embedding_count(self) -> int:
        """
        Get the number of embeddings in this response.
        
        Returns:
            Number of embeddings generated
        """
        pass


class GenAIBackend(ABC):
    """
    Abstract base class for GenAI backend implementations.
    
    Defines the interface that all GenAI backends must implement,
    enabling pluggable AI service providers while maintaining
    consistent functionality across the recommendation system.
    """
    
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, **kwargs):
        """
        Initialize the GenAI backend.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API endpoint
            **kwargs: Additional backend-specific configuration
        """
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, model: Optional[str] = None,
                     max_tokens: Optional[int] = None,
                     temperature: float = 0.7, **kwargs) -> GenAIResponse:
        """
        Generate text completion using the AI model.
        
        Args:
            prompt: Input prompt for text generation
            model: Model name to use (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            GenAIResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str], 
                          model: Optional[str] = None) -> EmbeddingResponse:
        """
        Generate embeddings for the provided texts.
        
        Args:
            texts: List of texts to generate embeddings for
            model: Embedding model to use (uses default if not specified)
            
        Returns:
            EmbeddingResponse containing the generated embeddings
        """
        pass
    
    @abstractmethod
    async def generate_text_async(self, prompt: str, model: Optional[str] = None,
                                 max_tokens: Optional[int] = None,
                                 temperature: float = 0.7, **kwargs) -> GenAIResponse:
        """
        Asynchronously generate text completion.
        
        Args:
            prompt: Input prompt for text generation
            model: Model name to use
            max_tokens: Maximum tokens to generate  
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            GenAIResponse containing the generated text
        """
        pass
    
    @abstractmethod
    async def generate_embeddings_async(self, texts: List[str],
                                      model: Optional[str] = None) -> EmbeddingResponse:
        """
        Asynchronously generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            EmbeddingResponse containing the embeddings
        """
        pass
    
    def health_check(self) -> bool:
        """
        Check if the backend is healthy and responsive.
        
        Returns:
            True if backend is healthy, False otherwise
        """
        pass


class OpenAIBackend(GenAIBackend):
    """
    OpenAI backend implementation using the official OpenAI SDK.
    
    Provides integration with OpenAI's API for text generation and embedding
    creation. Supports both synchronous and asynchronous operations with
    proper error handling and rate limiting.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 organization: Optional[str] = None,
                 default_model: str = "gpt-3.5-turbo",
                 default_embedding_model: str = "text-embedding-ada-002",
                 **kwargs):
        """
        Initialize the OpenAI backend.
        
        Args:
            api_key: OpenAI API key (reads from environment if not provided)
            base_url: Custom base URL for API endpoint
            organization: OpenAI organization ID
            default_model: Default model for text generation
            default_embedding_model: Default model for embeddings
            **kwargs: Additional OpenAI client configuration
        """
        pass
    
    def generate_text(self, prompt: str, model: Optional[str] = None,
                     max_tokens: Optional[int] = None,
                     temperature: float = 0.7, **kwargs) -> GenAIResponse:
        """
        Generate text using OpenAI's chat completion API.
        
        Args:
            prompt: Input prompt for generation
            model: OpenAI model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters
            
        Returns:
            GenAIResponse with generated text and usage info
        """
        pass
    
    def generate_embeddings(self, texts: List[str],
                          model: Optional[str] = None) -> EmbeddingResponse:
        """
        Generate embeddings using OpenAI's embedding API.
        
        Args:
            texts: Texts to generate embeddings for
            model: OpenAI embedding model name
            
        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        pass
    
    async def generate_text_async(self, prompt: str, model: Optional[str] = None,
                                 max_tokens: Optional[int] = None,
                                 temperature: float = 0.7, **kwargs) -> GenAIResponse:
        """
        Asynchronously generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Token limit
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            GenAIResponse with generated content
        """
        pass
    
    async def generate_embeddings_async(self, texts: List[str],
                                      model: Optional[str] = None) -> EmbeddingResponse:
        """
        Asynchronously generate embeddings using OpenAI API.
        
        Args:
            texts: Texts to embed
            model: Embedding model name
            
        Returns:
            EmbeddingResponse with embeddings
        """
        pass
    
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 100,
                                model: Optional[str] = None) -> EmbeddingResponse:
        """
        Generate embeddings in batches for efficient processing.
        
        Args:
            texts: Large list of texts to embed
            batch_size: Number of texts per batch
            model: Embedding model to use
            
        Returns:
            EmbeddingResponse with all embeddings combined
        """
        pass