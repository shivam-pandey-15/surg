"""
Adapters - LLM Providers

This module provides adapters for various LLM providers
(OpenAI, Anthropic, Hugging Face, etc.) with unified interfaces.
"""

import os
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass,field
import time
from enum import Enum
import json

try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI = None
    AzureOpenAI = None

from surg.utils.logger import get_logger

logger = get_logger(__name__)

class ProviderType(Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    VLLM = "vllm"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    CUSTOM = "custom"

@dataclass
class LLMResponse:
    """
    Structured response from LLM generation.
    
    Attributes:
        content: Generated text content
        model: Model used for generation
        provider: Provider name
        usage: Token usage statistics
        finish_reason: Reason for completion (stop, length, etc.)
        metadata: Additional response metadata
        raw_response: Raw response object from provider
    """
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    raw_response: Optional[Any] = None

    def __str__(self) -> str:
        """String representation of the LLM response."""
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert the LLM response to a dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
            "raw_response": self.raw_response,
        }

class LLMProvider:
    """
    Unified LLM provider interface using OpenAI SDK.
    
    Connects to multiple LLM providers through a common interface,
    handling API differences, retries, rate limiting, and error handling.
    """

    PROVIDER_CONFIGS = {
        ProviderType.OPENAI: {
            'base_url': 'https://api.openai.com/v1',
            'default_model': 'gpt-4.1',
            'env_var': 'OPENAI_API_KEY'
        },
        ProviderType.ANTHROPIC: {
            'base_url': 'https://api.anthropic.com/v1',
            'default_model': 'claude-sonnet-4-5',
            'env_var': 'ANTHROPIC_API_KEY',
            'headers': {'anthropic-version': '2023-06-01'}
        },
        ProviderType.GOOGLE: {
            'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
            'default_model': 'gemini-2.5-pro',
            'env_var': 'GOOGLE_API_KEY'
        },
        ProviderType.OLLAMA: {
            'base_url': 'http://localhost:11434/v1',
            'default_model': 'llama2',
            'env_var': None
        },
        ProviderType.LM_STUDIO: {
            'base_url': 'http://localhost:1234/v1',
            'default_model': 'local-model',
            'env_var': None
        },
        ProviderType.TOGETHER: {
            'base_url': 'https://api.together.xyz/v1',
            'default_model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'env_var': 'TOGETHER_API_KEY'
        }
    }

    def __init__(self,
                 provider_name: Union[str, ProviderType] = "openai",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 timeout: int = 60,
                 max_retries: int = 3,
                 organization: Optional[str] = None,
                 **kwargs
                ):
        """
        Initialize LLM provider.
        
        Args:
            provider_name: Provider type (openai, anthropic, google, ollama, etc.)
            api_key: API key (if None, reads from environment variable)
            base_url: Custom base UR
            f None, uses provider default)
            model: Model name (if None, uses provider default)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            organization: OpenAI organization ID (optional)
            **kwargs: Additional provider-specific parameters
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK  not installed. Install with: pip install openai"
            )

        if isinstance(provider_name,str):
            try:
                self.provider = ProviderType(provider_name.lower())
            except ValueError:
                logger.warning(f"Unknown provider '{provider_name}', using as custom")
        else:
            self.provider = provider_name
        
        self.config = self.PROVIDER_CONFIGS.get(self.provider, {'base_url': base_url, 'default_model': model, 'env_var': None})

        if api_key is None and self.config.get('env_var'):
            api_key = os.getenv(self.config['env_var'])
            if not api_key:
                logger.warning(
                    f"No API key provided and {self.config['env_var']} not set. "
                    f"Will only work with local providers."
                )
        
        self.api_key = api_key
        self.base_url = base_url or self.config.get('base_url')
        self.model = model or self.config.get('default_model')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.organization = organization
        
        # Additional parameters
        self.extra_params = kwargs
        
        # Initialize OpenAI client
        self._init_client()

        # Usage tracking
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        logger.info(
            f"Initialized LLM provider",
            provider=self.provider.value,
            model=self.model,
            base_url=self.base_url
        )

    def _init_client(self) -> None:
        """Initialize OpenAI client based on provider type"""
        client_kwargs = {
            'api_key': self.api_key or 'dummy-key',  # Dummy key for local providers
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }

        if self.base_url:
            client_kwargs['base_url'] = self.base_url
        
        if self.organization:
            client_kwargs['organization'] = self.organization
        
        # Add provider-specific headers
        if 'headers' in self.config:
            client_kwargs['default_headers'] = self.config['headers']
        
        # Create client
        if self.provider == ProviderType.AZURE:
            # Azure OpenAI requires different initialization
            self.client = AzureOpenAI(**client_kwargs)
        else:
            self.client = OpenAI(**client_kwargs)
        
        logger.debug(f"OpenAI client initialized for {self.provider.value}")

    def generate(self,
                 prompt: Union[str, List[Dict[str,str]]],
                 system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate completion from the LLM.
        Args:
            prompt: User prompt (string or message list)
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional generation parameters
        Returns:
            LLMResponse object
        Example:
            >>> provider = LLMProvider("openai")
            >>> response = provider.generate("Tell me a joke.")
            >>> print(response.content)
        """
        messages = self._build_messages(prompt, system_prompt)
        params = {
            'model': kwargs.pop('model', self.model),
            'messages': messages,
            'temperature': temperature if temperature is not None else self.temperature
        }
        if max_tokens is not None or self.max_tokens is not None:
            params['max_tokens'] = max_tokens or self.max_tokens
        params.update(self.extra_params)
        params.update(kwargs)
        logger.debug(
            f"Generating completion",
            model=params['model'],
            temperature=params['temperature']
        )
        try:
            response = self._call_with_retry(params)
            return self._parse_response(response, params['model'])
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Generation failed", error=str(e), provider=self.provider.value)
            raise
    def _parse_response(self, response: Any, model: str) -> LLMResponse:
        """
        Parse API response into LLMResponse object.
        """
        # OpenAI/Anthropic/Google compatible response parsing
        try:
            content = response.choices[0].message.content if hasattr(response.choices[0], "message") else getattr(response.choices[0], "text", "")
        except Exception:
            content = ""
        usage = getattr(response, "usage", None)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        metadata = {
            "id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", model),
        }
        return LLMResponse(
            content=content,
            model=model,
            provider=self.provider.value,
            usage=usage,
            finish_reason=finish_reason,
            metadata=metadata,
            raw_response=response
        )

    # Have to check for assistant role in messages for OpenAI chat completion
    def _build_messages(self,
        prompt: Union[str, List[Dict[str,str]]],
        system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Build message list from prompt and system prompt.
        
        Args:
            prompt: User prompt (string or message list)
            system_prompt: Optional system prompt
        Returns:
            List of message dicts for LLM input
        """

        if isinstance(prompt, list):
            messages = prompt.copy()
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            return messages
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return messages
    
    def _call_with_retry(self, params: Dict[str, Any]) -> Any:
        """Call the LLM API with retry logic."""
        attempt = 0
        last_error = None
        while attempt < self.max_retries:
            try:
                response = self.client.chat.completions.create(**params)
                self.total_requests += 1
                return response
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    self.failed_requests += 1
                    logger.error(f"LLM request failed after {self.max_retries} attempts: {e}")
                    raise last_error
        raise last_error

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Multi-turn chat completion.
        Args:
            messages: List of message dicts (role/content)
            **kwargs: Additional generation parameters
        Returns:
            LLMResponse object
        Example:
            >>> provider = LLMProvider("openai")
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Tell me a joke."},
            ...     {"role": "assistant", "content": "Tell me another joke."}
            ... ]
            >>> response = provider.chat(messages)
            >>> print(response.content)
        """
        return self.generate(prompt=messages, **kwargs)

    def batch_generate(self,
                       prompts: List[Union[str, List[Dict[str,str]]]],
                       system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> List[LLMResponse]:
        """
        Batch generation for multiple prompts.
        
        Args:
            prompts: List of user prompts (strings or message lists)
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional generation parameters
        """
        responses = []
        for i, prompt in enumerate(prompts):
            logger.progress(f"Generating for prompt", i + 1, len(prompts))
            try:
                response = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Generation failed for prompt {i+1}: {str(e)}")
                responses.append(LLMResponse(
                    content="",
                    model=self.model,
                    provider=self.provider.value if isinstance(self.provider, ProviderType) else str(self.provider),
                    metadata={"error": str(e)}
                ))
        return responses
    
    def get_embeddings(self,
                       inputs: Union[str, List[str]],
                       model: Optional[str] = None,
                       **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for input text(s).
        
        Args:
            inputs: Single string or list of input strings
            model: Embedding model name (if None, uses provider default)
            **kwargs: Additional embedding parameters
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        is_single = isinstance(inputs, str)
        if is_single:
            inputs = [inputs]
            
        # Default embedding models by provider
        embedding_models = {
            ProviderType.OPENAI: "text-embedding-3-small",
            ProviderType.ANTHROPIC: "voyage-3.5-lite",
            ProviderType.GOOGLE: "text-embedding-004",
            ProviderType.TOGETHER: "togethercomputer/m2-bert-80M-8k-retrieval"
        }
        
        model_to_use = model or embedding_models.get(self.provider, "text-embedding-3-small")
        
        try:
            response = self.client.embeddings.create(
                model=model_to_use,
                input=inputs,
                **kwargs
            )
            embeddings = [data.embedding for data in response.data]
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting (uses provider default if None)
            
        Returns:
            Estimated token count
            
        Note:
            This is an approximation. Actual token count may vary.
        """
        # Simple approximation: ~4 characters per token
        # For accurate counting, use tiktoken library
        return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'total_tokens_used': self.total_tokens_used,
            'success_rate': (
                (self.total_requests - self.failed_requests) / self.total_requests
                if self.total_requests > 0 else 0.0
            )
        }
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        logger.info("Usage statistics reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMProvider(provider={self.provider.value}, "
            f"model={self.model}, "
            f"requests={self.total_requests})"
        )

def get_llm_client(provider_name: Union[str, ProviderType] = "openai",
                   api_key: Optional[str] = None,
                   model: Optional[str] = None,
                   **kwargs) -> LLMProvider:
    """
    Get an LLM client with default configurations.
    Args:
        provider_name: Provider type (openai, anthropic, google, ollama, etc.)
        api_key: API key (if None, reads from environment variable)
        model: Model name (if None, uses provider default)
        **kwargs: Additional provider-specific parameters
    Returns:
        Configured LLMProvider instance
    Example:
        >>> client = get_llm_client("openai", model="gpt-4o-mini")  
        >>> response = client.generate("Your prompt here")  
    """ 
    return LLMProvider(
        provider_name=provider_name,
        api_key=api_key,
        model=model,
        **kwargs
    )

def quick_generate(prompt: str,
                   provider_name: Union[str, ProviderType] = "openai",
                   api_key: Optional[str] = None,
                   model: Optional[str] = None,
                   **kwargs) -> LLMResponse:
    """
    Quick generate a response from the LLM with minimal setup.
    
    Args:
        prompt: User prompt string
        provider_name: Provider type (openai, anthropic, google, ollama, etc.)
        api_key: API key (if None, reads from environment variable)
        model: Model name (if None, uses provider default)
        **kwargs: Additional generation parameters
        
    Returns:
        LLMResponse object
        
    Example:
        >>> response = quick_generate("Explain quantum computing", provider_name="openai", model="gpt-4o-mini")
        >>> print(response.content)
    """
    llm_client = get_llm_client(
        provider_name=provider_name,
        api_key=api_key,
        model=model
    )
    response = llm_client.generate(prompt=prompt, **kwargs)
    return response.content


# Example usage
if __name__ == "__main__":
    from surg.utils.logger import setup_logging
    
    setup_logging(log_level="INFO")
    
    # Test with OpenAI
    print("Testing OpenAI provider...")
    openai_client = LLMProvider(
        provider_name="openai",
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    response = openai_client.generate(
        "Explain recommendation systems in one sentence"
    )
    print(f"Response: {response.content}")
    print(f"Usage: {response.usage}")
    
    # Test usage stats
    stats = openai_client.get_usage_stats()
    print(f"\nUsage Stats: {stats}")
    
    print("\n✅ LLM provider testing complete!")