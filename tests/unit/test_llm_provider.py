"""
Test script for LLM providers using keys from keys.json

This script tests all LLM providers (OpenAI, Anthropic, Google) with various
capabilities including:
- Basic text generation
- Chat conversations
- Batch generation
- Embeddings
- Usage statistics
"""

import sys
from pathlib import Path

# Add parent directory to path to import surg package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from surg.adapters.llm_providers import LLMProvider, get_llm_client, quick_generate
from surg.utils.config_manager import get_llm_config, load_api_keys
from surg.utils.logger import setup_logging, get_logger


# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def test_basic_generation(provider_name: str):
    """Test basic text generation."""
    logger.section(f"Testing Basic Generation - {provider_name}")
    
    try:
        config = get_llm_config(provider_name)
        
        provider = LLMProvider(
            provider_name=provider_name,
            api_key=config['api_key'],
            model=config['model'],
            temperature=0.7
        )
        
        prompt = "Explain recommendation systems in exactly one sentence."
        logger.info(f"Prompt: {prompt}")
        
        response = provider.generate(prompt)
        
        logger.success(f"Response generated successfully")
        print(f"\n📝 Response: {response.content}\n")
        print(f"📊 Usage: {response.usage}")
        print(f"🏁 Finish Reason: {response.finish_reason}")
        print(f"🤖 Model: {response.model}")
        
        return True
        
    except Exception as e:
        logger.failure(f"Basic generation test failed: {str(e)}")
        return False


def test_system_prompt(provider_name: str):
    """Test generation with system prompt."""
    logger.section(f"Testing System Prompt - {provider_name}")
    
    try:
        config = get_llm_config(provider_name)
        
        provider = LLMProvider(
            provider_name=provider_name,
            api_key=config['api_key'],
            model=config['model']
        )

        system_prompt = "You are a helpful AI assistant specializing in machine learning. Answer only in 1 sentence ."
        user_prompt = "What is collaborative filtering?"
        
        logger.info(f"System: {system_prompt}")
        logger.info(f"User: {user_prompt}")
        
        response = provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5
        )
        
        logger.success(f"Response with system prompt generated")
        print(f"\n📝 Response: {response.content}\n")
        
        return True
        
    except Exception as e:
        logger.failure(f"System prompt test failed: {str(e)}")
        return False


def test_chat_conversation(provider_name: str):
    """Test multi-turn chat conversation."""
    logger.section(f"Testing Chat Conversation - {provider_name}")
    
    try:
        config = get_llm_config(provider_name)
        
        provider = LLMProvider(
            provider_name=provider_name,
            api_key=config['api_key'],
            model=config['model']
        )
        
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hi! My name is Alice.'},
            {'role': 'assistant', 'content': 'Hello Alice! Nice to meet you.'},
            {'role': 'user', 'content': 'What is my name?'}
        ]
        
        logger.info("Testing multi-turn conversation")
        
        response = provider.chat(messages)
        
        logger.success("Chat response received")
        print(f"\n💬 Assistant: {response.content}\n")
        
        return True
        
    except Exception as e:
        logger.failure(f"Chat test failed: {str(e)}")
        return False


def test_batch_generation(provider_name: str):
    """Test batch generation."""
    logger.section(f"Testing Batch Generation - {provider_name}")
    
    try:
        config = get_llm_config(provider_name)
        
        provider = LLMProvider(
            provider_name=provider_name,
            api_key=config['api_key'],
            model=config['model']
        )
        
        prompts = [
            "What is machine learning?",
            "What is deep learning?",
            "What is reinforcement learning?"
        ]
        
        logger.info(f"Generating responses for {len(prompts)} prompts")
        
        responses = provider.batch_generate(
            prompts,
            system_prompt="Answer in one sentence.",
            temperature=0.3
        )
        
        logger.success(f"Batch generation completed")
        
        for i, response in enumerate(responses):
            if response:
                print(f"\n{i+1}. Q: {prompts[i]}")
                print(f"   A: {response.content}")
        
        print()
        return True
        
    except Exception as e:
        logger.failure(f"Batch generation test failed: {str(e)}")
        return False


def test_embeddings(provider_name: str):
    """Test embeddings generation."""
    logger.section(f"Testing Embeddings - {provider_name}")
    
    # Only test with OpenAI and Anthropic as they have reliable embedding APIs
    if provider_name not in ["openai", "anthropic"]:
        logger.info(f"Skipping embeddings test for {provider_name} (limited support)")
        return True
    
    try:
        config = get_llm_config(provider_name)
        
        provider = LLMProvider(
            provider_name=provider_name,
            api_key=config['api_key'],
            model=config['model']
        )
        
        texts = [
            "I love recommendation systems",
            "Machine learning is fascinating"
        ]
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Test with default model
        embeddings = provider.get_embeddings(texts)
        
        logger.success(f"Embeddings generated with default model")
        print(f"\n📊 Embedding dimensions: {len(embeddings[0])}")
        print(f"📊 Number of embeddings: {len(embeddings)}")
        
        # Test with custom model if provider supports it
        if provider_name == "openai":
            custom_embeddings = provider.get_embeddings(
                texts[0], 
                model="text-embedding-3-large"
            )
            print(f"📊 Custom model embedding dimensions: {len(custom_embeddings)}")
        elif provider_name == "anthropic":
            # Test with custom Voyage model
            custom_embeddings = provider.get_embeddings(
                texts[0],
                model="voyage-3.5-lite"
            )
            print(f"📊 Custom Voyage model embedding dimensions: {len(custom_embeddings)}")
        
        print()
        return True
        
    except Exception as e:
        logger.failure(f"Embeddings test failed: {str(e)}")
        return False


def test_usage_stats(provider_name: str):
    """Test usage statistics tracking."""
    logger.section(f"Testing Usage Statistics - {provider_name}")
    
    try:
        config = get_llm_config(provider_name)
        
        provider = LLMProvider(
            provider_name=provider_name,
            api_key=config['api_key'],
            model=config['model']
        )
        
        # Make a few requests
        for i in range(3):
            provider.generate(f"Say the number {i+1}")
        
        stats = provider.get_usage_stats()
        
        logger.success("Usage statistics retrieved")
        print(f"\n📈 Usage Statistics:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Failed Requests: {stats['failed_requests']}")
        print(f"   Total Tokens: {stats['total_tokens_used']}")
        print(f"   Success Rate: {stats['success_rate']:.2%}\n")
        
        return True
        
    except Exception as e:
        logger.failure(f"Usage stats test failed: {str(e)}")
        return False


def test_convenience_functions():
    """Test convenience functions."""
    logger.section("Testing Convenience Functions")
    
    try:
        # Test quick_generate
        logger.info("Testing quick_generate()")
        
        response = quick_generate(
            "What is 2+2?",
            provider="openai",
            model="gpt-4o-mini"
        )
        
        logger.success("quick_generate() worked")
        print(f"\n📝 Response: {response}\n")
        
        # Test get_llm_client
        logger.info("Testing get_llm_client()")
        
        client = get_llm_client("openai", model="gpt-4o-mini")
        response = client.generate("Hello!")
        
        logger.success("get_llm_client() worked")
        print(f"📝 Response: {response.content}\n")
        
        return True
        
    except Exception as e:
        logger.failure(f"Convenience functions test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests for all providers."""
    logger.section("🚀 STARTING LLM PROVIDER TESTS")
    
    # Load available providers
    try:
        keys_config = load_api_keys()
        providers = list(keys_config['llm_providers'].keys())
        logger.info(f"Found providers: {', '.join(providers)}")
    except Exception as e:
        logger.failure(f"Failed to load keys.json: {str(e)}")
        return
    
    results = {}
    
    for provider_name in providers:
        logger.section(f"Testing Provider: {provider_name.upper()}")
        
        provider_results = {
            'basic_generation': test_basic_generation(provider_name),
            'system_prompt': test_system_prompt(provider_name),
            'chat': test_chat_conversation(provider_name),
            'batch': test_batch_generation(provider_name),
            'embeddings': test_embeddings(provider_name),
            'usage_stats': test_usage_stats(provider_name)
        }
        
        results[provider_name] = provider_results
        
        print("\n" + "="*60)
    
    # Test convenience functions (provider-agnostic)
    results['convenience'] = {
        'functions': test_convenience_functions()
    }
    
    # Print summary
    logger.section("📊 TEST SUMMARY")
    
    for provider, tests in results.items():
        print(f"\n{provider.upper()}:")
        for test_name, passed in tests.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
    
    # Calculate overall stats
    total_tests = sum(len(tests) for tests in results.values())
    passed_tests = sum(sum(1 for result in tests.values() if result) for tests in results.values())
    
    print(f"\n{'='*60}")
    print(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM providers")
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'anthropic', 'google', 'all'],
        default='all',
        help='Provider to test (default: all)'
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['basic', 'system', 'chat', 'batch', 'embeddings', 'stats', 'all'],
        default='all',
        help='Specific test to run (default: all)'
    )
    
    args = parser.parse_args()
    
    if args.provider == 'all' and args.test == 'all':
        run_all_tests()
    else:
        # Run specific test for specific provider
        if args.provider == 'all':
            providers = ['openai', 'anthropic', 'google']
        else:
            providers = [args.provider]
        
        test_map = {
            'basic': test_basic_generation,
            'system': test_system_prompt,
            'chat': test_chat_conversation,
            'batch': test_batch_generation,
            'embeddings': test_embeddings,
            'stats': test_usage_stats
        }
        
        if args.test == 'all':
            run_all_tests()
        else:
            for provider in providers:
                test_map[args.test](provider)