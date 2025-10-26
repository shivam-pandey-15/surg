"""
GenAI Analysis and Enhancement Module

This module provides AI-powered analysis capabilities that enhance the traditional
data analysis with intelligent insights, natural language descriptions, and
strategic planning for recommendation systems.

Enhanced from content generation to include:
- AI-powered dataset analysis and insights
- Intelligent problem interpretation and planning
- Recursive reasoning for complex problems
- Natural language insight generation
- Strategic planning for algorithm selection

Key Components:
- LLMEnhancer: Core LLM integration for recommendation enhancement
- ExplanationGenerator: Natural language explanation generation
- AdaptiveOptimizer: AI-driven hyperparameter and strategy optimization
- GenAIAnalyzer: AI-powered dataset analysis and insights
- ThinkingEngine: Recursive reasoning for complex problems
- ProblemUnderstanding: Intelligent problem interpretation
- StrategyPlanner: AI-driven approach planning
- InsightGenerator: Natural language insight generation

Features:
- Natural language descriptions of data patterns
- Intelligent problem type detection with reasoning
- Strategic planning for algorithm selection
- Recursive thinking for complex optimization problems
- Contextual insights based on domain knowledge
- Automated hypothesis generation and testing
- Risk assessment and mitigation planning
- Performance prediction and optimization suggestions

Integration with Traditional Analysis:
The GenAI layer enhances traditional statistical analysis by:
1. Providing context and domain expertise
2. Generating human-readable insights
3. Suggesting creative problem-solving approaches
4. Identifying non-obvious patterns and relationships
5. Planning comprehensive evaluation strategies
6. Recommending experimental designs

This module enables SURG to provide intelligent, adaptive recommendations
that go beyond traditional algorithmic approaches.
"""

from .analyzer import GenAIAnalyzer
from .thinking_engine import ThinkingEngine

__all__ = [
    'GenAIAnalyzer',
    'ThinkingEngine'
]