"""
Enhanced SURG Recommender with Data-Driven Capabilities

This is the main interface for the SURG recommendation system, enhanced with
data-driven initialization and GenAI integration while maintaining backward
compatibility with the existing algorithm-first approach.

New Features:
- Optional dataset initialization for data-driven workflows  
- Automatic problem type detection and analysis
- GenAI-powered insights and planning when enabled
- Recursive thinking mode for complex problem solving
- MCP tool integration for extensible processing
- Natural language descriptions and explanations

Usage Patterns:
1. Traditional (Algorithm-first):
   rec = SURGRecommender(algorithm='lightfm')
   rec.fit(interactions, user_features, item_features)
   
2. Data-driven (Dataset-first):
   rec = SURGRecommender(dataset=df, enable_genai=True)
   analysis = rec.analyze()
   rec.understand_problem("Predict movie ratings for personalization")
   rec.fit()
   
3. AI-Enhanced:
   rec = SURGRecommender(dataset=df, enable_genai=True, thinking_mode=True)
   plan = rec.think_and_plan("Maximize user engagement while ensuring diversity")
   rec.fit(execution_plan=plan)

The system automatically adapts to the provided inputs and maintains full
backward compatibility with existing implementations.
"""
