"""
Core Recommendation System Components

This module contains all core recommendation algorithms organized by approach:

1. Traditional Algorithms (traditional/):
   - Memory-based collaborative filtering
   - Content-based filtering  
   - Hybrid traditional approaches
   - LightFM wrapper integration

2. Matrix Factorization (matrix_factorization/):
   - SVD and SVD++ with bias terms
   - Non-negative Matrix Factorization (NMF)
   - Alternating Least Squares (ALS)
   - Factorization Machines (FM)

3. Deep Learning (deep_learning/):
   - Neural Collaborative Filtering variants
   - Autoencoder-based approaches
   - DeepFM for feature interactions
   - Wide & Deep learning models
   - Transformer-based sequential recommendations

All algorithms implement the base interface defined in base.py for:
- Consistent API across different approaches
- Easy algorithm swapping and comparison
- Standardized evaluation and benchmarking
- Integration with GenAI enhancement layers
"""
