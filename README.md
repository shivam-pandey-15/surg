# SURG (Smart User Recommendation with GenAI)

[![PyPI version](https://badge.fury.io/py/surg.svg)](https://badge.fury.io/py/surg)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end Python package for building recommendation systems enhanced with Generative AI. Provide user, item, and user–item interaction data, and the library builds a full pipeline for training, evaluation, and deployment. GenAI is used across layers (feature enrichment, cold-start handling, candidate generation, re-ranking, and tuning) to boost accuracy and performance.

## 🚀 Key Features

- **Data Ingestion**: Flexible data models for users, items, and interactions (implicit/explicit)
- **GenAI Integration**: Seamless OpenAI SDK integration with pluggable backends
- **Embedding Generation**: Advanced embedding creation via LLM/GenAI backends
- **Smart Recommendations**: Candidate generation + neural ranking with GenAI-assisted re-ranking
- **AutoML Pipeline**: Automated training, hyperparameter search, and evaluation
- **Evaluation Suite**: Comprehensive offline metrics and online A/B testing hooks
- **Performance Optimization**: Pluggable backends and intelligent caching for cost/performance control
- **Extensible Architecture**: Modular design for easy customization and extension

## 📦 Installation

### From PyPI
```bash
pip install surg
```

### From Source
```bash
git clone https://github.com/shivam-pandey-15/surg.git
cd surg
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/shivam-pandey-15/surg.git
cd surg
pip install -e ".[dev]"
```

## ⚙️ Configuration

Set up your GenAI backend (OpenAI or compatible):

```bash
export GENAI_API_BASE=https://api.openai.com/v1
export GENAI_API_KEY=your-api-key-here
export GENAI_MODEL=gpt-4  # Optional, defaults to gpt-3.5-turbo
```

## 🏗️ Architecture

SURG follows a modular, plugin-based architecture:

```
surg/
├── core/           # Core abstractions and interfaces
├── data/           # Data models and ingestion pipeline
├── genai/          # GenAI backends and integration
├── embeddings/     # Embedding generation and management
├── recommender/    # Recommendation algorithms and engines
├── evaluation/     # Metrics calculation and A/B testing
├── config/         # Configuration management
├── utils/          # Utilities and helpers
└── cli/            # Command-line interface
```

### Design Principles
- **Decoupled**: Clear interfaces with minimal dependencies
- **Pluggable**: Easy to swap backends and algorithms
- **Configurable**: Environment-driven with sensible defaults
- **Testable**: Comprehensive test coverage
- **Extensible**: Plugin architecture for custom components

## 🎯 Quick Start

```python
from surg import SURG
from surg.data import UserItemInteraction
import pandas as pd

# Initialize SURG with your data
surg = SURG()

# Load your data
users_df = pd.read_csv('users.csv')
items_df = pd.read_csv('items.csv')
interactions_df = pd.read_csv('interactions.csv')

# Train the model
surg.fit(users_df, items_df, interactions_df)

# Get recommendations
recommendations = surg.recommend(user_id=123, k=10)
```

## 📊 Advanced Usage

### Custom GenAI Backend
```python
from surg.genai import OpenAIBackend, CustomBackend

# Use OpenAI (default)
surg = SURG(genai_backend=OpenAIBackend())

# Or use custom backend
surg = SURG(genai_backend=CustomBackend(api_url="your-endpoint"))
```

### Evaluation and Metrics
```python
from surg.evaluation import RecommendationEvaluator

evaluator = RecommendationEvaluator(surg)
metrics = evaluator.evaluate(test_data, metrics=['precision', 'recall', 'ndcg'])
print(metrics)
```

### A/B Testing
```python
from surg.evaluation import ABTest

ab_test = ABTest(model_a=surg_v1, model_b=surg_v2)
results = ab_test.run(test_users, duration_days=7)
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
black surg/
flake8 surg/
mypy surg/
```

### Building Documentation
```bash
cd docs/
make html
```

## 📈 Performance

SURG is designed for production use with:
- Efficient caching mechanisms
- Batch processing capabilities
- Configurable resource limits
- Monitoring and logging integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GenAI capabilities
- The open-source recommendation systems community
- Contributors and early adopters

## 📚 Documentation

For detailed documentation, visit: [https://surg.readthedocs.io](https://surg.readthedocs.io)

## 🐛 Issues

Report bugs and request features at: [https://github.com/shivam-pandey-15/surg/issues](https://github.com/shivam-pandey-15/surg/issues)
