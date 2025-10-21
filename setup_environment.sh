#!/bin/bash

# SURG Development Environment Setup Script
# Run this script to set up the complete development environment

set -e  # Exit on any error

echo "🚀 Setting up SURG development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "surg" ]; then
    echo -e "${RED}❌ Error: Please run this script from the SURG project root directory${NC}"
    exit 1
fi

echo "📍 Current directory: $(pwd)"

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${RED}❌ Error: Python 3.8+ is required${NC}"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "surg-env" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Removing old one...${NC}"
    rm -rf surg-env
fi

python3 -m venv surg-env
echo -e "${GREEN}✅ Virtual environment created${NC}"

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source surg-env/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install SURG with development dependencies
echo "📚 Installing SURG with development dependencies..."
pip install -e ".[dev]"

# Set up environment file
echo "⚙️  Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file from template${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env file and add your OpenAI API key${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists${NC}"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data models cache logs
echo -e "${GREEN}✅ Directories created${NC}"

# Set up pre-commit hooks (if in development mode)
if pip list | grep -q "pre-commit"; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
    echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"
fi

# Verify installation
echo "🧪 Verifying installation..."
python -c "
import surg
print(f'SURG version: {surg.__version__}')

from surg import SURG, SURGConfig
print('✅ Core imports successful')

try:
    config = SURGConfig.from_environment()
    print('✅ Configuration loading successful')
except Exception as e:
    print(f'⚠️  Configuration warning: {e}')
"

# Check installed packages
echo "📋 Installed packages:"
pip list | grep -E "(surg|pandas|numpy|openai|scikit-learn|pytest)"

echo ""
echo -e "${GREEN}🎉 SURG development environment setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key:"
echo "   nano .env"
echo ""
echo "2. Activate the environment (for future sessions):"
echo "   source surg-env/bin/activate"
echo ""
echo "3. Test the installation:"
echo "   python examples/basic_usage.py  # (when examples are created)"
echo ""
echo "4. Run tests:"
echo "   pytest tests/  # (when tests are implemented)"
echo ""
echo "5. Start developing! 🚀"