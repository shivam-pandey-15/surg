#!/bin/bash

# SURG Example Runner Script
# 
# This script provides easy commands to run various SURG examples with
# automatic environment setup and dataset downloading.
#
# Usage:
#   ./run_examples.sh basic                    # Run basic example
#   ./run_examples.sh advanced benchmark       # Run advanced benchmark
#   ./run_examples.sh custom                   # Run custom GenAI layer example
#   ./run_examples.sh setup                    # Setup environment only

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Using Python $python_version"
    
    if [[ $(echo "$python_version >= 3.9" | bc -l) -eq 0 ]]; then
        print_warning "Python 3.9+ is recommended. Current: $python_version"
    fi
}

# Setup virtual environment and install dependencies
setup_environment() {
    print_info "Setting up environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install SURG package in development mode
    print_info "Installing SURG package..."
    pip install -e .
    
    # Install optional dependencies based on use case
    print_info "Installing optional dependencies..."
    pip install -e ".[dev,api]"
    
    # Check if GPU is available and install GPU dependencies
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU detected, installing GPU dependencies..."
        pip install -r requirements-gpu.txt
    else
        print_info "No GPU detected, using CPU-only dependencies"
    fi
    
    print_success "Environment setup complete!"
}

# Download and cache datasets
download_datasets() {
    print_info "Downloading example datasets..."
    
    # Create data directory
    mkdir -p data/cache
    
    # Run dataset download script
    python3 -c "
import sys
sys.path.insert(0, '.')
from surg.datasets import download_sample_datasets
download_sample_datasets()
"
    
    print_success "Datasets downloaded and cached!"
}

# Run basic example
run_basic_example() {
    print_info "Running basic SURG example with MovieLens..."
    
    # Ensure environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Run with different configurations
    echo ""
    print_info "Example 1: Basic usage with MovieLens 100K"
    python3 examples/basic_usage.py --dataset ml-100k --n_users 500
    
    echo ""
    print_info "Example 2: With GenAI enhancements (if API keys configured)"
    python3 examples/basic_usage.py --dataset ml-1m --n_users 1000 --enable_genai
    
    echo ""
    print_info "Example 3: Larger dataset for performance testing"
    python3 examples/basic_usage.py --dataset ml-1m --n_users 5000 --n_recommendations 20
    
    print_success "Basic examples completed!"
}

# Run advanced pipeline examples
run_advanced_example() {
    local mode=${1:-"benchmark"}
    
    print_info "Running advanced pipeline in $mode mode..."
    
    # Ensure environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    case $mode in
        "benchmark")
            print_info "Running multi-algorithm benchmark..."
            python3 examples/advanced_pipeline.py --mode benchmark \
                --algorithms lightfm svd neural_cf surg \
                --datasets movielens-1m amazon-books
            ;;
        "optimize")
            print_info "Running hyperparameter optimization..."
            python3 examples/advanced_pipeline.py --mode optimize \
                --algorithm lightfm --dataset movielens-1m --n_trials 50
            ;;
        "deploy")
            print_info "Deploying model for serving..."
            # First train and save a model
            python3 -c "
from surg import SURGRecommender
from surg.datasets import load_movielens
import pickle

# Quick training for demo
data = load_movielens('ml-100k')
recommender = SURGRecommender()
recommender.fit(data['interactions'])

# Save model
with open('demo_model.pkl', 'wb') as f:
    pickle.dump(recommender, f)
print('Demo model saved!')
"
            python3 examples/advanced_pipeline.py --mode deploy --model_path demo_model.pkl
            ;;
        *)
            print_error "Unknown advanced mode: $mode"
            echo "Available modes: benchmark, optimize, deploy"
            exit 1
            ;;
    esac
    
    print_success "Advanced pipeline example completed!"
}

# Run custom GenAI layer example
run_custom_example() {
    print_info "Running custom GenAI layer example..."
    
    # Ensure environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Check if API keys are configured
    if [[ -z "$OPENAI_API_KEY" && -z "$ANTHROPIC_API_KEY" ]]; then
        print_warning "No GenAI API keys found in environment variables"
        print_info "Running with local/mock implementations..."
        python3 examples/custom_genai_layer.py --backend local
    else
        print_info "Using configured API keys for GenAI services"
        python3 examples/custom_genai_layer.py --backend openai
    fi
    
    print_success "Custom GenAI layer example completed!"
}

# Function to run benchmarking example
run_benchmarking_example() {
    print_info "🚀 Running Benchmarking Example"
    
    # Setup if needed
    setup_environment
    
    # Run benchmarking example
    cd "$(dirname "$0")"
    python3 examples/benchmarking_example.py \
        --suite quick \
        --algorithms lightfm,svd \
        --output-dir benchmark_results
    
    print_success "Benchmarking example completed! Check benchmark_results/ for detailed analysis."
}

# Function to run deployment example
run_deployment_example() {
    print_info "🚀 Running Deployment Example"
    
    # Setup if needed
    setup_environment
    
    # Run deployment example
    cd "$(dirname "$0")"
    python3 examples/deployment_example.py --mode api
    
    print_success "Deployment example completed! Check demo_*_deployment/ for configuration files."
}
run_all_examples() {
    print_info "Running all SURG examples..."
    
    run_basic_example
    echo ""
    run_advanced_example "benchmark"
    echo ""
    run_custom_example
    
    print_success "All examples completed successfully! 🎉"
}

# Show usage information
show_usage() {
    echo "SURG Examples Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup                     Setup environment and install dependencies"
    echo "  basic                     Run basic MovieLens example"
    echo "  advanced <mode>           Run advanced pipeline (benchmark|optimize|deploy)"
    echo "  custom                    Run custom GenAI layer example"
    echo "  benchmark                 Run performance benchmarking example"
    echo "  deploy                    Run deployment configuration example"
    echo "  all                       Run all examples"
    echo "  download                  Download sample datasets"
    echo "  clean                     Clean up generated files"
    echo ""
    echo "Examples:"
    echo "  $0 setup                  # First time setup"
    echo "  $0 basic                  # Quick demo"
    echo "  $0 advanced benchmark     # Compare algorithms"
    echo "  $0 advanced optimize      # Optimize hyperparameters"
    echo "  $0 custom                 # GenAI integration demo"
    echo "  $0 benchmark              # Performance benchmarking"
    echo "  $0 deploy                 # Deployment configuration"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY           OpenAI API key for GenAI features"
    echo "  ANTHROPIC_API_KEY        Anthropic API key for GenAI features"
    echo "  SURG_DATA_DIR            Custom data directory (default: ./data)"
}

# Clean up generated files
clean_up() {
    print_info "Cleaning up generated files..."
    
    # Remove generated models and logs
    rm -f *.pkl *.log
    rm -f benchmark_results_*.json
    rm -f best_params_*.json
    
    # Clean cache directories
    rm -rf __pycache__
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Cleanup completed!"
}

# Main script logic
main() {
    case ${1:-""} in
        "setup")
            check_python
            setup_environment
            download_datasets
            ;;
        "basic")
            check_python
            run_basic_example
            ;;
        "advanced")
            check_python
            run_advanced_example $2
            ;;
        "custom")
            check_python
            run_custom_example
            ;;
        "benchmark")
            run_benchmarking_example
            ;;
        "deploy")
            run_deployment_example
            ;;
        "all")
            check_python
            setup_environment
            download_datasets
            run_basic_example
            run_advanced_example
            run_custom_example
            run_benchmarking_example
            run_deployment_example
            print_success "🎉 All examples completed successfully!"
            ;;
        "download")
            download_datasets
            ;;
        "clean")
            clean_up
            ;;
        "help"|"--help"|"-h"|"")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"