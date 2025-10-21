# SURG Documentation

This directory contains the documentation source files for the SURG recommendation system.

## Documentation Structure

- `api/` - API reference documentation
- `tutorials/` - Step-by-step tutorials
- `guides/` - How-to guides and best practices
- `architecture/` - System architecture documentation
- `deployment/` - Deployment guides

## Building Documentation

1. Install documentation dependencies:
   ```bash
   pip install surg[dev]
   ```

2. Build the documentation:
   ```bash
   cd docs/
   make html
   ```

3. View the documentation:
   ```bash
   open _build/html/index.html
   ```

## Contributing to Documentation

Please follow these guidelines when contributing to documentation:

1. Use clear, concise language
2. Include code examples where appropriate
3. Test all code examples before submitting
4. Follow the existing documentation structure
5. Update the table of contents when adding new sections