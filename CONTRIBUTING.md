# Contributing to Azure ML Model Builder

First off, thank you for considering contributing to Azure ML Model Builder! It's people like you that make this project better.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots if relevant**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and explain which behavior you expected to see instead**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Python style guide (PEP 8)
* Include thoughtfully-worded, well-structured tests
* Document new code
* End all files with a newline

## Development Setup

1. Fork the repo and create your branch from `main`
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/azure-ml-model-builder.git
   cd azure-ml-model-builder
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install development dependencies
   ```

## Coding Guidelines

### Python Style Guide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use 4 spaces for indentation (not tabs)
* Maximum line length is 100 characters
* Use type hints where applicable
* Write docstrings for all public functions, classes, and modules

### Code Formatting

We use several tools to maintain code quality:

```bash
# Format code with Black
black .

# Check code style with flake8
flake8 .

# Type check with mypy
mypy .
```

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Example:
```
Add batch prediction functionality

- Implement batch_predict() method
- Add tests for batch prediction
- Update documentation

Closes #123
```

### Testing

* Write unit tests for new features
* Ensure all tests pass before submitting PR
* Aim for high code coverage

Run tests:
```bash
pytest tests/
pytest --cov=. --cov-report=html
```

## Project Structure

```
azure-ml-model-builder/
├── automl_driver.py          # Main driver (legacy)
├── automl_driver_refactored.py  # Refactored driver
├── scripts/                   # Utility scripts
│   ├── train_model.py
│   ├── deploy_model.py
│   └── test_model.py
├── outputs/                   # Generated artifacts
├── explanation/               # Model interpretability
├── tests/                     # Test files
├── docs/                      # Documentation
└── requirements.txt           # Dependencies
```

## Documentation

* Update the README.md if you change functionality
* Add docstrings to all public APIs
* Update the docs/ folder for major changes
* Comment your code where necessary

## Review Process

1. Create a pull request
2. Ensure CI/CD checks pass
3. Request review from maintainers
4. Address review comments
5. Once approved, a maintainer will merge your PR

## Recognition

Contributors will be recognized in:
* The project README
* Release notes
* The project's contributors page

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Azure ML Model Builder! 🎉
