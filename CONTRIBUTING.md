# Contributing to Xetrack

Thank you for your interest in contributing to Xetrack! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites
- Python 3.9 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Setting up the Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xdssio/xetrack.git
   cd xetrack
   ```

2. **Install dependencies with Poetry**:
   ```bash
   poetry install --with dev
   ```

3. **Install optional dependencies for full functionality**:
   ```bash
   poetry install --extras "duckdb assets bashplotlib"
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

### Running Tests

We use pytest for testing:

```bash
# Run all tests
poetry run pytest tests/ -v

# Run tests with fast feedback (testmon)
poetry run pytest -x -q -p no:warnings --testmon tests

# Run tests with coverage
poetry run pytest --cov=xetrack --cov-report=html tests

# Run specific test file
poetry run pytest tests/tracker_test.py -v

# Run specific test function
poetry run pytest tests/cli_test.py::test_specific_function -v
```

### Code Quality

We maintain high code quality standards:

- **Type hints**: Use type hints throughout the codebase
- **Documentation**: Add docstrings to all public functions and classes
- **Testing**: Write tests for new functionality

#### Optional Code Formatting
While not enforced, we recommend:
```bash
# Format code with black (if installed)
poetry run black xetrack tests

# Type checking with mypy (if installed)
poetry run mypy xetrack --ignore-missing-imports
```

## Project Structure

```
xetrack/
├── xetrack/           # Main package
│   ├── tracker.py     # Core tracking functionality
│   ├── engine.py      # Database abstraction layer
│   ├── reader.py      # Data reading and analysis
│   ├── cli.py         # Command-line interface
│   ├── assets.py      # Asset management
│   └── ...
├── tests/             # Test suite
├── .github/           # GitHub workflows
├── pyproject.toml     # Poetry configuration
└── README.md          # Project documentation
```

## Making Contributions

### Types of Contributions

- **Bug Reports**: Found a bug? Please create an issue with detailed reproduction steps
- **Feature Requests**: Have an idea? Open an issue to discuss it first
- **Code Contributions**: Bug fixes, new features, documentation improvements
- **Documentation**: Help improve our docs, examples, and guides

### Pull Request Process

1. **Fork the repository** and create your branch from `main`

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Make your changes**:
   - Follow existing code patterns and style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**:
   ```bash
   poetry run pytest tests/ -v
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "feat: add new feature description"
   # or
   git commit -m "fix: resolve issue with specific component"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/) format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**:
   - Provide a clear title and description
   - Link any related issues
   - Include screenshots if relevant

### Development Guidelines

#### Database Engine Support
- Test both SQLite and DuckDB engines when making changes
- Handle engine-specific differences in table naming
- Use the engine abstraction layer properly

#### CLI Development
- All CLI commands should support both engines via `--engine` flag
- Test CLI functionality in CI
- Maintain backward compatibility

#### Asset Management
- Asset functionality is optional (requires sqlitedict)
- Gracefully handle missing dependencies
- Test with and without asset functionality

#### Testing Strategy
- Write unit tests for new functions
- Add integration tests for workflows
- Test optional dependencies separately
- Include CLI testing where relevant

## Architecture Guidelines

### Core Components

1. **Tracker**: Main interface for experiment tracking
2. **Engine Layer**: Database abstraction (SQLite/DuckDB)
3. **Reader**: Read-only data access interface
4. **CLI**: Command-line interface using Typer
5. **Assets**: Object storage with deduplication

### Key Design Patterns

- **Polymorphic Engine Design**: Use abstract base classes
- **Dynamic Schema**: Support automatic column addition
- **Asset Deduplication**: Store complex objects efficiently
- **Multiprocess Safety**: Ensure thread-safe operations

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Documentation**: Check the README and inline documentation

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Focus on technical merit

## License

By contributing to Xetrack, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Xetrack! 🚀