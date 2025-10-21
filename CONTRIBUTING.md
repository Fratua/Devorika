# Contributing to Devorika

Thank you for your interest in contributing to Devorika! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive criticism
- Respect differing viewpoints

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Use the bug report template
3. Include:
   - Detailed description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Relevant logs

### Suggesting Features

1. Check existing feature requests
2. Explain the use case
3. Describe the proposed solution
4. Consider alternatives

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write tests
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/devorika.git
cd devorika

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Maximum line length: 100 characters

### Example

```python
def process_task(task: str, timeout: int = 30) -> str:
    """
    Process a task with the given timeout.

    Args:
        task: The task description
        timeout: Maximum execution time in seconds

    Returns:
        The result of the task execution

    Raises:
        TimeoutError: If the task exceeds the timeout
    """
    pass
```

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use pytest for testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=devorika tests/
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions/classes
- Include examples for new features

## Project Structure

```
devorika/
├── devorika/          # Main package
│   ├── core/          # Core functionality
│   ├── agents/        # Specialized agents
│   ├── tools/         # Tool implementations
│   └── cli.py         # CLI interface
├── tests/             # Test files
├── examples/          # Usage examples
└── docs/              # Documentation
```

## Adding New Tools

1. Create a new tool class in `devorika/tools/`
2. Inherit from `Tool` base class
3. Implement the `execute` method
4. Register the tool in `DevorikaAgent._register_tools()`
5. Write tests
6. Update documentation

Example:

```python
from devorika.tools.base import Tool

class MyNewTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "A parameter"}
        },
        "required": ["param"]
    }

    def execute(self, param: str) -> str:
        # Implementation
        return f"Result: {param}"
```

## Adding New Agents

1. Create in `devorika/agents/`
2. Inherit from `DevorikaAgent`
3. Override `execute` to add specialization
4. Add to `Orchestrator` if needed
5. Write tests
6. Document usage

## Release Process

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Create a git tag
4. Push to GitHub
5. Create a release on GitHub
6. Publish to PyPI (maintainers only)

## Questions?

- Open an issue for questions
- Join our Discord community
- Email: dev@devorika.ai

Thank you for contributing to Devorika!
