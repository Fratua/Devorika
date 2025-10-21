# Devorika Quick Start Guide

Get up and running with Devorika in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- API key from Anthropic (Claude) or OpenAI (GPT)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/devorika.git
cd devorika

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Devorika
pip install -e .
```

## Configuration

```bash
# 1. Copy the example environment file
cp .env.example .env

# 2. Edit .env and add your API key
# For Claude (recommended):
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Or for GPT:
OPENAI_API_KEY=your_openai_api_key_here
```

## Your First Task

Execute a simple task:

```bash
devorika execute "Create a Python function that checks if a number is prime"
```

## Try Different Modes

### Chat Mode
Interactive conversation with Devorika:

```bash
devorika chat
```

Then type your questions or tasks.

### Pipeline Mode
High-quality multi-stage execution:

```bash
devorika pipeline "Create a todo list API"
```

This runs through: Planning → Coding → Testing → Review → Documentation

### Parallel Mode
Execute multiple tasks simultaneously:

```bash
devorika parallel \
  "coder:Create user model" \
  "tester:Write user tests" \
  "documenter:Document user API"
```

## Python API

```python
from devorika import DevorikaAgent

# Create an agent
agent = DevorikaAgent(verbose=True)

# Execute a task
result = agent.execute("Create a web scraper for news articles")
print(result)

# Chat mode
response = agent.chat("What are Python decorators?")
print(response)
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/](examples/) for more usage examples
- Join our community for support

## Troubleshooting

### API Key Issues

```bash
# Make sure your API key is set
echo $ANTHROPIC_API_KEY
# or
echo $OPENAI_API_KEY
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .
```

### Package Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Common Commands

```bash
# Execute a task
devorika execute "your task here"

# Chat mode
devorika chat

# Pipeline mode
devorika pipeline "your task here"

# Check memory stats
devorika memory stats

# View history
devorika memory history

# Show version
devorika version
```

## What Makes Devorika Special?

1. **Multi-LLM Support**: Uses the best model for each task
2. **Parallel Execution**: Runs multiple tasks simultaneously
3. **Learning**: Remembers solutions and improves over time
4. **Self-Debugging**: Fixes its own errors autonomously
5. **Open Source**: Fully transparent and customizable

## Example Tasks to Try

```bash
# Code generation
devorika execute "Create a Flask REST API with CRUD operations"

# Debugging
devorika execute "Analyze this Python file for bugs: app.py"

# Refactoring
devorika execute "Refactor this code to use design patterns: legacy.py"

# Testing
devorika execute "Generate comprehensive tests for my authentication module"

# Documentation
devorika execute "Document this API and create a README"
```

## Get Help

- Documentation: README.md
- Examples: examples/basic_usage.py
- Issues: GitHub Issues
- Community: Discord

Happy coding with Devorika! 🚀
