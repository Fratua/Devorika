"""
Devorika - Advanced AI Software Programmer
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="devorika",
    version="1.0.0",
    author="Devorika Team",
    author_email="contact@devorika.ai",
    description="Advanced AI Software Programmer - Superior to Devin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/devorika",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "duckduckgo-search>=6.0.0",
        "pylint>=3.0.0",
        "pytest>=8.0.0",
        "pytest-cov>=4.1.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "isort>=5.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "devorika=devorika.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
