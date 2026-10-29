# python-package-template
This is a template on how to package a simple Python project

## Table of Contents

1. Installation
2. Setting Up Your Package
3. Installing Dependencies
4. Building Your Package
5. Publishing to PyPI

## Installation

To install the package in editable mode (ideal for development), follow these steps:

### Requirements

- Python 3.7 or higher
- `pip` (ensure it's the latest version)
- `setuptools` 42 or higher (for building the package)

### 1. Clone the repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-package-name.git
cd your-package-name
```

### 2. Set Up Your Python Environment

Create a virtual environment for your package:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Build Dependencies

Make sure setuptools and pip are up to date:

```bash
pip install --upgrade pip setuptools wheel
```

## Setting Up Your Package
### 1. Update pyproject.toml

The pyproject.toml file contains the configuration for building and packaging your Python project. You'll want to customize this to reflect your package's name, version, dependencies, license, etc.
```yml
    name: The name of your package.
    version: The version of your package (e.g., "0.1.0").
    dependencies: List any runtime dependencies your package requires (e.g., requests, numpy).
    license: Specify your package's license, either as text or a file. For example:
        license = { text = "MIT" }
        Or, if you have a LICENSE file: license = { file = "LICENSE.txt" }
```

### 2. Update README.md

Edit this README file to reflect your package's functionality.

## Installing Dependencies

To install your package in editable mode for development, use the following command:

```bash
pip install -e .
```

This will install the package, allowing you to edit it directly and have changes take effect immediately without reinstalling.

To install any optional dependencies, such as development dependencies, use:

```bash
pip install -e .[dev]
```

## Building Your Package

To build your package for distribution (e.g., for uploading to PyPI), you can use:

```bash
python -m build
```

This will create .tar.gz and .whl files in the dist/ directory.

## Publishing to PyPI

To publish your package to PyPI, you can use the twine tool:

```bash
pip install twine
twine upload dist/*
```

You'll need to have a PyPI account and have your credentials set up for this.

---
