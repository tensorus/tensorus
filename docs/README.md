# Tensorus Documentation

Welcome to the Tensorus documentation! This documentation provides comprehensive guides and references for using the Tensorus tensor database and computation framework.

## Interactive API Documentation

For the most up-to-date API reference, use our interactive documentation:

- **Swagger UI**: [/docs](http://localhost:8000/docs) - Interactive API exploration with "Try it out" functionality
- **ReDoc**: [/redoc](http://localhost:8000/redoc) - Clean, responsive API documentation

These tools are automatically generated from the code and always reflect the latest API changes.

## Table of Contents

1. [Getting Started](#getting-started)
2. [User Guide](user_guide.md)
3. [API Reference](api/README.md)
4. [Examples](#examples)
5. [Contributing](#contributing)
6. [License](#license)

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tensorus.git
cd tensorus

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### Quick Example

```python
from tensorus import Tensorus
import numpy as np

# Initialize Tensorus
ts = Tensorus()

# Create a tensor
data = np.random.rand(3, 3)
tensor = ts.create_tensor(data, name="example_tensor")

# Perform operations
result = tensor.transpose()
print(result)
```

## User Guide

For detailed usage instructions, see the [User Guide](user_guide.md).

## API Reference

For complete API documentation, see the [API Reference](api/README.md).

## Examples

Explore our collection of examples to learn how to use Tensorus:

- [Basic Usage](examples/basic_usage.py) - Introduction to core features
- [Advanced Usage](examples/advanced_usage.py) - Complex scenarios and patterns
- [Vector Search](examples/vector_search.py) - Semantic search examples
- [Model Integration](examples/model_integration.py) - Using Tensorus with ML models

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

Tensorus is licensed under the MIT License. See [LICENSE](../LICENSE) for more information.
