# Quality Measures for Dynamic Graph Generative Models

This repository contains the source code for the JL-Metric, which provides a principled approach to evaluating generative models for continuous-time dynamic graphs (CTDGs).

## Quick Start

To evaluate the quality of your generated dynamic graphs, follow the instructions below:

### Installation

```bash
pip install jl-metric
```

### Basic Usage

Our evaluator accepts graphs in the form of PyTorch Geometric [TemporalData](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.TemporalData.html#torch_geometric.data.TemporalData) objects. These should contain temporal interactions with source nodes (`src`), destination nodes (`dst`), timestamps (`t`), and optionally messages/features (`msg`).

```python
from jl_metric import JLEvaluator

# Initialize the evaluator
evaluator = JLEvaluator()

# Prepare your graphs as PyG TemporalData objects
# reference_graph = your ground truth or reference graph
# generated_graph = your model's generated graph

# Create input dictionary
input_dict = {
    'reference': reference_graph,
    'generated': generated_graph
}

# Evaluate and get results
result_dict = evaluator.eval(input_dict)
print(f"JL-Metric score: {result_dict['JL-Metric']}")
```

### Advanced Configuration

The evaluator accepts several optional parameters:

```python
evaluator = JLEvaluator(
    node_dim=100,     # Dimension for node embeddings
    graph_dim=100,    # Dimension for graph embeddings
    seed=42           # Random seed for reproducibility
)
```
