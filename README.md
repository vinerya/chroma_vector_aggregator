# Chroma Embeddings Aggregation Library

This Python library provides a suite of advanced methods for aggregating multiple embeddings associated with a single document or entity into a single representative embedding. It is designed to work with Chroma vector stores and is compatible with LangChain's Chroma integration.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Example: Simple Average Aggregation](#example-simple-average-aggregation)
- [Aggregation Methods](#aggregation-methods)
- [Parameters](#parameters)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- Multiple aggregation methods (average, weighted average, geometric mean, harmonic mean, centroid, PCA, etc.)
- Compatible with Chroma vector stores and LangChain
- Easy-to-use API for aggregating embeddings

## Installation

To install the package, you can use pip:

```bash
pip install chroma_vector_aggregator
```

## Usage

Here's an example demonstrating how to use the library to aggregate embeddings using simple averaging:

### Example: Simple Average Aggregation

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain.schema import Document
from chroma_vector_aggregator import aggregate_embeddings

# Create a sample Chroma collection
embeddings = FakeEmbeddings(size=10)
documents = [
    Document(page_content="Test document 1", metadata={"id": "group1"}),
    Document(page_content="Test document 2", metadata={"id": "group1"}),
    Document(page_content="Test document 3", metadata={"id": "group2"}),
    Document(page_content="Test document 4", metadata={"id": "group2"}),
]
chroma_collection = Chroma.from_documents(documents, embeddings)

# Aggregate embeddings using simple averaging
aggregated_collection = aggregate_embeddings(
    chroma_collection=chroma_collection,
    column_name="id",
    method="average"
)

# Use the aggregated collection for similarity search
results = aggregated_collection.similarity_search("Test query", k=2)
```

## Aggregation Methods

- `average`: Compute the arithmetic mean of embeddings.
- `weighted_average`: Compute a weighted average of embeddings.
- `geometric_mean`: Compute the geometric mean across embeddings.
- `harmonic_mean`: Compute the harmonic mean across embeddings.
- `median`: Compute the element-wise median of embeddings.
- `trimmed_mean`: Compute the mean after trimming outliers.
- `centroid`: Use K-Means clustering to find the centroid of the embeddings.
- `pca`: Use Principal Component Analysis to reduce embeddings.
- `exemplar`: Select the embedding that best represents the group.
- `max_pooling`: Take the maximum value for each dimension across embeddings.
- `min_pooling`: Take the minimum value for each dimension across embeddings.
- `entropy_weighted_average`: Weight embeddings by their entropy.
- `attentive_pooling`: Use an attention mechanism to aggregate embeddings.
- `tukeys_biweight`: A robust method to down-weight outliers.

## Parameters

- `chroma_collection`: The Chroma collection to aggregate embeddings from.
- `column_name`: The metadata field by which to aggregate embeddings (e.g., 'id').
- `method`: The aggregation method to use.
- `weights` (optional): Weights for the `weighted_average` method.
- `trim_percentage` (optional): Fraction to trim from each end for `trimmed_mean`.

## Dependencies

- chromadb
- numpy
- scipy
- scikit-learn
- langchain

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.