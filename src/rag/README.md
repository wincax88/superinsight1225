# RAG (Retrieval-Augmented Generation) Module

## Overview

The RAG module provides retrieval-augmented generation functionality for AI applications, enabling semantic search and document retrieval capabilities.

## Features

- **Document Search**: Semantic search across document collections
- **Text Chunking**: Intelligent text splitting with overlap
- **Similarity Scoring**: Relevance scoring for search results
- **Caching**: Performance optimization with result caching
- **Metrics Tracking**: Performance monitoring and analytics

## API Endpoints

### RAG Search
```
POST /api/v1/rag/search
```

### Get Metrics
```
GET /api/v1/rag/metrics
```

### Clear Cache
```
POST /api/v1/rag/clear-cache
```

### Reset Metrics
```
POST /api/v1/rag/reset-metrics
```

### Get Document Chunks
```
GET /api/v1/rag/document/{document_id}/chunks
```

## Usage Examples

### Basic RAG Search
```python
request = RAGRequest(
    query="What is machine learning?",
    top_k=5,
    similarity_threshold=0.7,
    chunk_size=512
)
```

### Project-Specific Search
```python
request = RAGRequest(
    query="customer feedback analysis",
    project_id="feedback_project",
    top_k=10,
    include_metadata=True
)
```

## Requirements

- Requirements 6.4: RAG testing interface
- Requirements 6.5: Agent testing interface