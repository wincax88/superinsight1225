# Agent Testing Module

## Overview

The Agent module provides AI agent testing functionality, supporting various task types including classification, extraction, summarization, and more.

## Features

- **Multiple Task Types**: Classification, extraction, summarization, Q&A, text generation, analysis
- **Step-by-Step Execution**: Detailed execution tracking with confidence scores
- **Timeout Handling**: Configurable execution timeouts
- **Metrics Tracking**: Performance monitoring and success rate tracking
- **Error Recovery**: Robust error handling and reporting

## Supported Task Types

### Classification
- Text classification into predefined categories
- Confidence scoring for predictions
- Multi-class and multi-label support

### Extraction
- Named entity recognition
- Information extraction from text
- Structured data extraction

### Summarization
- Text summarization with length control
- Key point extraction
- Abstract generation

### Question Answering
- Context-based question answering
- Factual information retrieval
- Multi-step reasoning

### Text Generation
- Prompt-based text generation
- Creative writing assistance
- Content completion

### Analysis
- Data analysis and insights
- Pattern recognition
- Recommendation generation

## API Endpoints

### Execute Agent Task
```
POST /api/v1/agent/execute
```

### Get Metrics
```
GET /api/v1/agent/metrics
```

### Reset Metrics
```
POST /api/v1/agent/reset-metrics
```

### Get Supported Tasks
```
GET /api/v1/agent/tasks
```

### Combined RAG-Agent Pipeline
```
POST /api/v1/test/rag-agent-pipeline
```

### Health Check
```
GET /api/v1/test/health
```

### Performance Test
```
GET /api/v1/test/performance
```

## Usage Examples

### Text Classification
```python
request = AgentRequest(
    task_type="classification",
    input_data={
        "text": "This product is amazing!",
        "categories": ["positive", "negative", "neutral"]
    },
    max_iterations=5
)
```

### Entity Extraction
```python
request = AgentRequest(
    task_type="extraction",
    input_data={
        "text": "John Smith works at Microsoft in Seattle.",
        "entities": ["PERSON", "ORG", "LOC"]
    }
)
```

### Question Answering
```python
request = AgentRequest(
    task_type="question_answering",
    input_data={
        "question": "What is the capital of France?",
        "context": "France is a country in Europe. Paris is its capital city."
    }
)
```

## Requirements

- Requirements 6.4: RAG testing interface
- Requirements 6.5: Agent testing interface