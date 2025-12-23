# Data Export Module

## Overview

The Export module provides comprehensive data export functionality for the SuperInsight Platform, supporting multiple formats including JSON, CSV, and COCO.

## Features

- **Multiple Export Formats**: JSON, CSV, COCO
- **Batch Processing**: Support for large datasets with batch export
- **Progress Tracking**: Real-time export progress monitoring
- **Error Handling**: Robust error handling and recovery
- **Flexible Filtering**: Filter by project, document, task, and date range

## Supported Formats

### JSON Format
- Full document and annotation data
- Hierarchical structure
- Includes metadata and relationships
- Best for: Complete data backup, data migration

### CSV Format
- Tabular data format
- Flat structure with one row per task
- Easy to import into spreadsheets
- Best for: Data analysis, reporting

### COCO Format
- Computer vision standard format
- Includes images, annotations, and categories
- Compatible with CV frameworks
- Best for: Computer vision applications

## API Endpoints

### Start Export
```
POST /api/v1/export/start
```

### Get Export Status
```
GET /api/v1/export/status/{export_id}
```

### Download Export
```
GET /api/v1/export/download/{export_id}
```

### List Exports
```
GET /api/v1/export/list
```

### Delete Export
```
DELETE /api/v1/export/delete/{export_id}
```

### Batch Export
```
POST /api/v1/export/batch
```

### Preview Export
```
POST /api/v1/export/preview
```

### Get Supported Formats
```
GET /api/v1/export/formats
```

## Usage Examples

### Basic JSON Export
```python
request = ExportRequest(
    format=ExportFormat.JSON,
    project_id="my_project",
    include_annotations=True,
    include_metadata=True
)
```

### CSV Export with Filters
```python
request = ExportRequest(
    format=ExportFormat.CSV,
    document_ids=["doc1", "doc2"],
    date_from=datetime(2024, 1, 1),
    date_to=datetime(2024, 12, 31)
)
```

### COCO Export
```python
request = ExportRequest(
    format=ExportFormat.COCO,
    project_id="vision_project",
    include_annotations=True
)
```

## Requirements

- Requirements 6.1: JSON format export
- Requirements 6.2: CSV format export
- Requirements 6.3: COCO format export