# Knowledge Graph API Reference

## Overview

SuperInsight Knowledge Graph API provides comprehensive endpoints for managing entities, relations, and performing intelligent graph queries. This document describes all available endpoints, request/response formats, and usage examples.

**Base URL**: `/api/v1/knowledge-graph`

**Authentication**: Bearer Token (JWT) required for all endpoints

## Table of Contents

1. [Health & Monitoring](#health--monitoring)
2. [Entity Management](#entity-management)
3. [Relation Management](#relation-management)
4. [NLP Extraction](#nlp-extraction)
5. [Graph Queries](#graph-queries)
6. [Statistics](#statistics)
7. [Error Handling](#error-handling)

---

## Health & Monitoring

### GET /health

Check the health status of the Knowledge Graph service.

**Response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

**Status Codes**:
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Database connection issues

---

## Entity Management

### POST /entities

Create a new entity in the knowledge graph.

**Request Body**:
```json
{
  "name": "Acme Corporation",
  "type": "ORGANIZATION",
  "properties": {
    "industry": "Technology",
    "founded": "2010",
    "employees": 500
  },
  "metadata": {
    "source": "manual",
    "confidence": 1.0
  }
}
```

**Response** (201 Created):
```json
{
  "id": "entity_abc123",
  "name": "Acme Corporation",
  "type": "ORGANIZATION",
  "properties": {
    "industry": "Technology",
    "founded": "2010",
    "employees": 500
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### GET /entities/{entity_id}

Retrieve an entity by its ID.

**Parameters**:
- `entity_id` (path, required): The unique entity identifier

**Response** (200 OK):
```json
{
  "id": "entity_abc123",
  "name": "Acme Corporation",
  "type": "ORGANIZATION",
  "properties": {...},
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### PUT /entities/{entity_id}

Update an existing entity.

**Request Body**:
```json
{
  "name": "Acme Corp (Updated)",
  "properties": {
    "employees": 600
  }
}
```

### DELETE /entities/{entity_id}

Delete an entity and its associated relations.

**Response** (204 No Content)

### GET /entities

Search and list entities with filtering.

**Query Parameters**:
- `type` (optional): Filter by entity type (e.g., ORGANIZATION, PERSON)
- `name` (optional): Search by name (partial match)
- `limit` (optional, default: 100): Maximum results
- `offset` (optional, default: 0): Pagination offset

**Response**:
```json
{
  "entities": [...],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

### POST /bulk/entities

Bulk create multiple entities.

**Request Body**:
```json
{
  "entities": [
    {"name": "Entity 1", "type": "PERSON"},
    {"name": "Entity 2", "type": "ORGANIZATION"}
  ]
}
```

**Response**:
```json
{
  "created": 2,
  "failed": 0,
  "entities": [...]
}
```

---

## Relation Management

### POST /relations

Create a relation between two entities.

**Request Body**:
```json
{
  "source_id": "entity_abc123",
  "target_id": "entity_def456",
  "type": "WORKS_FOR",
  "properties": {
    "since": "2020-01-01",
    "role": "Engineer"
  }
}
```

**Response** (201 Created):
```json
{
  "id": "rel_xyz789",
  "source_id": "entity_abc123",
  "target_id": "entity_def456",
  "type": "WORKS_FOR",
  "properties": {...},
  "created_at": "2024-01-01T00:00:00Z"
}
```

### GET /relations/{relation_id}

Retrieve a relation by its ID.

### GET /entities/{entity_id}/relations

Get all relations for an entity.

**Query Parameters**:
- `direction` (optional): `in`, `out`, or `both` (default: both)
- `type` (optional): Filter by relation type
- `limit` (optional, default: 100): Maximum results

**Response**:
```json
{
  "entity_id": "entity_abc123",
  "relations": [
    {
      "id": "rel_xyz789",
      "type": "WORKS_FOR",
      "direction": "out",
      "connected_entity": {...}
    }
  ],
  "total": 5
}
```

### DELETE /relations/{relation_id}

Delete a relation.

### POST /bulk/relations

Bulk create multiple relations.

---

## NLP Extraction

### POST /extract

Extract entities and relations from text using NLP.

**Request Body**:
```json
{
  "text": "John Smith works at Acme Corporation as a Senior Engineer. He reports to Jane Doe.",
  "options": {
    "extract_entities": true,
    "extract_relations": true,
    "language": "en",
    "confidence_threshold": 0.7
  }
}
```

**Response**:
```json
{
  "entities": [
    {
      "name": "John Smith",
      "type": "PERSON",
      "confidence": 0.95,
      "span": [0, 10]
    },
    {
      "name": "Acme Corporation",
      "type": "ORGANIZATION",
      "confidence": 0.92,
      "span": [20, 36]
    }
  ],
  "relations": [
    {
      "source": "John Smith",
      "target": "Acme Corporation",
      "type": "WORKS_FOR",
      "confidence": 0.88
    }
  ],
  "processing_time_ms": 125
}
```

### POST /extract/entities

Extract entities only from text.

---

## Graph Queries

### GET /neighbors/{entity_id}

Get neighboring entities (connected nodes).

**Query Parameters**:
- `depth` (optional, default: 1): Traversal depth (1-5)
- `direction` (optional): `in`, `out`, or `both`
- `relation_types` (optional): Comma-separated relation types
- `limit` (optional, default: 50): Maximum results

**Response**:
```json
{
  "entity_id": "entity_abc123",
  "neighbors": [
    {
      "entity": {...},
      "relation": {...},
      "distance": 1
    }
  ],
  "total": 10
}
```

### GET /path

Find the shortest path between two entities.

**Query Parameters**:
- `source_id` (required): Starting entity ID
- `target_id` (required): Target entity ID
- `max_depth` (optional, default: 5): Maximum path length
- `relation_types` (optional): Allowed relation types

**Response**:
```json
{
  "found": true,
  "path": [
    {"entity": {...}},
    {"relation": {...}},
    {"entity": {...}}
  ],
  "length": 2
}
```

### POST /query/cypher

Execute a read-only Cypher query.

**Request Body**:
```json
{
  "query": "MATCH (p:Person)-[:WORKS_FOR]->(o:Organization) RETURN p, o LIMIT 10",
  "parameters": {}
}
```

**Response**:
```json
{
  "results": [...],
  "columns": ["p", "o"],
  "query_time_ms": 45
}
```

**Security Note**: Only read queries (MATCH, RETURN) are allowed. Write queries will be rejected.

---

## Statistics

### GET /statistics

Get knowledge graph statistics.

**Response**:
```json
{
  "entities": {
    "total": 15000,
    "by_type": {
      "PERSON": 5000,
      "ORGANIZATION": 3000,
      "LOCATION": 2000,
      "OTHER": 5000
    }
  },
  "relations": {
    "total": 45000,
    "by_type": {
      "WORKS_FOR": 10000,
      "LOCATED_IN": 8000,
      "RELATED_TO": 27000
    }
  },
  "graph_density": 0.0004,
  "average_degree": 6.0,
  "last_updated": "2024-01-01T00:00:00Z"
}
```

---

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "ENTITY_NOT_FOUND",
    "message": "Entity with ID 'entity_xyz' not found",
    "details": {
      "entity_id": "entity_xyz"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_abc123"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `ENTITY_NOT_FOUND` | 404 | Entity does not exist |
| `RELATION_NOT_FOUND` | 404 | Relation does not exist |
| `INVALID_ENTITY_TYPE` | 400 | Invalid entity type specified |
| `INVALID_RELATION_TYPE` | 400 | Invalid relation type specified |
| `DUPLICATE_ENTITY` | 409 | Entity with same name/type exists |
| `INVALID_CYPHER_QUERY` | 400 | Cypher query syntax error |
| `WRITE_QUERY_REJECTED` | 403 | Write queries not allowed |
| `DATABASE_ERROR` | 500 | Neo4j connection/query error |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

---

## Entity Types

Supported entity types:

- `PERSON` - Individual person
- `ORGANIZATION` - Company, institution, group
- `LOCATION` - Geographic location, address
- `EVENT` - Event or occurrence
- `PRODUCT` - Product or service
- `CONCEPT` - Abstract concept or idea
- `DOCUMENT` - Document or file
- `DATE` - Date or time period
- `QUANTITY` - Numeric quantity with unit
- `TECHNOLOGY` - Technology, tool, system
- `PROCESS` - Business process or workflow
- `TASK` - Task or activity
- `PROJECT` - Project or initiative
- `SKILL` - Skill or competency
- `DOMAIN` - Domain or field of knowledge
- `OTHER` - Unclassified entity

---

## Relation Types

Supported relation types:

- `WORKS_FOR` - Employment relationship
- `LOCATED_IN` - Geographic location
- `PART_OF` - Part-whole relationship
- `RELATED_TO` - General relationship
- `CREATED_BY` - Creator relationship
- `OWNS` - Ownership
- `MANAGES` - Management
- `REPORTS_TO` - Reporting structure
- `COLLABORATES_WITH` - Collaboration
- `DEPENDS_ON` - Dependency
- `PRECEDES` - Temporal precedence
- `FOLLOWS` - Temporal following
- `SIMILAR_TO` - Similarity
- `OPPOSITE_OF` - Opposition
- `DERIVED_FROM` - Derivation
- `INSTANCE_OF` - Instance relationship
- `SUBCLASS_OF` - Subclass relationship
- `HAS_PROPERTY` - Property association
- `MENTIONS` - Reference/mention
- `AUTHORED_BY` - Authorship
- `PARTICIPATED_IN` - Event participation
- `USES` - Usage relationship
- `PRODUCES` - Production relationship
- `CONSUMES` - Consumption relationship

---

## Rate Limiting

- **Default**: 1000 requests per minute per API key
- **Bulk Operations**: 100 requests per minute
- **Cypher Queries**: 50 requests per minute

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1704067200
```

---

## Versioning

The API uses URL-based versioning. Current version: `v1`

Future versions will be available at `/api/v2/knowledge-graph` etc.

---

## SDK Examples

### Python

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8000/api/v1/knowledge-graph",
    headers={"Authorization": "Bearer <token>"}
)

# Create entity
entity = client.post("/entities", json={
    "name": "John Doe",
    "type": "PERSON"
}).json()

# Extract from text
result = client.post("/extract", json={
    "text": "John works at Acme Corp"
}).json()
```

### JavaScript

```javascript
const response = await fetch('/api/v1/knowledge-graph/entities', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer <token>'
  },
  body: JSON.stringify({
    name: 'John Doe',
    type: 'PERSON'
  })
});

const entity = await response.json();
```

---

## Changelog

### v1.0.0 (2024-01-01)
- Initial release with full CRUD operations
- NLP extraction endpoints
- Graph query capabilities
- Cypher query support
