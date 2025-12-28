"""
Query module for Knowledge Graph.

Provides natural language query and Cypher generation.
"""

from .nl_query_engine import (
    QueryIntent,
    QueryEntity,
    QueryRelation,
    QueryParameters,
    ParsedQuery,
    NLQueryEngine,
    get_nl_query_engine,
)

from .cypher_generator import (
    CypherTemplate,
    CypherQuery,
    CypherGenerationResult,
    CypherGenerator,
    get_cypher_generator,
)

from .result_formatter import (
    OutputFormat,
    ResultType,
    FormattedEntity,
    FormattedRelation,
    FormattedPath,
    Recommendation,
    FormattedResult,
    ResultFormatter,
    get_result_formatter,
)

__all__ = [
    # NL Query Engine
    "QueryIntent",
    "QueryEntity",
    "QueryRelation",
    "QueryParameters",
    "ParsedQuery",
    "NLQueryEngine",
    "get_nl_query_engine",
    # Cypher Generator
    "CypherTemplate",
    "CypherQuery",
    "CypherGenerationResult",
    "CypherGenerator",
    "get_cypher_generator",
    # Result Formatter
    "OutputFormat",
    "ResultType",
    "FormattedEntity",
    "FormattedRelation",
    "FormattedPath",
    "Recommendation",
    "FormattedResult",
    "ResultFormatter",
    "get_result_formatter",
]
