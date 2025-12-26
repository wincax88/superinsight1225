"""
LLM Adapter for Text-to-SQL.

Provides unified interface to various AI models for SQL generation,
reusing existing AI model integrations.
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from src.ai.base import ModelType, ModelConfig
from src.ai.integration_service import get_integration_service, AIModelIntegrationService

logger = logging.getLogger(__name__)


# SQL Generation Prompt Templates
SQL_GENERATION_PROMPT = """You are an expert SQL query generator. Convert the following natural language query into SQL.

{schema_context}

User Query: {user_query}

Requirements:
1. Generate only valid {dialect} SQL syntax
2. Use appropriate JOINs when querying multiple tables
3. Include necessary WHERE conditions
4. Use aliases for readability when joining tables
5. Add LIMIT clause if result set could be large
6. Do not include any explanations, only the SQL query

SQL Query:"""

SQL_GENERATION_WITH_EXPLANATION_PROMPT = """You are an expert SQL query generator. Convert the following natural language query into SQL.

{schema_context}

User Query: {user_query}

Requirements:
1. Generate only valid {dialect} SQL syntax
2. Use appropriate JOINs when querying multiple tables
3. Include necessary WHERE conditions
4. Use aliases for readability when joining tables
5. Add LIMIT clause if result set could be large

Please provide:
1. The SQL query
2. A brief explanation of what the query does

Format your response as:
```sql
<your SQL query here>
```

Explanation: <brief explanation>"""

COMPLEX_SQL_PROMPT = """You are an expert SQL query generator specializing in complex analytical queries.

{schema_context}

User Query: {user_query}

This query may require:
- Subqueries
- Common Table Expressions (CTEs)
- Window functions
- Complex aggregations
- Multiple JOINs

Requirements:
1. Generate valid {dialect} SQL syntax
2. Use CTEs for better readability when appropriate
3. Use window functions for ranking/running totals if needed
4. Optimize for performance where possible

SQL Query:"""


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    @abstractmethod
    async def generate_sql(self, prompt: str) -> Dict[str, Any]:
        """Generate SQL from prompt."""
        pass

    @abstractmethod
    async def check_availability(self) -> bool:
        """Check if the model is available."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class LLMAdapter(BaseLLMAdapter):
    """
    Unified LLM adapter for SQL generation.

    Wraps existing AI model integrations for Text-to-SQL use cases.
    """

    def __init__(self,
                 model_type: Optional[ModelType] = None,
                 model_name: Optional[str] = None,
                 integration_service: Optional[AIModelIntegrationService] = None):
        """
        Initialize LLM adapter.

        Args:
            model_type: Preferred model type (optional)
            model_name: Preferred model name (optional)
            integration_service: AI integration service instance
        """
        self.model_type = model_type
        self.model_name = model_name
        self._service = integration_service or get_integration_service()
        self._current_model: Optional[str] = None

        # Performance tracking
        self._request_count = 0
        self._total_time = 0.0
        self._error_count = 0

    async def generate_sql(self, prompt: str) -> Dict[str, Any]:
        """
        Generate SQL using the AI model.

        Args:
            prompt: The formatted prompt for SQL generation

        Returns:
            Dictionary with generated SQL and metadata
        """
        start_time = time.time()

        try:
            # Create a mock task for the AI service
            from src.models.task import Task
            from uuid import uuid4

            task = Task(
                id=uuid4(),
                project_id=uuid4(),
                data={"prompt": prompt},
                task_type="text_generation"
            )

            # Use best available model
            prediction = await self._service.predict_with_best_model(
                task,
                task_type="text_generation",
                requirements={"capability": "sql_generation"}
            )

            processing_time = time.time() - start_time
            self._request_count += 1
            self._total_time += processing_time

            # Extract SQL from response
            response_text = prediction.prediction_data.get("text", "")
            sql, explanation = self._extract_sql_and_explanation(response_text)

            return {
                "success": True,
                "sql": sql,
                "explanation": explanation,
                "raw_response": response_text,
                "confidence": prediction.confidence,
                "processing_time": processing_time,
                "model_used": str(prediction.ai_model_config.model_name)
            }

        except Exception as e:
            self._error_count += 1
            logger.error(f"SQL generation failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def generate_sql_simple(self,
                                  user_query: str,
                                  schema_context: str,
                                  dialect: str = "postgresql") -> Dict[str, Any]:
        """
        Generate SQL with simple prompt.

        Args:
            user_query: Natural language query
            schema_context: Database schema context
            dialect: SQL dialect

        Returns:
            Generation result
        """
        prompt = SQL_GENERATION_PROMPT.format(
            schema_context=schema_context,
            user_query=user_query,
            dialect=dialect
        )
        return await self.generate_sql(prompt)

    async def generate_sql_with_explanation(self,
                                           user_query: str,
                                           schema_context: str,
                                           dialect: str = "postgresql") -> Dict[str, Any]:
        """
        Generate SQL with explanation.

        Args:
            user_query: Natural language query
            schema_context: Database schema context
            dialect: SQL dialect

        Returns:
            Generation result with explanation
        """
        prompt = SQL_GENERATION_WITH_EXPLANATION_PROMPT.format(
            schema_context=schema_context,
            user_query=user_query,
            dialect=dialect
        )
        return await self.generate_sql(prompt)

    async def generate_complex_sql(self,
                                   user_query: str,
                                   schema_context: str,
                                   dialect: str = "postgresql") -> Dict[str, Any]:
        """
        Generate complex SQL with advanced features.

        Args:
            user_query: Natural language query
            schema_context: Database schema context
            dialect: SQL dialect

        Returns:
            Generation result
        """
        prompt = COMPLEX_SQL_PROMPT.format(
            schema_context=schema_context,
            user_query=user_query,
            dialect=dialect
        )
        return await self.generate_sql(prompt)

    def _extract_sql_and_explanation(self, response: str) -> tuple:
        """
        Extract SQL and explanation from model response.

        Args:
            response: Raw model response

        Returns:
            Tuple of (sql, explanation)
        """
        sql = ""
        explanation = ""

        # Try to extract SQL from code blocks
        sql_pattern = r"```sql\s*(.*?)\s*```"
        sql_matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)

        if sql_matches:
            sql = sql_matches[0].strip()
            # Extract explanation after the code block
            explanation_pattern = r"```\s*(?:Explanation|解释|说明)[:：]?\s*(.*)"
            exp_matches = re.findall(explanation_pattern, response, re.DOTALL | re.IGNORECASE)
            if exp_matches:
                explanation = exp_matches[0].strip()
        else:
            # Try generic code block
            code_pattern = r"```\s*(.*?)\s*```"
            code_matches = re.findall(code_pattern, response, re.DOTALL)

            if code_matches:
                sql = code_matches[0].strip()
            else:
                # Assume entire response is SQL
                # Look for SELECT, INSERT, UPDATE, DELETE patterns
                sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE"]
                for keyword in sql_keywords:
                    if keyword in response.upper():
                        sql = response.strip()
                        break

        # Clean up SQL
        sql = self._clean_sql(sql)

        return sql, explanation

    def _clean_sql(self, sql: str) -> str:
        """Clean and format SQL query."""
        if not sql:
            return ""

        # Remove leading/trailing whitespace and newlines
        sql = sql.strip()

        # Remove markdown code block markers if present
        if sql.startswith("```"):
            sql = re.sub(r"^```\w*\n?", "", sql)
        if sql.endswith("```"):
            sql = re.sub(r"\n?```$", "", sql)

        # Ensure SQL ends with semicolon
        if sql and not sql.endswith(";"):
            sql += ";"

        return sql

    async def check_availability(self) -> bool:
        """Check if the AI service is available."""
        try:
            status = await self._service.get_integration_status()
            return status.get("status") == "operational"
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_type": str(self.model_type) if self.model_type else "auto",
            "model_name": self.model_name,
            "current_model": self._current_model,
            "statistics": {
                "request_count": self._request_count,
                "total_time": self._total_time,
                "average_time": self._total_time / max(1, self._request_count),
                "error_count": self._error_count,
                "error_rate": self._error_count / max(1, self._request_count)
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "request_count": self._request_count,
            "total_time": self._total_time,
            "average_time": self._total_time / max(1, self._request_count),
            "error_count": self._error_count,
            "success_rate": 1 - (self._error_count / max(1, self._request_count))
        }

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self._request_count = 0
        self._total_time = 0.0
        self._error_count = 0


class MockLLMAdapter(BaseLLMAdapter):
    """Mock LLM adapter for testing."""

    def __init__(self):
        self._responses: Dict[str, str] = {}

    def add_response(self, query_pattern: str, sql: str) -> None:
        """Add a mock response."""
        self._responses[query_pattern.lower()] = sql

    async def generate_sql(self, prompt: str) -> Dict[str, Any]:
        """Generate mock SQL response."""
        # Find matching pattern
        for pattern, sql in self._responses.items():
            if pattern in prompt.lower():
                return {
                    "success": True,
                    "sql": sql,
                    "explanation": "Mock response",
                    "confidence": 0.95,
                    "processing_time": 0.1,
                    "model_used": "mock"
                }

        # Default response
        return {
            "success": True,
            "sql": "SELECT * FROM table LIMIT 10;",
            "explanation": "Default mock response",
            "confidence": 0.5,
            "processing_time": 0.05,
            "model_used": "mock"
        }

    async def check_availability(self) -> bool:
        """Always available."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model info."""
        return {
            "model_type": "mock",
            "model_name": "mock_llm",
            "response_count": len(self._responses)
        }


# Global instance
_llm_adapter: Optional[LLMAdapter] = None


def get_llm_adapter(model_type: Optional[ModelType] = None,
                   model_name: Optional[str] = None) -> LLMAdapter:
    """Get or create global LLM adapter instance."""
    global _llm_adapter

    if _llm_adapter is None or model_type or model_name:
        _llm_adapter = LLMAdapter(model_type, model_name)

    return _llm_adapter
