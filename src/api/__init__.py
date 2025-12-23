"""
API endpoints module for SuperInsight Platform.

Provides RESTful API endpoints for data extraction and management.
"""

from .extraction import router as extraction_router

__all__ = ['extraction_router']