"""
RAG service for SuperInsight Platform.

Provides RAG (Retrieval-Augmented Generation) functionality for AI applications.
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select

from .models import RAGRequest, RAGResponse, DocumentChunk, RAGMetrics
from src.database.models import DocumentModel, TaskModel
from src.database.connection import db_manager

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG operations and document retrieval."""
    
    def __init__(self):
        """Initialize RAG service."""
        # Simple in-memory cache for embeddings and results
        self.embedding_cache: Dict[str, List[float]] = {}
        self.query_cache: Dict[str, RAGResponse] = {}
        self.cache_ttl = timedelta(hours=1)  # Cache TTL
        
        # Metrics tracking
        self.metrics = {
            "query_count": 0,
            "total_response_time": 0.0,
            "total_chunks_returned": 0,
            "total_similarity_score": 0.0,
            "cache_hits": 0
        }
    
    def search_documents(self, request: RAGRequest) -> RAGResponse:
        """Search documents using RAG approach."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                self.metrics["cache_hits"] += 1
                logger.info(f"RAG query cache hit for: {request.query[:50]}...")
                return cached_response
            
            # Query documents from database
            with db_manager.get_session() as db:
                documents = self._query_documents(db, request)
                
                # Process documents into chunks
                chunks = self._process_documents_to_chunks(documents, request)
                
                # Perform similarity search (simplified implementation)
                relevant_chunks = self._similarity_search(chunks, request)
                
                # Create response
                processing_time = time.time() - start_time
                response = RAGResponse(
                    query=request.query,
                    chunks=relevant_chunks,
                    total_chunks=len(relevant_chunks),
                    processing_time=processing_time,
                    metadata={
                        "total_documents": len(documents),
                        "chunk_size": request.chunk_size,
                        "similarity_threshold": request.similarity_threshold
                    }
                )
                
                # Cache response
                self._cache_response(cache_key, response)
                
                # Update metrics
                self._update_metrics(response)
                
                logger.info(f"RAG search completed: {len(relevant_chunks)} chunks in {processing_time:.2f}s")
                return response
                
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            # Return empty response on error
            return RAGResponse(
                query=request.query,
                chunks=[],
                total_chunks=0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _query_documents(self, db: Session, request: RAGRequest) -> List[DocumentModel]:
        """Query documents from database based on request filters."""
        stmt = select(DocumentModel)
        
        # Apply filters
        filters = []
        
        if request.document_ids:
            filters.append(DocumentModel.id.in_(request.document_ids))
        
        if request.project_id:
            # Join with tasks table for project filtering
            stmt = stmt.join(TaskModel)
            filters.append(TaskModel.project_id == request.project_id)
        
        if filters:
            stmt = stmt.where(and_(*filters))
        
        # Limit results for performance
        stmt = stmt.limit(1000)
        documents = db.execute(stmt).scalars().all()
        
        return documents
    
    def _process_documents_to_chunks(self, documents: List[DocumentModel], 
                                   request: RAGRequest) -> List[DocumentChunk]:
        """Process documents into chunks for RAG."""
        chunks = []
        
        for doc in documents:
            # Split document content into chunks
            doc_chunks = self._split_text_into_chunks(
                doc.content, 
                request.chunk_size, 
                request.chunk_overlap
            )
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk = DocumentChunk(
                    id=f"{doc.id}_{i}",
                    document_id=str(doc.id),
                    content=chunk_text,
                    metadata={
                        "source_type": doc.source_type,
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                        "document_created_at": doc.created_at.isoformat()
                    }
                )
                
                if request.include_metadata:
                    chunk.metadata.update(doc.document_metadata or {})
                
                chunks.append(chunk)
        
        return chunks
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space before end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _similarity_search(self, chunks: List[DocumentChunk], 
                          request: RAGRequest) -> List[DocumentChunk]:
        """Perform similarity search on chunks (simplified implementation)."""
        # Simplified similarity calculation using keyword matching
        # In production, use proper vector embeddings and similarity search
        
        query_words = set(request.query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            
            # Calculate simple Jaccard similarity
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0
            
            # Apply similarity threshold
            if similarity >= request.similarity_threshold:
                chunk.similarity_score = similarity
                scored_chunks.append(chunk)
        
        # Sort by similarity score (descending)
        scored_chunks.sort(key=lambda x: x.similarity_score or 0, reverse=True)
        
        # Return top-k results
        return scored_chunks[:request.top_k]
    
    def _generate_cache_key(self, request: RAGRequest) -> str:
        """Generate cache key for request."""
        # Create hash from request parameters
        key_data = {
            "query": request.query,
            "project_id": request.project_id,
            "document_ids": sorted(request.document_ids) if request.document_ids else None,
            "top_k": request.top_k,
            "similarity_threshold": request.similarity_threshold,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap
        }
        
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[RAGResponse]:
        """Get cached response if available and not expired."""
        if cache_key in self.query_cache:
            response = self.query_cache[cache_key]
            # Check if cache is still valid (simplified - no actual TTL check)
            return response
        return None
    
    def _cache_response(self, cache_key: str, response: RAGResponse) -> None:
        """Cache response."""
        # Simple cache without TTL implementation
        # In production, use Redis with proper TTL
        if len(self.query_cache) > 1000:  # Simple cache size limit
            # Remove oldest entries
            keys_to_remove = list(self.query_cache.keys())[:100]
            for key in keys_to_remove:
                del self.query_cache[key]
        
        self.query_cache[cache_key] = response
    
    def _update_metrics(self, response: RAGResponse) -> None:
        """Update service metrics."""
        self.metrics["query_count"] += 1
        self.metrics["total_response_time"] += response.processing_time
        self.metrics["total_chunks_returned"] += response.total_chunks
        
        if response.chunks:
            avg_similarity = sum(chunk.similarity_score or 0 for chunk in response.chunks) / len(response.chunks)
            self.metrics["total_similarity_score"] += avg_similarity
    
    def get_metrics(self) -> RAGMetrics:
        """Get RAG service metrics."""
        query_count = self.metrics["query_count"]
        
        if query_count == 0:
            return RAGMetrics(
                query_count=0,
                avg_response_time=0.0,
                avg_chunks_returned=0.0,
                avg_similarity_score=0.0,
                cache_hit_rate=0.0
            )
        
        return RAGMetrics(
            query_count=query_count,
            avg_response_time=self.metrics["total_response_time"] / query_count,
            avg_chunks_returned=self.metrics["total_chunks_returned"] / query_count,
            avg_similarity_score=self.metrics["total_similarity_score"] / query_count,
            cache_hit_rate=(self.metrics["cache_hits"] / query_count) * 100
        )
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.query_cache.clear()
        logger.info("RAG service caches cleared")
    
    def reset_metrics(self) -> None:
        """Reset service metrics."""
        self.metrics = {
            "query_count": 0,
            "total_response_time": 0.0,
            "total_chunks_returned": 0,
            "total_similarity_score": 0.0,
            "cache_hits": 0
        }
        logger.info("RAG service metrics reset")
    
    def get_document_chunks(self, document_id: str, chunk_size: int = 512, 
                           chunk_overlap: int = 50) -> List[DocumentChunk]:
        """Get chunks for a specific document."""
        try:
            with db_manager.get_session() as db:
                stmt = select(DocumentModel).where(
                    DocumentModel.id == document_id
                )
                document = db.execute(stmt).scalar_one_or_none()
                
                if not document:
                    return []
                
                # Create chunks
                chunk_texts = self._split_text_into_chunks(
                    document.content, chunk_size, chunk_overlap
                )
                
                chunks = []
                for i, chunk_text in enumerate(chunk_texts):
                    chunk = DocumentChunk(
                        id=f"{document_id}_{i}",
                        document_id=document_id,
                        content=chunk_text,
                        metadata={
                            "source_type": document.source_type,
                            "chunk_index": i,
                            "total_chunks": len(chunk_texts),
                            "document_created_at": document.created_at.isoformat()
                        }
                    )
                    chunks.append(chunk)
                
                return chunks
                
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []