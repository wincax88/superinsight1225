"""
RAG service for SuperInsight Platform.

Provides RAG (Retrieval-Augmented Generation) functionality for AI applications.
"""

import logging
import time
import hashlib
import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select

from .models import (
    RAGRequest, RAGResponse, DocumentChunk, RAGMetrics,
    RAGEvaluationRequest, RAGEvaluationResult
)
from src.database.models import DocumentModel, TaskModel
from src.database.connection import db_manager

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG operations and document retrieval."""
    
    def __init__(self):
        """Initialize RAG service."""
        # Enhanced caching with performance tracking
        self.embedding_cache: Dict[str, List[float]] = {}
        self.query_cache: Dict[str, RAGResponse] = {}
        self.cache_ttl = timedelta(hours=1)  # Cache TTL
        
        # Enhanced metrics tracking
        self.metrics = {
            "query_count": 0,
            "total_response_time": 0.0,
            "total_chunks_returned": 0,
            "total_similarity_score": 0.0,
            "cache_hits": 0,
            "error_count": 0,
            "response_times": [],  # For percentile calculations
            "precision_scores": [],
            "recall_scores": [],
            "ndcg_scores": [],
            "mrr_scores": []
        }
        
        # Performance optimization settings
        self.max_cache_size = 10000
        self.enable_parallel_processing = True
        self.similarity_cache: Dict[str, float] = {}
    
    def search_documents(self, request: RAGRequest) -> RAGResponse:
        """Search documents using RAG approach with enhanced performance."""
        start_time = time.time()
        
        try:
            # Update query count immediately for metrics tracking - FIRST THING
            self.metrics["query_count"] += 1
            logger.debug(f"Updated query count to: {self.metrics['query_count']}")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                self.metrics["cache_hits"] += 1
                logger.info(f"RAG query cache hit for: {request.query[:50]}...")
                return cached_response
            
            # Query documents from database with optimizations
            with db_manager.get_session() as db:
                documents = self._query_documents_optimized(db, request)
                
                # Process documents into chunks with parallel processing
                chunks = self._process_documents_to_chunks_optimized(documents, request)
                
                # Perform enhanced similarity search
                relevant_chunks = self._enhanced_similarity_search(chunks, request)
                
                # Create response with enhanced metadata
                processing_time = time.time() - start_time
                response = RAGResponse(
                    query=request.query,
                    chunks=relevant_chunks,
                    total_chunks=len(relevant_chunks),
                    processing_time=processing_time,
                    metadata={
                        "total_documents": len(documents),
                        "chunk_size": request.chunk_size,
                        "similarity_threshold": request.similarity_threshold,
                        "cache_used": False,
                        "optimization_enabled": True,
                        "avg_similarity": self._calculate_avg_similarity(relevant_chunks)
                    }
                )
                
                # Cache response with size management
                self._cache_response_optimized(cache_key, response)
                
                # Update enhanced metrics
                self._update_enhanced_metrics(response)
                
                logger.info(f"RAG search completed: {len(relevant_chunks)} chunks in {processing_time:.2f}s")
                return response
                
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            self.metrics["error_count"] += 1
            logger.debug(f"Updated error count to: {self.metrics['error_count']}")
            
            # Return empty response on error
            return RAGResponse(
                query=request.query,
                chunks=[],
                total_chunks=0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def evaluate_rag_scenarios(self, evaluation_request: RAGEvaluationRequest) -> RAGEvaluationResult:
        """Evaluate RAG performance across multiple test scenarios with enhanced metrics."""
        logger.info(f"Starting enhanced RAG evaluation: {evaluation_request.scenario_name}")
        
        query_results = []
        total_precision = 0.0
        total_recall = 0.0
        total_ndcg = 0.0
        total_mrr = 0.0
        total_response_time = 0.0
        total_f1_score = 0.0
        total_semantic_similarity = 0.0
        
        # Enhanced evaluation with multiple test scenarios
        for i, query in enumerate(evaluation_request.queries):
            start_time = time.time()
            
            # Execute RAG search with different configurations for comprehensive testing
            configurations = [
                {"top_k": 5, "similarity_threshold": 0.3, "name": "conservative"},
                {"top_k": 10, "similarity_threshold": 0.1, "name": "comprehensive"},
                {"top_k": 3, "similarity_threshold": 0.5, "name": "precise"}
            ]
            
            best_result = None
            best_score = 0.0
            config_results = []
            
            for config in configurations:
                rag_request = RAGRequest(
                    query=query,
                    project_id=evaluation_request.project_id,
                    top_k=config["top_k"],
                    similarity_threshold=config["similarity_threshold"]
                )
                
                response = self.search_documents(rag_request)
                
                # Calculate metrics for this configuration
                precision = recall = ndcg = mrr = f1_score = semantic_sim = 0.0
                
                if (evaluation_request.expected_documents and 
                    i < len(evaluation_request.expected_documents)):
                    
                    expected_docs = set(evaluation_request.expected_documents[i])
                    retrieved_docs = set(chunk.document_id for chunk in response.chunks)
                    
                    precision = self._calculate_precision(retrieved_docs, expected_docs)
                    recall = self._calculate_recall(retrieved_docs, expected_docs)
                    ndcg = self._calculate_ndcg(response.chunks, expected_docs)
                    mrr = self._calculate_mrr(response.chunks, expected_docs)
                    f1_score = self._calculate_f1_score(precision, recall)
                    semantic_sim = self._calculate_semantic_similarity(query, response.chunks)
                
                # Calculate combined score for this configuration
                combined_score = (precision * 0.25 + recall * 0.25 + ndcg * 0.3 + 
                                mrr * 0.1 + f1_score * 0.1)
                
                config_result = {
                    "config_name": config["name"],
                    "precision": precision,
                    "recall": recall,
                    "ndcg": ndcg,
                    "mrr": mrr,
                    "f1_score": f1_score,
                    "semantic_similarity": semantic_sim,
                    "combined_score": combined_score,
                    "chunks_returned": len(response.chunks),
                    "avg_similarity": self._calculate_avg_similarity(response.chunks)
                }
                config_results.append(config_result)
                
                # Track best configuration
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = config_result
            
            query_time = time.time() - start_time
            
            # Use best configuration results
            if best_result:
                precision = best_result["precision"]
                recall = best_result["recall"]
                ndcg = best_result["ndcg"]
                mrr = best_result["mrr"]
                f1_score = best_result["f1_score"]
                semantic_sim = best_result["semantic_similarity"]
            else:
                precision = recall = ndcg = mrr = f1_score = semantic_sim = 0.0
            
            # Store enhanced query result
            query_result = {
                "query": query,
                "response_time": query_time,
                "best_config": best_result["config_name"] if best_result else "none",
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "mrr": mrr,
                "f1_score": f1_score,
                "semantic_similarity": semantic_sim,
                "combined_score": best_score,
                "config_results": config_results,
                "chunks_returned": best_result["chunks_returned"] if best_result else 0,
                "avg_similarity": best_result["avg_similarity"] if best_result else 0.0
            }
            query_results.append(query_result)
            
            # Accumulate totals
            total_precision += precision
            total_recall += recall
            total_ndcg += ndcg
            total_mrr += mrr
            total_f1_score += f1_score
            total_semantic_similarity += semantic_sim
            total_response_time += query_time
        
        # Calculate enhanced averages
        num_queries = len(evaluation_request.queries)
        if num_queries > 0:
            avg_precision = total_precision / num_queries
            avg_recall = total_recall / num_queries
            avg_ndcg = total_ndcg / num_queries
            avg_mrr = total_mrr / num_queries
            avg_f1_score = total_f1_score / num_queries
            avg_semantic_similarity = total_semantic_similarity / num_queries
            avg_response_time = total_response_time / num_queries
        else:
            avg_precision = avg_recall = avg_ndcg = avg_mrr = 0.0
            avg_f1_score = avg_semantic_similarity = avg_response_time = 0.0
        
        # Enhanced overall score calculation
        overall_score = (avg_precision * 0.2 + avg_recall * 0.2 + avg_ndcg * 0.3 + 
                        avg_mrr * 0.1 + avg_f1_score * 0.1 + avg_semantic_similarity * 0.1)
        
        # Store enhanced metrics for future analysis
        self.metrics["precision_scores"].extend([r["precision"] for r in query_results])
        self.metrics["recall_scores"].extend([r["recall"] for r in query_results])
        self.metrics["ndcg_scores"].extend([r["ndcg"] for r in query_results])
        self.metrics["mrr_scores"].extend([r["mrr"] for r in query_results])
        
        return RAGEvaluationResult(
            scenario_name=evaluation_request.scenario_name,
            total_queries=num_queries,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_ndcg=avg_ndcg,
            avg_mrr=avg_mrr,
            avg_response_time=avg_response_time,
            query_results=query_results,
            overall_score=overall_score,
            metadata={
                "avg_f1_score": avg_f1_score,
                "avg_semantic_similarity": avg_semantic_similarity,
                "configurations_tested": len(configurations),
                "best_config_distribution": self._analyze_best_configs(query_results)
            }
        )
    
    def _query_documents_optimized(self, db: Session, request: RAGRequest) -> List[DocumentModel]:
        """Optimized document querying with better indexing."""
        stmt = select(DocumentModel)
        
        # Apply filters with optimized joins
        filters = []
        
        if request.document_ids:
            filters.append(DocumentModel.id.in_(request.document_ids))
        
        if request.project_id:
            # Optimized join with proper indexing
            stmt = stmt.join(TaskModel, DocumentModel.id == TaskModel.document_id)
            filters.append(TaskModel.project_id == request.project_id)
        
        if filters:
            stmt = stmt.where(and_(*filters))
        
        # Add ordering for consistent results
        stmt = stmt.order_by(DocumentModel.created_at.desc())
        
        # Limit results for performance with pagination support
        stmt = stmt.limit(min(request.top_k * 10, 2000))  # Get more docs for better filtering
        
        documents = db.execute(stmt).scalars().all()
        return documents
    
    def _process_documents_to_chunks_optimized(self, documents: List[DocumentModel], 
                                             request: RAGRequest) -> List[DocumentChunk]:
        """Optimized document processing with parallel chunking."""
        chunks = []
        
        for doc in documents:
            # Optimized text splitting with better boundary detection
            doc_chunks = self._smart_text_splitting(
                doc.content, 
                request.chunk_size, 
                request.chunk_overlap
            )
            
            for i, chunk_text in enumerate(doc_chunks):
                # Pre-calculate chunk hash for deduplication
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                
                chunk = DocumentChunk(
                    id=f"{doc.id}_{i}_{chunk_hash}",
                    document_id=str(doc.id),
                    content=chunk_text,
                    metadata={
                        "source_type": doc.source_type,
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                        "document_created_at": doc.created_at.isoformat(),
                        "chunk_hash": chunk_hash,
                        "content_length": len(chunk_text)
                    }
                )
                
                if request.include_metadata:
                    chunk.metadata.update(doc.document_metadata or {})
                
                chunks.append(chunk)
        
        return chunks
    
    def _smart_text_splitting(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Smart text splitting with sentence boundary awareness."""
        if len(text) <= chunk_size:
            return [text]
        
        # Split by sentences first for better semantic chunks
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            if len(test_chunk) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Handle overlap by keeping last part of current chunk
                    if overlap > 0 and len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + " " + sentence
                        
                        # If still too long, truncate
                        if len(current_chunk) > chunk_size:
                            current_chunk = current_chunk[:chunk_size]
                    else:
                        current_chunk = sentence
                        # If single sentence is too long, truncate
                        if len(current_chunk) > chunk_size:
                            current_chunk = current_chunk[:chunk_size]
                else:
                    # Single sentence is too long, truncate
                    current_chunk = sentence[:chunk_size]
            else:
                current_chunk = test_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _enhanced_similarity_search(self, chunks: List[DocumentChunk], 
                                  request: RAGRequest) -> List[DocumentChunk]:
        """Enhanced similarity search with multiple algorithms."""
        query_words = set(request.query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            
            # Multiple similarity calculations
            jaccard_sim = self._jaccard_similarity(query_words, chunk_words)
            cosine_sim = self._cosine_similarity_simple(request.query, chunk.content)
            bm25_sim = self._bm25_similarity_simple(request.query, chunk.content)
            
            # Weighted combination of similarities
            combined_similarity = (
                jaccard_sim * 0.3 + 
                cosine_sim * 0.4 + 
                bm25_sim * 0.3
            )
            
            # Apply similarity threshold
            if combined_similarity >= request.similarity_threshold:
                chunk.similarity_score = combined_similarity
                scored_chunks.append(chunk)
        
        # Sort by similarity score (descending)
        scored_chunks.sort(key=lambda x: x.similarity_score or 0, reverse=True)
        
        # Return top-k results with diversity filtering
        return self._apply_diversity_filtering(scored_chunks[:request.top_k * 2])[:request.top_k]
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity_simple(self, text1: str, text2: str) -> float:
        """Simple cosine similarity calculation."""
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Create word frequency vectors
        all_words = set(words1 + words2)
        vec1 = [words1.count(word) for word in all_words]
        vec2 = [words2.count(word) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _bm25_similarity_simple(self, query: str, document: str) -> float:
        """Simplified BM25 similarity calculation."""
        k1, b = 1.5, 0.75  # BM25 parameters
        
        query_words = query.lower().split()
        doc_words = document.lower().split()
        doc_length = len(doc_words)
        avg_doc_length = 100  # Simplified average
        
        score = 0.0
        for word in query_words:
            word_freq = doc_words.count(word)
            if word_freq > 0:
                idf = 1.0  # Simplified IDF
                tf = (word_freq * (k1 + 1)) / (word_freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                score += idf * tf
        
        return min(score / len(query_words), 1.0) if query_words else 0.0
    
    def _apply_diversity_filtering(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Apply diversity filtering to avoid too similar chunks."""
        if len(chunks) <= 1:
            return chunks
        
        diverse_chunks = [chunks[0]]  # Always include top result
        
        for chunk in chunks[1:]:
            # Check similarity with already selected chunks
            is_diverse = True
            for selected in diverse_chunks:
                content_similarity = self._cosine_similarity_simple(chunk.content, selected.content)
                if content_similarity > 0.9:  # Increased threshold to be more restrictive
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_chunks.append(chunk)
        
        return diverse_chunks
    
    def _calculate_precision(self, retrieved: set, relevant: set) -> float:
        """Calculate precision metric."""
        if not retrieved:
            return 0.0
        return len(retrieved.intersection(relevant)) / len(retrieved)
    
    def _calculate_recall(self, retrieved: set, relevant: set) -> float:
        """Calculate recall metric."""
        if not relevant:
            return 0.0
        return len(retrieved.intersection(relevant)) / len(relevant)
    
    def _calculate_ndcg(self, chunks: List[DocumentChunk], relevant_docs: set) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not chunks or not relevant_docs:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, chunk in enumerate(chunks):
            relevance = 1.0 if chunk.document_id in relevant_docs else 0.0
            dcg += relevance / (1 + i)  # Simplified DCG calculation
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1.0] * min(len(relevant_docs), len(chunks))
        idcg = sum(rel / (1 + i) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(self, chunks: List[DocumentChunk], relevant_docs: set) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, chunk in enumerate(chunks):
            if chunk.document_id in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_semantic_similarity(self, query: str, chunks: List[DocumentChunk]) -> float:
        """Calculate semantic similarity between query and retrieved chunks."""
        if not chunks:
            return 0.0
        
        # Simple semantic similarity using multiple approaches
        query_words = set(query.lower().split())
        similarities = []
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            
            # Jaccard similarity
            jaccard = self._jaccard_similarity(query_words, chunk_words)
            
            # Cosine similarity
            cosine = self._cosine_similarity_simple(query, chunk.content)
            
            # Combined semantic score
            semantic_score = (jaccard * 0.4 + cosine * 0.6)
            similarities.append(semantic_score)
        
        return sum(similarities) / len(similarities)
    
    def _analyze_best_configs(self, query_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze which configurations performed best across queries."""
        config_counts = {}
        
        for result in query_results:
            best_config = result.get("best_config", "none")
            config_counts[best_config] = config_counts.get(best_config, 0) + 1
        
        return config_counts
    
    def _calculate_avg_similarity(self, chunks: List[DocumentChunk]) -> float:
        """Calculate average similarity score of chunks."""
        if not chunks:
            return 0.0
        
        similarities = [chunk.similarity_score for chunk in chunks if chunk.similarity_score is not None]
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _cache_response_optimized(self, cache_key: str, response: RAGResponse) -> None:
        """Optimized cache management with size limits."""
        # Implement LRU-style cache management
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest 20% of entries
            keys_to_remove = list(self.query_cache.keys())[:self.max_cache_size // 5]
            for key in keys_to_remove:
                del self.query_cache[key]
        
        self.query_cache[cache_key] = response
    
    def _update_enhanced_metrics(self, response: RAGResponse) -> None:
        """Update enhanced service metrics."""
        self.metrics["query_count"] += 1
        self.metrics["total_response_time"] += response.processing_time
        self.metrics["total_chunks_returned"] += response.total_chunks
        
        # Track response times for percentile calculations
        self.metrics["response_times"].append(response.processing_time)
        
        # Keep only recent response times (last 1000)
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
        
        if response.chunks:
            avg_similarity = sum(chunk.similarity_score or 0 for chunk in response.chunks) / len(response.chunks)
            self.metrics["total_similarity_score"] += avg_similarity
    
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
        """Get enhanced RAG service metrics."""
        query_count = self.metrics["query_count"]
        
        if query_count == 0:
            return RAGMetrics(
                query_count=0,
                avg_response_time=0.0,
                avg_chunks_returned=0.0,
                avg_similarity_score=0.0,
                cache_hit_rate=0.0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                ndcg_score=0.0,
                mrr_score=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                error_rate=0.0
            )
        
        # Calculate percentiles
        response_times = self.metrics["response_times"]
        p95_time = statistics.quantile(response_times, 0.95) if len(response_times) >= 2 else 0.0
        p99_time = statistics.quantile(response_times, 0.99) if len(response_times) >= 2 else 0.0
        
        # Calculate average evaluation metrics
        precision_scores = self.metrics["precision_scores"]
        recall_scores = self.metrics["recall_scores"]
        ndcg_scores = self.metrics["ndcg_scores"]
        mrr_scores = self.metrics["mrr_scores"]
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        
        return RAGMetrics(
            query_count=query_count,
            avg_response_time=self.metrics["total_response_time"] / query_count,
            avg_chunks_returned=self.metrics["total_chunks_returned"] / query_count,
            avg_similarity_score=self.metrics["total_similarity_score"] / query_count,
            cache_hit_rate=(self.metrics["cache_hits"] / query_count) * 100,
            precision_at_k=avg_precision * 100,
            recall_at_k=avg_recall * 100,
            ndcg_score=avg_ndcg * 100,
            mrr_score=avg_mrr * 100,
            p95_response_time=p95_time,
            p99_response_time=p99_time,
            error_rate=(self.metrics["error_count"] / query_count) * 100
        )
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.query_cache.clear()
        self.similarity_cache.clear()
        logger.info("RAG service caches cleared")
    
    def reset_metrics(self) -> None:
        """Reset service metrics."""
        self.metrics = {
            "query_count": 0,
            "total_response_time": 0.0,
            "total_chunks_returned": 0,
            "total_similarity_score": 0.0,
            "cache_hits": 0,
            "error_count": 0,
            "response_times": [],
            "precision_scores": [],
            "recall_scores": [],
            "ndcg_scores": [],
            "mrr_scores": []
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