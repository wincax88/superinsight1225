"""
FastAPI endpoints for RAG and Agent testing in SuperInsight Platform.

Provides RESTful API for RAG and Agent testing interfaces.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.rag import RAGService, RAGRequest, RAGResponse, DocumentChunk
from src.rag.models import RAGMetrics
from src.agent import AgentService, AgentRequest, AgentResponse
from src.agent.models import AgentMetrics

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["RAG & Agent Testing"])

# Global service instances
rag_service = RAGService()
agent_service = AgentService()


class RAGTestResponse(BaseModel):
    """Response model for RAG testing."""
    success: bool = Field(..., description="Test success status")
    query: str = Field(..., description="Original query")
    results_count: int = Field(..., description="Number of results returned")
    processing_time: float = Field(..., description="Processing time in seconds")
    chunks: List[DocumentChunk] = Field(..., description="Retrieved chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentTestResponse(BaseModel):
    """Response model for Agent testing."""
    success: bool = Field(..., description="Test success status")
    task_type: str = Field(..., description="Agent task type")
    execution_time: float = Field(..., description="Execution time in seconds")
    steps_count: int = Field(..., description="Number of execution steps")
    confidence: float = Field(..., description="Overall confidence score")
    result: Dict[str, Any] = Field(..., description="Task result")
    error: str = Field(None, description="Error message if failed")


# RAG Testing Endpoints
@router.post("/rag/search", response_model=RAGTestResponse)
async def test_rag_search(request: RAGRequest) -> RAGTestResponse:
    """Test RAG search functionality."""
    try:
        logger.info(f"RAG search test: {request.query[:50]}...")
        
        # Perform RAG search
        response = rag_service.search_documents(request)
        
        return RAGTestResponse(
            success=True,
            query=response.query,
            results_count=response.total_chunks,
            processing_time=response.processing_time,
            chunks=response.chunks,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"RAG search test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG search failed: {str(e)}"
        )


@router.get("/rag/metrics", response_model=RAGMetrics)
async def get_rag_metrics() -> RAGMetrics:
    """Get RAG service performance metrics."""
    return rag_service.get_metrics()


@router.post("/rag/clear-cache")
async def clear_rag_cache() -> JSONResponse:
    """Clear RAG service cache."""
    try:
        rag_service.clear_cache()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "RAG cache cleared successfully"}
        )
    except Exception as e:
        logger.error(f"Failed to clear RAG cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/rag/reset-metrics")
async def reset_rag_metrics() -> JSONResponse:
    """Reset RAG service metrics."""
    try:
        rag_service.reset_metrics()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "RAG metrics reset successfully"}
        )
    except Exception as e:
        logger.error(f"Failed to reset RAG metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {str(e)}"
        )


@router.get("/rag/document/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    chunk_size: int = Query(512, ge=100, le=2000, description="Chunk size"),
    chunk_overlap: int = Query(50, ge=0, le=500, description="Chunk overlap")
) -> List[DocumentChunk]:
    """Get chunks for a specific document."""
    try:
        chunks = rag_service.get_document_chunks(document_id, chunk_size, chunk_overlap)
        return chunks
    except Exception as e:
        logger.error(f"Failed to get document chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunks: {str(e)}"
        )


# Agent Testing Endpoints
@router.post("/agent/execute", response_model=AgentTestResponse)
async def test_agent_execution(request: AgentRequest) -> AgentTestResponse:
    """Test Agent execution functionality."""
    try:
        logger.info(f"Agent execution test: {request.task_type}")
        
        # Execute agent task
        response = agent_service.execute_task(request)
        
        return AgentTestResponse(
            success=response.status == "completed",
            task_type=response.task_type,
            execution_time=response.execution_time,
            steps_count=response.total_steps,
            confidence=response.confidence,
            result=response.result,
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Agent execution test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}"
        )


@router.get("/agent/metrics", response_model=AgentMetrics)
async def get_agent_metrics() -> AgentMetrics:
    """Get Agent service performance metrics."""
    return agent_service.get_metrics()


@router.post("/agent/reset-metrics")
async def reset_agent_metrics() -> JSONResponse:
    """Reset Agent service metrics."""
    try:
        agent_service.reset_metrics()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Agent metrics reset successfully"}
        )
    except Exception as e:
        logger.error(f"Failed to reset Agent metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {str(e)}"
        )


@router.get("/agent/tasks")
async def get_supported_agent_tasks() -> JSONResponse:
    """Get list of supported Agent task types."""
    try:
        tasks = agent_service.get_supported_tasks()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "supported_tasks": tasks,
                "total_tasks": len(tasks)
            }
        )
    except Exception as e:
        logger.error(f"Failed to get supported tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get tasks: {str(e)}"
        )


# Combined Testing Endpoints
@router.post("/test/rag-agent-pipeline")
async def test_rag_agent_pipeline(
    rag_query: str = Query(..., description="RAG search query"),
    agent_task_type: str = Query(..., description="Agent task type"),
    project_id: str = Query(None, description="Project ID filter")
) -> JSONResponse:
    """Test combined RAG + Agent pipeline."""
    try:
        logger.info(f"Testing RAG-Agent pipeline: {rag_query} -> {agent_task_type}")
        
        # Step 1: RAG search
        rag_request = RAGRequest(
            query=rag_query,
            project_id=project_id,
            top_k=3,
            similarity_threshold=0.5
        )
        
        rag_response = rag_service.search_documents(rag_request)
        
        # Step 2: Use RAG results in Agent task
        if rag_response.chunks:
            # Combine chunk content for agent input
            combined_content = " ".join([chunk.content for chunk in rag_response.chunks[:3]])
            
            agent_request = AgentRequest(
                task_type=agent_task_type,
                input_data={
                    "text": combined_content,
                    "context": rag_query,
                    "rag_chunks": len(rag_response.chunks)
                },
                project_id=project_id
            )
            
            agent_response = agent_service.execute_task(agent_request)
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": True,
                    "pipeline_result": {
                        "rag_phase": {
                            "query": rag_response.query,
                            "chunks_found": rag_response.total_chunks,
                            "processing_time": rag_response.processing_time
                        },
                        "agent_phase": {
                            "task_type": agent_response.task_type,
                            "status": agent_response.status,
                            "steps": agent_response.total_steps,
                            "confidence": agent_response.confidence,
                            "execution_time": agent_response.execution_time,
                            "result": agent_response.result
                        }
                    },
                    "total_time": rag_response.processing_time + agent_response.execution_time
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": False,
                    "message": "No relevant documents found in RAG search",
                    "rag_result": {
                        "query": rag_response.query,
                        "chunks_found": 0,
                        "processing_time": rag_response.processing_time
                    }
                }
            )
            
    except Exception as e:
        logger.error(f"RAG-Agent pipeline test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline test failed: {str(e)}"
        )


@router.get("/test/health")
async def test_services_health() -> JSONResponse:
    """Check health status of RAG and Agent services."""
    try:
        # Test RAG service
        rag_test = RAGRequest(
            query="test query",
            top_k=1,
            similarity_threshold=0.1
        )
        rag_response = rag_service.search_documents(rag_test)
        rag_healthy = True
        
        # Test Agent service
        agent_test = AgentRequest(
            task_type="classification",
            input_data={"text": "test text", "categories": ["test"]}
        )
        agent_response = agent_service.execute_task(agent_test)
        agent_healthy = agent_response.status in ["completed", "failed"]  # Both are valid responses
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "services": {
                    "rag_service": {
                        "healthy": rag_healthy,
                        "response_time": rag_response.processing_time,
                        "metrics": rag_service.get_metrics().dict()
                    },
                    "agent_service": {
                        "healthy": agent_healthy,
                        "response_time": agent_response.execution_time,
                        "metrics": agent_service.get_metrics().dict()
                    }
                },
                "overall_health": rag_healthy and agent_healthy,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "services": {
                    "rag_service": {"healthy": False},
                    "agent_service": {"healthy": False}
                },
                "overall_health": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/test/performance")
async def test_services_performance(
    iterations: int = Query(10, ge=1, le=100, description="Number of test iterations")
) -> JSONResponse:
    """Run performance tests on RAG and Agent services."""
    try:
        logger.info(f"Running performance test with {iterations} iterations")
        
        rag_times = []
        agent_times = []
        
        # Run performance tests
        for i in range(iterations):
            # Test RAG performance
            rag_request = RAGRequest(
                query=f"performance test query {i}",
                top_k=5,
                similarity_threshold=0.3
            )
            rag_response = rag_service.search_documents(rag_request)
            rag_times.append(rag_response.processing_time)
            
            # Test Agent performance
            agent_request = AgentRequest(
                task_type="classification",
                input_data={
                    "text": f"performance test text {i}",
                    "categories": ["test1", "test2", "test3"]
                }
            )
            agent_response = agent_service.execute_task(agent_request)
            agent_times.append(agent_response.execution_time)
        
        # Calculate statistics
        rag_avg = sum(rag_times) / len(rag_times)
        rag_min = min(rag_times)
        rag_max = max(rag_times)
        
        agent_avg = sum(agent_times) / len(agent_times)
        agent_min = min(agent_times)
        agent_max = max(agent_times)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "performance_results": {
                    "iterations": iterations,
                    "rag_service": {
                        "avg_response_time": rag_avg,
                        "min_response_time": rag_min,
                        "max_response_time": rag_max,
                        "total_time": sum(rag_times)
                    },
                    "agent_service": {
                        "avg_response_time": agent_avg,
                        "min_response_time": agent_min,
                        "max_response_time": agent_max,
                        "total_time": sum(agent_times)
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance test failed: {str(e)}"
        )