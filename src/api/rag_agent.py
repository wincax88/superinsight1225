"""
FastAPI endpoints for RAG and Agent testing in SuperInsight Platform.

Provides RESTful API for RAG and Agent testing interfaces.
"""

import logging
import time
import random
import uuid
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.rag import RAGService, RAGRequest, RAGResponse, DocumentChunk
from src.rag.models import RAGMetrics, RAGEvaluationRequest, RAGEvaluationResult
from src.agent import AgentService, AgentRequest, AgentResponse
from src.agent.models import (
    AgentMetrics, MultiTurnAgentRequest, MultiTurnAgentResponse,
    ConversationHistory
)

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


# Enhanced RAG Testing Endpoints
@router.post("/rag/evaluate", response_model=RAGEvaluationResult)
async def evaluate_rag_scenarios(request: RAGEvaluationRequest) -> RAGEvaluationResult:
    """Evaluate RAG performance across multiple test scenarios."""
    try:
        logger.info(f"Starting RAG evaluation: {request.scenario_name}")
        
        # Perform RAG evaluation
        result = rag_service.evaluate_rag_scenarios(request)
        
        return result
        
    except Exception as e:
        logger.error(f"RAG evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG evaluation failed: {str(e)}"
        )


@router.get("/rag/test-scenarios")
async def get_predefined_test_scenarios() -> JSONResponse:
    """Get predefined RAG test scenarios."""
    try:
        scenarios = [
            {
                "name": "basic_search",
                "description": "Basic document search functionality",
                "queries": [
                    "What is machine learning?",
                    "How does data processing work?",
                    "Explain artificial intelligence"
                ]
            },
            {
                "name": "domain_specific",
                "description": "Domain-specific knowledge retrieval",
                "queries": [
                    "Database optimization techniques",
                    "API security best practices",
                    "Cloud deployment strategies"
                ]
            },
            {
                "name": "multi_intent",
                "description": "Multi-intent and complex queries",
                "queries": [
                    "Compare machine learning and deep learning approaches for text classification",
                    "What are the pros and cons of microservices vs monolithic architecture?",
                    "How to implement authentication and authorization in REST APIs?"
                ]
            },
            {
                "name": "edge_cases",
                "description": "Edge cases and challenging queries",
                "queries": [
                    "very short query",
                    "This is an extremely long and detailed query that contains multiple concepts and ideas that might be challenging for the RAG system to process effectively and return relevant results",
                    "query with special characters: @#$%^&*()",
                    ""  # Empty query
                ]
            }
        ]
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "scenarios": scenarios,
                "total_scenarios": len(scenarios)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get test scenarios: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scenarios: {str(e)}"
        )


# Enhanced Agent Testing Endpoints
@router.post("/agent/multi-turn", response_model=MultiTurnAgentResponse)
async def execute_multi_turn_agent(request: MultiTurnAgentRequest) -> MultiTurnAgentResponse:
    """Execute multi-turn agent conversation."""
    try:
        logger.info(f"Multi-turn agent request: {request.task_type}")
        
        # Execute multi-turn task
        response = agent_service.execute_multi_turn_task(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Multi-turn agent execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-turn execution failed: {str(e)}"
        )


@router.get("/agent/conversations")
async def list_conversations(
    user_id: str = Query(None, description="Filter by user ID"),
    project_id: str = Query(None, description="Filter by project ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum conversations to return")
) -> JSONResponse:
    """List agent conversations."""
    try:
        conversations = agent_service.list_conversations(user_id, project_id)
        
        # Apply limit
        limited_conversations = conversations[:limit]
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "conversations": limited_conversations,
                "total_count": len(conversations),
                "returned_count": len(limited_conversations)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.get("/agent/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str) -> JSONResponse:
    """Get conversation history by ID."""
    try:
        conversation = agent_service.get_conversation_history(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "conversation_id": conversation.conversation_id,
                "message_count": len(conversation.messages),
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "project_id": conversation.project_id,
                "user_id": conversation.user_id,
                "context": conversation.context,
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata
                    }
                    for msg in conversation.messages
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )


@router.delete("/agent/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str) -> JSONResponse:
    """Clear a specific conversation."""
    try:
        success = agent_service.clear_conversation(conversation_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Conversation {conversation_id} cleared successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.post("/agent/conversations/cleanup")
async def cleanup_old_conversations(
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age in hours")
) -> JSONResponse:
    """Clean up old conversations."""
    try:
        cleared_count = agent_service.clear_old_conversations(max_age_hours)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Cleared {cleared_count} old conversations",
                "cleared_count": cleared_count,
                "max_age_hours": max_age_hours
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to cleanup conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup conversations: {str(e)}"
        )
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


@router.get("/test/performance/advanced")
async def test_advanced_performance(
    iterations: int = Query(10, ge=1, le=100, description="Number of test iterations"),
    concurrent_requests: int = Query(1, ge=1, le=10, description="Concurrent requests"),
    test_multi_turn: bool = Query(True, description="Include multi-turn conversation tests")
) -> JSONResponse:
    """Run advanced performance tests with concurrency and multi-turn support."""
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        logger.info(f"Running advanced performance test: {iterations} iterations, {concurrent_requests} concurrent")
        
        async def run_rag_test():
            """Run single RAG test."""
            rag_request = RAGRequest(
                query=f"performance test query {random.randint(1, 1000)}",
                top_k=5,
                similarity_threshold=0.3
            )
            start_time = time.time()
            response = rag_service.search_documents(rag_request)
            return time.time() - start_time, len(response.chunks)
        
        async def run_agent_test():
            """Run single Agent test."""
            agent_request = AgentRequest(
                task_type="classification",
                input_data={
                    "text": f"performance test text {random.randint(1, 1000)}",
                    "categories": ["test1", "test2", "test3"]
                }
            )
            start_time = time.time()
            response = agent_service.execute_task(agent_request)
            return time.time() - start_time, response.confidence
        
        async def run_multi_turn_test():
            """Run multi-turn conversation test."""
            multi_turn_request = MultiTurnAgentRequest(
                message=f"Hello, this is test message {random.randint(1, 1000)}",
                task_type="conversation",
                context_window=5
            )
            start_time = time.time()
            response = agent_service.execute_multi_turn_task(multi_turn_request)
            return time.time() - start_time, response.confidence
        
        # Run concurrent tests
        rag_times = []
        agent_times = []
        multi_turn_times = []
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            # Submit all tasks
            rag_futures = []
            agent_futures = []
            multi_turn_futures = []
            
            for i in range(iterations):
                rag_futures.append(executor.submit(lambda: asyncio.run(run_rag_test())))
                agent_futures.append(executor.submit(lambda: asyncio.run(run_agent_test())))
                
                if test_multi_turn:
                    multi_turn_futures.append(executor.submit(lambda: asyncio.run(run_multi_turn_test())))
            
            # Collect results
            for future in rag_futures:
                exec_time, chunks = future.result()
                rag_times.append(exec_time)
            
            for future in agent_futures:
                exec_time, confidence = future.result()
                agent_times.append(exec_time)
            
            if test_multi_turn:
                for future in multi_turn_futures:
                    exec_time, confidence = future.result()
                    multi_turn_times.append(exec_time)
        
        # Calculate advanced statistics
        def calculate_stats(times):
            if not times:
                return {}
            
            times.sort()
            return {
                "count": len(times),
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "median": times[len(times) // 2],
                "p95": times[int(len(times) * 0.95)] if len(times) > 1 else times[0],
                "p99": times[int(len(times) * 0.99)] if len(times) > 1 else times[0],
                "total": sum(times)
            }
        
        result = {
            "test_configuration": {
                "iterations": iterations,
                "concurrent_requests": concurrent_requests,
                "test_multi_turn": test_multi_turn
            },
            "rag_performance": calculate_stats(rag_times),
            "agent_performance": calculate_stats(agent_times),
            "overall_throughput": {
                "total_requests": len(rag_times) + len(agent_times) + len(multi_turn_times),
                "total_time": max(
                    sum(rag_times) if rag_times else 0,
                    sum(agent_times) if agent_times else 0,
                    sum(multi_turn_times) if multi_turn_times else 0
                ),
                "requests_per_second": (len(rag_times) + len(agent_times) + len(multi_turn_times)) / max(
                    sum(rag_times) if rag_times else 1,
                    sum(agent_times) if agent_times else 1,
                    sum(multi_turn_times) if multi_turn_times else 1
                )
            }
        }
        
        if test_multi_turn:
            result["multi_turn_performance"] = calculate_stats(multi_turn_times)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "performance_results": result,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Advanced performance test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced performance test failed: {str(e)}"
        )


@router.post("/test/rag-agent-conversation")
async def test_rag_agent_conversation_pipeline(
    initial_query: str = Query(..., description="Initial RAG query"),
    follow_up_questions: List[str] = Query(..., description="Follow-up questions for conversation"),
    project_id: str = Query(None, description="Project ID filter")
) -> JSONResponse:
    """Test RAG + Agent conversation pipeline with multi-turn support."""
    try:
        logger.info(f"Testing RAG-Agent conversation pipeline")
        
        conversation_id = str(uuid.uuid4())
        pipeline_results = []
        
        # Step 1: Initial RAG search
        rag_request = RAGRequest(
            query=initial_query,
            project_id=project_id,
            top_k=3,
            similarity_threshold=0.5
        )
        
        rag_response = rag_service.search_documents(rag_request)
        
        # Step 2: Start conversation with RAG results
        if rag_response.chunks:
            combined_content = " ".join([chunk.content for chunk in rag_response.chunks[:2]])
            
            initial_message = f"Based on this information: {combined_content[:500]}... {initial_query}"
            
            multi_turn_request = MultiTurnAgentRequest(
                conversation_id=conversation_id,
                message=initial_message,
                task_type="conversation",
                context_window=10,
                project_id=project_id
            )
            
            agent_response = agent_service.execute_multi_turn_task(multi_turn_request)
            
            pipeline_results.append({
                "step": "initial_rag_conversation",
                "rag_query": initial_query,
                "rag_chunks_found": len(rag_response.chunks),
                "rag_processing_time": rag_response.processing_time,
                "agent_response": agent_response.response,
                "agent_confidence": agent_response.confidence,
                "agent_execution_time": agent_response.execution_time,
                "conversation_id": conversation_id
            })
            
            # Step 3: Continue conversation with follow-up questions
            for i, follow_up in enumerate(follow_up_questions):
                follow_up_request = MultiTurnAgentRequest(
                    conversation_id=conversation_id,
                    message=follow_up,
                    task_type="conversation",
                    context_window=10,
                    project_id=project_id
                )
                
                follow_up_response = agent_service.execute_multi_turn_task(follow_up_request)
                
                pipeline_results.append({
                    "step": f"follow_up_{i+1}",
                    "question": follow_up,
                    "agent_response": follow_up_response.response,
                    "agent_confidence": follow_up_response.confidence,
                    "agent_execution_time": follow_up_response.execution_time,
                    "context_messages_used": follow_up_response.context_used,
                    "total_conversation_messages": follow_up_response.total_messages
                })
        
        # Calculate total pipeline metrics
        total_time = sum(result.get("rag_processing_time", 0) + result.get("agent_execution_time", 0) 
                        for result in pipeline_results)
        
        avg_confidence = sum(result.get("agent_confidence", 0) for result in pipeline_results) / len(pipeline_results) if pipeline_results else 0
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "conversation_pipeline": {
                    "conversation_id": conversation_id,
                    "initial_query": initial_query,
                    "follow_up_count": len(follow_up_questions),
                    "total_steps": len(pipeline_results),
                    "total_time": total_time,
                    "avg_confidence": avg_confidence,
                    "steps": pipeline_results
                },
                "conversation_summary": {
                    "messages_exchanged": sum(result.get("total_conversation_messages", 0) for result in pipeline_results[-1:]),
                    "context_preservation": all(result.get("context_messages_used", 0) > 0 for result in pipeline_results[1:]),
                    "response_quality": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"RAG-Agent conversation pipeline test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversation pipeline test failed: {str(e)}"
        )
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