"""
Agent service for SuperInsight Platform.

Provides Agent testing functionality for AI applications with conversation history management.
"""

import logging
import time
import random
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .models import (
    AgentRequest, AgentResponse, AgentStep, AgentMetrics,
    ConversationHistory, ConversationMessage, 
    MultiTurnAgentRequest, MultiTurnAgentResponse
)

logger = logging.getLogger(__name__)


class AgentService:
    """Service for Agent operations and testing with conversation management."""
    
    def __init__(self):
        """Initialize Agent service with enhanced conversation management."""
        # Enhanced metrics tracking
        self.metrics = {
            "task_count": 0,
            "successful_tasks": 0,
            "total_execution_time": 0.0,
            "total_steps": 0,
            "total_confidence": 0.0,
            "conversation_count": 0,
            "multi_turn_sessions": 0,
            "response_quality_scores": [],
            "context_utilization_scores": [],
            "conversation_lengths": [],
            "avg_response_time": 0.0,
            "conversation_satisfaction_scores": []
        }
        
        # Enhanced conversation history storage with better management
        self.conversations: Dict[str, ConversationHistory] = {}
        self.conversation_analytics: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced task handlers with conversation support
        self.task_handlers = {
            'classification': self._handle_classification,
            'extraction': self._handle_extraction,
            'summarization': self._handle_summarization,
            'question_answering': self._handle_question_answering,
            'text_generation': self._handle_text_generation,
            'analysis': self._handle_analysis,
            'conversation': self._handle_conversation,
            'chat': self._handle_chat,
            'follow_up': self._handle_follow_up,
            'clarification': self._handle_clarification,
            'context_aware': self._handle_context_aware
        }
        
        # Enhanced context management settings
        self.max_conversation_age_hours = 24
        self.max_conversations = 10000
        self.context_optimization_enabled = True
        self.response_quality_threshold = 0.7
        
        # Response optimization cache
        self.response_templates: Dict[str, List[str]] = {
            "greeting": [
                "Hello! I'm here to help you. What can I assist you with today?",
                "Hi there! How can I help you?",
                "Welcome! What would you like to know or discuss?"
            ],
            "clarification": [
                "Could you provide more details about that?",
                "I'd like to understand better. Can you elaborate?",
                "That's interesting. Can you tell me more?"
            ],
            "acknowledgment": [
                "I understand what you're saying.",
                "That makes sense. Let me help with that.",
                "I see. Here's what I can do to help."
            ]
        }
    
    def execute_task(self, request: AgentRequest) -> AgentResponse:
        """Execute an agent task."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting agent task: {request.task_type}")
            
            # Get task handler
            handler = self.task_handlers.get(request.task_type)
            if not handler:
                raise ValueError(f"Unsupported task type: {request.task_type}")
            
            # Execute task with timeout simulation
            steps = []
            result = {}
            
            # Simulate agent execution steps
            for i in range(min(request.max_iterations, 5)):  # Limit to 5 steps for demo
                step_start = time.time()
                
                # Execute step
                step_result = handler(request, i + 1, steps)
                
                step_time = time.time() - step_start
                
                step = AgentStep(
                    step_number=i + 1,
                    action=step_result['action'],
                    input_data=step_result['input'],
                    output_data=step_result['output'],
                    confidence=step_result['confidence'],
                    execution_time=step_time
                )
                
                steps.append(step)
                
                # Check if task is complete
                if step_result.get('complete', False):
                    result = step_result['output']
                    break
                
                # Simulate timeout check
                if time.time() - start_time > request.timeout:
                    raise TimeoutError(f"Task exceeded timeout of {request.timeout} seconds")
            
            # Calculate overall confidence
            if steps:
                overall_confidence = sum(step.confidence for step in steps) / len(steps)
            else:
                overall_confidence = 0.0
            
            execution_time = time.time() - start_time
            
            # Create response
            response = AgentResponse(
                task_type=request.task_type,
                status="completed",
                result=result,
                steps=steps,
                total_steps=len(steps),
                execution_time=execution_time,
                confidence=overall_confidence,
                metadata={
                    "max_iterations": request.max_iterations,
                    "timeout": request.timeout,
                    "project_id": request.project_id
                }
            )
            
            # Update metrics
            self._update_metrics(response, success=True)
            
            logger.info(f"Agent task completed: {len(steps)} steps in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent task failed: {e}")
            
            # Create error response
            response = AgentResponse(
                task_type=request.task_type,
                status="failed",
                result={},
                steps=steps if 'steps' in locals() else [],
                total_steps=len(steps) if 'steps' in locals() else 0,
                execution_time=execution_time,
                confidence=0.0,
                error=str(e),
                metadata={
                    "max_iterations": request.max_iterations,
                    "timeout": request.timeout,
                    "project_id": request.project_id
                }
            )
            
            # Update metrics
            self._update_metrics(response, success=False)
            
            return response
    
    def _handle_classification(self, request: AgentRequest, step_num: int, 
                             previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle classification task."""
        text = request.input_data.get('text', '')
        categories = request.input_data.get('categories', ['positive', 'negative', 'neutral'])
        
        if step_num == 1:
            # Step 1: Analyze text
            return {
                'action': 'analyze_text',
                'input': {'text': text},
                'output': {
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'analysis': 'Text analyzed for classification'
                },
                'confidence': 0.8,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Classify
            # Simulate classification result
            predicted_category = random.choice(categories)
            confidence = random.uniform(0.7, 0.95)
            
            return {
                'action': 'classify',
                'input': {'text': text, 'categories': categories},
                'output': {
                    'predicted_category': predicted_category,
                    'confidence': confidence,
                    'all_scores': {cat: random.uniform(0.1, 0.9) for cat in categories}
                },
                'confidence': confidence,
                'complete': True
            }
    
    def _handle_extraction(self, request: AgentRequest, step_num: int, 
                          previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle extraction task."""
        text = request.input_data.get('text', '')
        entities = request.input_data.get('entities', ['PERSON', 'ORG', 'LOC'])
        
        if step_num == 1:
            # Step 1: Preprocess text
            return {
                'action': 'preprocess',
                'input': {'text': text},
                'output': {
                    'cleaned_text': text.strip(),
                    'sentences': text.split('.'),
                    'preprocessing': 'Text preprocessed for extraction'
                },
                'confidence': 0.9,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Extract entities
            # Simulate entity extraction
            extracted_entities = []
            words = text.split()
            
            for i, word in enumerate(words[:5]):  # Limit for demo
                if len(word) > 3:  # Simple heuristic
                    extracted_entities.append({
                        'text': word,
                        'label': random.choice(entities),
                        'start': i * 5,  # Approximate position
                        'end': i * 5 + len(word),
                        'confidence': random.uniform(0.6, 0.9)
                    })
            
            return {
                'action': 'extract_entities',
                'input': {'text': text, 'entities': entities},
                'output': {
                    'entities': extracted_entities,
                    'entity_count': len(extracted_entities)
                },
                'confidence': 0.85,
                'complete': True
            }
    
    def _handle_summarization(self, request: AgentRequest, step_num: int, 
                            previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle summarization task."""
        text = request.input_data.get('text', '')
        max_length = request.input_data.get('max_length', 100)
        
        if step_num == 1:
            # Step 1: Analyze text structure
            sentences = text.split('.')
            return {
                'action': 'analyze_structure',
                'input': {'text': text},
                'output': {
                    'sentence_count': len(sentences),
                    'word_count': len(text.split()),
                    'structure_analysis': 'Text structure analyzed'
                },
                'confidence': 0.85,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Generate summary
            # Simulate summarization
            words = text.split()
            summary_words = words[:min(max_length // 5, len(words))]  # Simple truncation
            summary = ' '.join(summary_words) + '...'
            
            return {
                'action': 'generate_summary',
                'input': {'text': text, 'max_length': max_length},
                'output': {
                    'summary': summary,
                    'summary_length': len(summary),
                    'compression_ratio': len(summary) / len(text)
                },
                'confidence': 0.8,
                'complete': True
            }
    
    def _handle_question_answering(self, request: AgentRequest, step_num: int, 
                                 previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle question answering task."""
        question = request.input_data.get('question', '')
        context = request.input_data.get('context', '')
        
        if step_num == 1:
            # Step 1: Analyze question
            return {
                'action': 'analyze_question',
                'input': {'question': question},
                'output': {
                    'question_type': 'factual',  # Simplified
                    'question_words': question.split(),
                    'analysis': 'Question analyzed for answering'
                },
                'confidence': 0.9,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Search context
            return {
                'action': 'search_context',
                'input': {'question': question, 'context': context},
                'output': {
                    'relevant_passages': [context[:200]],  # Simplified
                    'search_results': 'Context searched for relevant information'
                },
                'confidence': 0.8,
                'complete': False
            }
        elif step_num == 3:
            # Step 3: Generate answer
            # Simulate answer generation
            answer = f"Based on the context, the answer is related to: {question[:50]}..."
            
            return {
                'action': 'generate_answer',
                'input': {'question': question, 'context': context},
                'output': {
                    'answer': answer,
                    'confidence_score': random.uniform(0.7, 0.9)
                },
                'confidence': 0.85,
                'complete': True
            }
    
    def _handle_text_generation(self, request: AgentRequest, step_num: int, 
                              previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle text generation task."""
        prompt = request.input_data.get('prompt', '')
        max_tokens = request.input_data.get('max_tokens', 100)
        
        if step_num == 1:
            # Step 1: Analyze prompt
            return {
                'action': 'analyze_prompt',
                'input': {'prompt': prompt},
                'output': {
                    'prompt_length': len(prompt),
                    'prompt_type': 'creative',  # Simplified
                    'analysis': 'Prompt analyzed for generation'
                },
                'confidence': 0.9,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Generate text
            # Simulate text generation
            generated_text = f"{prompt} This is a generated continuation of the text. " * (max_tokens // 50)
            generated_text = generated_text[:max_tokens]
            
            return {
                'action': 'generate_text',
                'input': {'prompt': prompt, 'max_tokens': max_tokens},
                'output': {
                    'generated_text': generated_text,
                    'token_count': len(generated_text.split()),
                    'generation_quality': 'high'
                },
                'confidence': 0.8,
                'complete': True
            }
    
    def _handle_analysis(self, request: AgentRequest, step_num: int, 
                        previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle analysis task."""
        data = request.input_data.get('data', {})
        analysis_type = request.input_data.get('analysis_type', 'general')
        
        if step_num == 1:
            # Step 1: Data preprocessing
            return {
                'action': 'preprocess_data',
                'input': {'data': data},
                'output': {
                    'data_size': len(str(data)),
                    'data_type': type(data).__name__,
                    'preprocessing': 'Data preprocessed for analysis'
                },
                'confidence': 0.9,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Perform analysis
            # Simulate analysis results
            analysis_results = {
                'insights': [
                    'Data shows positive trends',
                    'Key patterns identified',
                    'Anomalies detected in subset'
                ],
                'metrics': {
                    'accuracy': random.uniform(0.8, 0.95),
                    'completeness': random.uniform(0.85, 0.98),
                    'quality_score': random.uniform(0.7, 0.9)
                },
                'recommendations': [
                    'Continue monitoring trends',
                    'Investigate anomalies',
                    'Expand data collection'
                ]
            }
            
            return {
                'action': 'perform_analysis',
                'input': {'data': data, 'analysis_type': analysis_type},
                'output': analysis_results,
                'confidence': 0.85,
                'complete': True
            }
    
    
    def execute_multi_turn_task(self, request: MultiTurnAgentRequest) -> MultiTurnAgentResponse:
        """Execute a multi-turn agent task with conversation history."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting multi-turn agent task: {request.task_type}")
            
            # Get or create conversation
            conversation = self._get_or_create_conversation(
                request.conversation_id, 
                request.project_id, 
                request.user_id
            )
            
            # Add user message to conversation
            user_message = conversation.add_message("user", request.message)
            
            # Get conversation context
            context_messages = conversation.get_context_window(request.max_context_tokens)
            
            # Prepare enhanced input with conversation context
            enhanced_input = {
                "current_message": request.message,
                "conversation_history": [
                    {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()}
                    for msg in context_messages[-request.context_window:]
                ],
                "conversation_context": conversation.context,
                "preserve_context": request.preserve_context
            }
            
            # Create enhanced agent request
            agent_request = AgentRequest(
                task_type=request.task_type,
                input_data=enhanced_input,
                context={"conversation_id": conversation.conversation_id},
                project_id=request.project_id,
                max_iterations=5,  # Limit for multi-turn
                timeout=30
            )
            
            # Execute agent task
            agent_response = self.execute_task(agent_request)
            
            # Extract response content
            response_content = self._extract_response_content(agent_response)
            
            # Add agent response to conversation
            agent_message = conversation.add_message(
                "assistant", 
                response_content,
                {
                    "task_type": request.task_type,
                    "confidence": agent_response.confidence,
                    "execution_time": agent_response.execution_time,
                    "steps": agent_response.total_steps
                }
            )
            
            # Update conversation context if needed
            if request.preserve_context:
                self._update_conversation_context(conversation, agent_response)
            
            execution_time = time.time() - start_time
            
            # Create multi-turn response
            response = MultiTurnAgentResponse(
                conversation_id=conversation.conversation_id,
                message_id=agent_message.id,
                response=response_content,
                task_type=request.task_type,
                execution_time=execution_time,
                confidence=agent_response.confidence,
                context_used=len(context_messages),
                total_messages=len(conversation.messages),
                metadata={
                    "agent_steps": agent_response.total_steps,
                    "context_preserved": request.preserve_context,
                    "conversation_age": (datetime.now() - conversation.created_at).total_seconds()
                }
            )
            
            # Update metrics
            self.metrics["multi_turn_sessions"] += 1
            
            logger.info(f"Multi-turn task completed: {len(conversation.messages)} total messages")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Multi-turn agent task failed: {e}")
            
            # Create error response
            return MultiTurnAgentResponse(
                conversation_id=request.conversation_id or "error",
                message_id="error",
                response=f"Error: {str(e)}",
                task_type=request.task_type,
                execution_time=execution_time,
                confidence=0.0,
                context_used=0,
                total_messages=0,
                metadata={"error": str(e)}
            )
    
    def _get_or_create_conversation(self, conversation_id: Optional[str], 
                                  project_id: Optional[str], 
                                  user_id: Optional[str]) -> ConversationHistory:
        """Get existing conversation or create new one with enhanced analytics."""
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            # Check if conversation is still valid (not too old)
            age_hours = (datetime.now() - conversation.created_at).total_seconds() / 3600
            if age_hours <= self.max_conversation_age_hours:
                # Update conversation analytics
                self._update_conversation_analytics(conversation_id, "accessed")
                return conversation
            else:
                # Remove expired conversation
                self._archive_conversation_analytics(conversation_id)
                del self.conversations[conversation_id]
        
        # Create new conversation with enhanced tracking
        new_conversation_id = conversation_id or str(uuid.uuid4())
        conversation = ConversationHistory(
            conversation_id=new_conversation_id,
            project_id=project_id,
            user_id=user_id
        )
        
        # Initialize conversation analytics
        self.conversation_analytics[new_conversation_id] = {
            "created_at": datetime.now(),
            "message_count": 0,
            "avg_response_time": 0.0,
            "context_utilization_score": 0.0,
            "satisfaction_indicators": [],
            "topic_progression": [],
            "interaction_quality": 0.0,
            "user_engagement_score": 0.0
        }
        
        # Manage conversation storage size with intelligent cleanup
        if len(self.conversations) >= self.max_conversations:
            self._intelligent_conversation_cleanup()
        
        self.conversations[new_conversation_id] = conversation
        self.metrics["conversation_count"] += 1
        
        return conversation
    
    def _intelligent_conversation_cleanup(self) -> None:
        """Intelligently clean up conversations based on usage patterns."""
        # Score conversations for cleanup priority
        conversation_scores = []
        
        for conv_id, conv in self.conversations.items():
            analytics = self.conversation_analytics.get(conv_id, {})
            
            # Calculate cleanup score (lower = higher priority for removal)
            age_hours = (datetime.now() - conv.created_at).total_seconds() / 3600
            message_count = len(conv.messages)
            last_activity_hours = (datetime.now() - conv.updated_at).total_seconds() / 3600
            
            # Score factors (lower score = more likely to be removed)
            age_factor = max(0, 1 - (age_hours / self.max_conversation_age_hours))
            activity_factor = min(1, message_count / 10)  # Normalize to 10 messages
            recency_factor = max(0, 1 - (last_activity_hours / 24))  # Last 24 hours
            
            cleanup_score = (age_factor * 0.3 + activity_factor * 0.4 + recency_factor * 0.3)
            
            conversation_scores.append((conv_id, cleanup_score))
        
        # Sort by cleanup score and remove lowest scoring conversations
        conversation_scores.sort(key=lambda x: x[1])
        conversations_to_remove = conversation_scores[:self.max_conversations // 5]  # Remove 20%
        
        for conv_id, _ in conversations_to_remove:
            self._archive_conversation_analytics(conv_id)
            del self.conversations[conv_id]
        
        logger.info(f"Cleaned up {len(conversations_to_remove)} conversations using intelligent scoring")
    
    def _update_conversation_analytics(self, conversation_id: str, event_type: str, 
                                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update conversation analytics with enhanced tracking."""
        if conversation_id not in self.conversation_analytics:
            return
        
        analytics = self.conversation_analytics[conversation_id]
        
        if event_type == "message_added":
            analytics["message_count"] += 1
            
            if metadata:
                # Update response time tracking
                if "response_time" in metadata:
                    current_avg = analytics["avg_response_time"]
                    count = analytics["message_count"]
                    analytics["avg_response_time"] = ((current_avg * (count - 1)) + metadata["response_time"]) / count
                
                # Track context utilization
                if "context_score" in metadata:
                    analytics["context_utilization_score"] = metadata["context_score"]
                
                # Track satisfaction indicators
                if "confidence" in metadata and metadata["confidence"] > 0.8:
                    analytics["satisfaction_indicators"].append("high_confidence")
                
                # Track topic progression
                if "topics" in metadata:
                    analytics["topic_progression"].extend(metadata["topics"])
        
        elif event_type == "accessed":
            # Track conversation access patterns
            analytics.setdefault("access_count", 0)
            analytics["access_count"] += 1
    
    def _archive_conversation_analytics(self, conversation_id: str) -> None:
        """Archive conversation analytics before cleanup."""
        if conversation_id in self.conversation_analytics:
            analytics = self.conversation_analytics[conversation_id]
            
            # Update global metrics
            if analytics["message_count"] > 0:
                self.metrics["conversation_lengths"].append(analytics["message_count"])
                self.metrics["response_quality_scores"].append(analytics.get("interaction_quality", 0.0))
                self.metrics["context_utilization_scores"].append(analytics.get("context_utilization_score", 0.0))
            
            # Remove from active analytics
            del self.conversation_analytics[conversation_id]
    
    def _extract_response_content(self, agent_response: AgentResponse) -> str:
        """Extract meaningful response content from agent response with enhanced quality."""
        if agent_response.error:
            return f"I encountered an error: {agent_response.error}"
        
        # Extract content based on task type with enhanced responses
        result = agent_response.result
        
        if agent_response.task_type == "classification":
            category = result.get("predicted_category", "unknown")
            confidence = result.get("confidence", 0.0)
            all_scores = result.get("all_scores", {})
            
            response = f"I classify this as '{category}' with {confidence:.1%} confidence."
            
            # Add additional context if available
            if all_scores and len(all_scores) > 1:
                sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_scores) > 1:
                    second_best = sorted_scores[1]
                    response += f" The second most likely category is '{second_best[0]}' at {second_best[1]:.1%}."
            
            return response
        
        elif agent_response.task_type == "extraction":
            entities = result.get("entities", [])
            if entities:
                # Group entities by type for better presentation
                entity_groups = {}
                for entity in entities[:10]:  # Limit to top 10
                    label = entity['label']
                    if label not in entity_groups:
                        entity_groups[label] = []
                    entity_groups[label].append(entity['text'])
                
                response_parts = []
                for label, texts in entity_groups.items():
                    unique_texts = list(set(texts))  # Remove duplicates
                    response_parts.append(f"{label}: {', '.join(unique_texts[:3])}")
                
                return f"I found these entities: {'; '.join(response_parts)}"
            else:
                return "I didn't find any notable entities in the text."
        
        elif agent_response.task_type == "summarization":
            summary = result.get("summary", "")
            compression_ratio = result.get("compression_ratio", 0.0)
            
            response = f"Here's a summary: {summary}"
            if compression_ratio > 0:
                response += f" (Compressed to {compression_ratio:.1%} of original length)"
            
            return response
        
        elif agent_response.task_type == "question_answering":
            answer = result.get("answer", "")
            confidence_score = result.get("confidence_score", 0.0)
            
            if answer:
                response = answer
                if confidence_score > 0:
                    response += f" (Confidence: {confidence_score:.1%})"
                return response
            else:
                return "I couldn't find a clear answer to your question. Could you provide more context or rephrase it?"
        
        elif agent_response.task_type == "text_generation":
            generated = result.get("generated_text", "")
            token_count = result.get("token_count", 0)
            
            if generated:
                response = generated
                if token_count > 0:
                    response += f" ({token_count} tokens generated)"
                return response
            else:
                return "I couldn't generate appropriate text for your request. Could you provide a more specific prompt?"
        
        elif agent_response.task_type == "analysis":
            insights = result.get("insights", [])
            metrics = result.get("metrics", {})
            recommendations = result.get("recommendations", [])
            
            response_parts = []
            
            if insights:
                response_parts.append(f"Key insights: {'; '.join(insights[:3])}")
            
            if metrics:
                metric_strs = [f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in list(metrics.items())[:3]]
                response_parts.append(f"Metrics: {', '.join(metric_strs)}")
            
            if recommendations:
                response_parts.append(f"Recommendations: {'; '.join(recommendations[:2])}")
            
            return ". ".join(response_parts) if response_parts else "Analysis completed successfully."
        
        elif agent_response.task_type in ["conversation", "chat"]:
            return result.get("response", "I'm here to help! What would you like to know?")
        
        elif agent_response.task_type == "follow_up":
            return result.get("response", "I understand your follow-up question. Let me help with that.")
        
        elif agent_response.task_type == "clarification":
            return result.get("response", "I'd be happy to clarify that for you.")
        
        elif agent_response.task_type == "context_aware":
            response = result.get("response", "")
            context_utilized = result.get("context_utilized", False)
            
            if context_utilized:
                return response
            else:
                return f"Based on your message: {response}"
        
        else:
            # Enhanced fallback response
            if isinstance(result, dict) and result:
                key_info = []
                for key, value in list(result.items())[:3]:
                    if isinstance(value, (str, int, float)):
                        key_info.append(f"{key}: {value}")
                
                if key_info:
                    return f"Task completed successfully. {', '.join(key_info)}"
            
            return f"Task completed successfully. Result: {str(result)[:200]}..."
    
    def _update_conversation_context(self, conversation: ConversationHistory, 
                                   agent_response: AgentResponse) -> None:
        """Update conversation context based on agent response."""
        # Extract relevant context from the response
        if agent_response.task_type == "classification":
            conversation.context["last_classification"] = agent_response.result.get("predicted_category")
        
        elif agent_response.task_type == "extraction":
            entities = agent_response.result.get("entities", [])
            conversation.context["extracted_entities"] = entities
        
        elif agent_response.task_type == "analysis":
            insights = agent_response.result.get("insights", [])
            conversation.context["analysis_insights"] = insights
        
        # Update general context
        conversation.context["last_task_type"] = agent_response.task_type
        conversation.context["last_confidence"] = agent_response.confidence
        conversation.context["total_interactions"] = len(conversation.messages)
    
    def _handle_conversation(self, request: AgentRequest, step_num: int, 
                           previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle general conversation task."""
        current_message = request.input_data.get("current_message", "")
        conversation_history = request.input_data.get("conversation_history", [])
        
        if step_num == 1:
            # Step 1: Analyze conversation context
            return {
                'action': 'analyze_context',
                'input': {'message': current_message, 'history_length': len(conversation_history)},
                'output': {
                    'message_type': 'question' if '?' in current_message else 'statement',
                    'context_available': len(conversation_history) > 0,
                    'analysis': 'Conversation context analyzed'
                },
                'confidence': 0.9,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Generate contextual response
            # Simple response generation based on message content
            if '?' in current_message:
                response = f"That's an interesting question about: {current_message[:50]}... Let me help you with that."
            elif any(word in current_message.lower() for word in ['hello', 'hi', 'hey']):
                response = "Hello! I'm here to help you. What can I assist you with today?"
            elif any(word in current_message.lower() for word in ['thank', 'thanks']):
                response = "You're welcome! Is there anything else I can help you with?"
            else:
                response = f"I understand you're mentioning: {current_message[:50]}... How can I help you further?"
            
            return {
                'action': 'generate_response',
                'input': {'message': current_message, 'history': conversation_history},
                'output': {
                    'response': response,
                    'response_type': 'contextual',
                    'uses_history': len(conversation_history) > 0
                },
                'confidence': 0.8,
                'complete': True
            }
    
    def _handle_chat(self, request: AgentRequest, step_num: int, 
                    previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle chat-specific task (similar to conversation but more casual)."""
        return self._handle_conversation(request, step_num, previous_steps)
    
    def _handle_follow_up(self, request: AgentRequest, step_num: int, 
                         previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle follow-up questions in conversation context."""
        current_message = request.input_data.get("current_message", "")
        conversation_history = request.input_data.get("conversation_history", [])
        
        if step_num == 1:
            # Step 1: Analyze follow-up context
            return {
                'action': 'analyze_follow_up',
                'input': {'message': current_message, 'history_length': len(conversation_history)},
                'output': {
                    'is_follow_up': True,
                    'references_previous': len(conversation_history) > 0,
                    'follow_up_type': self._classify_follow_up_type(current_message),
                    'analysis': 'Follow-up context analyzed'
                },
                'confidence': 0.9,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Generate contextual follow-up response
            follow_up_type = self._classify_follow_up_type(current_message)
            
            if follow_up_type == "clarification":
                response = f"To clarify regarding your previous question: {current_message}"
            elif follow_up_type == "expansion":
                response = f"Building on what we discussed: {current_message}"
            elif follow_up_type == "related":
                response = f"That's related to our conversation. {current_message}"
            else:
                response = f"Continuing our discussion: {current_message}"
            
            return {
                'action': 'generate_follow_up_response',
                'input': {'message': current_message, 'type': follow_up_type},
                'output': {
                    'response': response,
                    'follow_up_type': follow_up_type,
                    'context_preserved': True
                },
                'confidence': 0.85,
                'complete': True
            }
    
    def _handle_clarification(self, request: AgentRequest, step_num: int, 
                            previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle clarification requests."""
        current_message = request.input_data.get("current_message", "")
        
        if step_num == 1:
            # Step 1: Identify what needs clarification
            return {
                'action': 'identify_clarification_need',
                'input': {'message': current_message},
                'output': {
                    'clarification_type': 'general',
                    'ambiguous_terms': self._extract_ambiguous_terms(current_message),
                    'analysis': 'Clarification needs identified'
                },
                'confidence': 0.8,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Provide clarification
            clarification_response = random.choice(self.response_templates["clarification"])
            
            return {
                'action': 'provide_clarification',
                'input': {'message': current_message},
                'output': {
                    'response': f"{clarification_response} Specifically about: {current_message[:100]}...",
                    'clarification_provided': True
                },
                'confidence': 0.8,
                'complete': True
            }
    
    def _handle_context_aware(self, request: AgentRequest, step_num: int, 
                            previous_steps: List[AgentStep]) -> Dict[str, Any]:
        """Handle context-aware responses using conversation history."""
        current_message = request.input_data.get("current_message", "")
        conversation_history = request.input_data.get("conversation_history", [])
        conversation_context = request.input_data.get("conversation_context", {})
        
        if step_num == 1:
            # Step 1: Analyze conversation context
            context_score = self._calculate_context_relevance(current_message, conversation_history)
            
            return {
                'action': 'analyze_conversation_context',
                'input': {'message': current_message, 'history': conversation_history},
                'output': {
                    'context_relevance_score': context_score,
                    'previous_topics': self._extract_topics_from_history(conversation_history),
                    'context_available': len(conversation_history) > 0,
                    'analysis': 'Conversation context analyzed'
                },
                'confidence': 0.9,
                'complete': False
            }
        elif step_num == 2:
            # Step 2: Generate context-aware response
            context_score = self._calculate_context_relevance(current_message, conversation_history)
            previous_topics = self._extract_topics_from_history(conversation_history)
            
            if context_score > 0.7 and previous_topics:
                response = f"Based on our previous discussion about {', '.join(previous_topics[:2])}, {current_message}"
            elif context_score > 0.4:
                response = f"Considering what we've talked about, {current_message}"
            else:
                response = f"I understand you're asking about: {current_message}"
            
            return {
                'action': 'generate_context_aware_response',
                'input': {'message': current_message, 'context_score': context_score},
                'output': {
                    'response': response,
                    'context_utilized': context_score > 0.4,
                    'context_score': context_score
                },
                'confidence': min(0.9, 0.6 + context_score * 0.3),
                'complete': True
            }
    
    def _classify_follow_up_type(self, message: str) -> str:
        """Classify the type of follow-up question."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return "clarification"
        elif any(word in message_lower for word in ['more', 'also', 'additionally', 'furthermore']):
            return "expansion"
        elif any(word in message_lower for word in ['related', 'similar', 'like', 'about']):
            return "related"
        else:
            return "general"
    
    def _extract_ambiguous_terms(self, message: str) -> List[str]:
        """Extract potentially ambiguous terms from message."""
        # Simple implementation - in production, use NLP libraries
        ambiguous_indicators = ['it', 'this', 'that', 'they', 'them', 'thing', 'stuff']
        words = message.lower().split()
        return [word for word in words if word in ambiguous_indicators]
    
    def _calculate_context_relevance(self, current_message: str, history: List[Dict[str, Any]]) -> float:
        """Calculate how relevant the current message is to conversation history."""
        if not history:
            return 0.0
        
        current_words = set(current_message.lower().split())
        
        # Get words from recent history
        recent_words = set()
        for msg in history[-3:]:  # Last 3 messages
            content = msg.get('content', '')
            recent_words.update(content.lower().split())
        
        if not recent_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(current_words.intersection(recent_words))
        total_unique = len(current_words.union(recent_words))
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def _extract_topics_from_history(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from conversation history."""
        if not history:
            return []
        
        # Simple topic extraction - in production, use proper NLP
        all_words = []
        for msg in history:
            content = msg.get('content', '')
            words = content.lower().split()
            all_words.extend(words)
        
        # Find most common meaningful words (simple approach)
        word_counts = {}
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        for word in all_words:
            if len(word) > 3 and word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top topics
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5] if count > 1]
    
    def get_conversation_history(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get conversation history by ID."""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self, user_id: Optional[str] = None, 
                          project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List conversations with optional filtering."""
        conversations = []
        
        for conv_id, conv in self.conversations.items():
            # Apply filters
            if user_id and conv.user_id != user_id:
                continue
            if project_id and conv.project_id != project_id:
                continue
            
            conversations.append({
                "conversation_id": conv_id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "project_id": conv.project_id,
                "user_id": conv.user_id,
                "last_message": conv.messages[-1].content[:100] if conv.messages else ""
            })
        
        # Sort by most recent first
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a specific conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    
    def clear_old_conversations(self, max_age_hours: int = 24) -> int:
        """Clear conversations older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleared_count = 0
        
        conversations_to_remove = []
        for conv_id, conv in self.conversations.items():
            if conv.updated_at < cutoff_time:
                conversations_to_remove.append(conv_id)
        
        for conv_id in conversations_to_remove:
            del self.conversations[conv_id]
            cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} old conversations")
        return cleared_count
    
    
    def _update_metrics(self, response: AgentResponse, success: bool) -> None:
        """Update service metrics with enhanced tracking."""
        self.metrics["task_count"] += 1
        
        if success:
            self.metrics["successful_tasks"] += 1
        
        self.metrics["total_execution_time"] += response.execution_time
        self.metrics["total_steps"] += response.total_steps
        self.metrics["total_confidence"] += response.confidence
        
        # Update enhanced metrics
        if response.confidence > self.response_quality_threshold:
            self.metrics["response_quality_scores"].append(response.confidence)
        
        # Update conversation analytics if this was a multi-turn task
        if hasattr(response, 'metadata') and response.metadata:
            conversation_id = response.metadata.get('conversation_id')
            if conversation_id:
                self._update_conversation_analytics(
                    conversation_id, 
                    "message_added",
                    {
                        "response_time": response.execution_time,
                        "confidence": response.confidence,
                        "topics": self._extract_topics_from_response(response)
                    }
                )
    
    def _extract_topics_from_response(self, response: AgentResponse) -> List[str]:
        """Extract topics from agent response for analytics."""
        # Simple topic extraction from response result
        topics = []
        
        if response.task_type == "classification":
            category = response.result.get("predicted_category")
            if category:
                topics.append(category)
        
        elif response.task_type == "extraction":
            entities = response.result.get("entities", [])
            topics.extend([entity.get("label", "") for entity in entities[:3]])
        
        elif response.task_type == "analysis":
            insights = response.result.get("insights", [])
            topics.extend([insight[:20] for insight in insights[:2]])  # First 20 chars
        
        return [topic for topic in topics if topic]
    
    def get_metrics(self) -> AgentMetrics:
        """Get enhanced Agent service metrics."""
        task_count = self.metrics["task_count"]
        
        if task_count == 0:
            return AgentMetrics(
                task_count=0,
                success_rate=0.0,
                avg_execution_time=0.0,
                avg_steps=0.0,
                avg_confidence=0.0,
                conversation_count=self.metrics["conversation_count"],
                multi_turn_sessions=self.metrics["multi_turn_sessions"],
                avg_conversation_length=0.0,
                avg_response_quality=0.0,
                context_utilization_rate=0.0
            )
        
        # Calculate enhanced metrics
        avg_conversation_length = 0.0
        if self.metrics["conversation_lengths"]:
            avg_conversation_length = sum(self.metrics["conversation_lengths"]) / len(self.metrics["conversation_lengths"])
        
        avg_response_quality = 0.0
        if self.metrics["response_quality_scores"]:
            avg_response_quality = sum(self.metrics["response_quality_scores"]) / len(self.metrics["response_quality_scores"])
        
        context_utilization_rate = 0.0
        if self.metrics["context_utilization_scores"]:
            context_utilization_rate = sum(self.metrics["context_utilization_scores"]) / len(self.metrics["context_utilization_scores"])
        
        return AgentMetrics(
            task_count=task_count,
            success_rate=(self.metrics["successful_tasks"] / task_count) * 100,
            avg_execution_time=self.metrics["total_execution_time"] / task_count,
            avg_steps=self.metrics["total_steps"] / task_count,
            avg_confidence=self.metrics["total_confidence"] / task_count,
            conversation_count=self.metrics["conversation_count"],
            multi_turn_sessions=self.metrics["multi_turn_sessions"],
            avg_conversation_length=avg_conversation_length,
            avg_response_quality=avg_response_quality,
            context_utilization_rate=context_utilization_rate * 100
        )
    
    def reset_metrics(self) -> None:
        """Reset enhanced service metrics."""
        self.metrics = {
            "task_count": 0,
            "successful_tasks": 0,
            "total_execution_time": 0.0,
            "total_steps": 0,
            "total_confidence": 0.0,
            "conversation_count": 0,
            "multi_turn_sessions": 0,
            "response_quality_scores": [],
            "context_utilization_scores": [],
            "conversation_lengths": [],
            "avg_response_time": 0.0,
            "conversation_satisfaction_scores": []
        }
        
        # Clear conversation analytics but keep conversations
        self.conversation_analytics.clear()
        
        logger.info("Agent service metrics reset with enhanced tracking")
    
    def get_supported_tasks(self) -> List[Dict[str, Any]]:
        """Get list of supported task types with enhanced capabilities."""
        return [
            {
                "task_type": "classification",
                "description": "Classify text into predefined categories with confidence scoring",
                "required_inputs": ["text", "categories"],
                "optional_inputs": ["threshold", "multi_label"],
                "supports_multi_turn": True,
                "enhanced_features": ["confidence_scoring", "multi_category_support"]
            },
            {
                "task_type": "extraction",
                "description": "Extract named entities from text with entity grouping",
                "required_inputs": ["text"],
                "optional_inputs": ["entities", "confidence_threshold", "group_by_type"],
                "supports_multi_turn": True,
                "enhanced_features": ["entity_grouping", "confidence_filtering"]
            },
            {
                "task_type": "summarization",
                "description": "Generate text summaries with compression metrics",
                "required_inputs": ["text"],
                "optional_inputs": ["max_length", "style", "compression_target"],
                "supports_multi_turn": True,
                "enhanced_features": ["compression_metrics", "style_adaptation"]
            },
            {
                "task_type": "question_answering",
                "description": "Answer questions based on context with confidence scoring",
                "required_inputs": ["question", "context"],
                "optional_inputs": ["max_answer_length", "confidence_threshold"],
                "supports_multi_turn": True,
                "enhanced_features": ["confidence_scoring", "context_highlighting"]
            },
            {
                "task_type": "text_generation",
                "description": "Generate text from prompts with quality metrics",
                "required_inputs": ["prompt"],
                "optional_inputs": ["max_tokens", "temperature", "quality_threshold"],
                "supports_multi_turn": True,
                "enhanced_features": ["quality_metrics", "token_counting"]
            },
            {
                "task_type": "analysis",
                "description": "Analyze data and provide insights with recommendations",
                "required_inputs": ["data"],
                "optional_inputs": ["analysis_type", "metrics", "recommendation_count"],
                "supports_multi_turn": True,
                "enhanced_features": ["insight_generation", "recommendation_engine"]
            },
            {
                "task_type": "conversation",
                "description": "General conversation and dialogue with context awareness",
                "required_inputs": ["current_message"],
                "optional_inputs": ["conversation_history", "context", "personality"],
                "supports_multi_turn": True,
                "enhanced_features": ["context_awareness", "personality_adaptation"]
            },
            {
                "task_type": "chat",
                "description": "Casual chat and interaction with engagement tracking",
                "required_inputs": ["current_message"],
                "optional_inputs": ["conversation_history", "context", "engagement_level"],
                "supports_multi_turn": True,
                "enhanced_features": ["engagement_tracking", "casual_tone"]
            },
            {
                "task_type": "follow_up",
                "description": "Handle follow-up questions with context preservation",
                "required_inputs": ["current_message", "conversation_history"],
                "optional_inputs": ["follow_up_type", "context_window"],
                "supports_multi_turn": True,
                "enhanced_features": ["context_preservation", "follow_up_classification"]
            },
            {
                "task_type": "clarification",
                "description": "Provide clarifications and handle ambiguous requests",
                "required_inputs": ["current_message"],
                "optional_inputs": ["ambiguity_threshold", "clarification_style"],
                "supports_multi_turn": True,
                "enhanced_features": ["ambiguity_detection", "clarification_templates"]
            },
            {
                "task_type": "context_aware",
                "description": "Context-aware responses using conversation history",
                "required_inputs": ["current_message", "conversation_history"],
                "optional_inputs": ["context_window", "relevance_threshold"],
                "supports_multi_turn": True,
                "enhanced_features": ["context_scoring", "topic_tracking"]
            }
        ]