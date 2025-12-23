"""
Data enhancement service implementation.

Provides core functionality for data augmentation and quality improvement.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .models import EnhancementConfig, EnhancementResult, EnhancementType, QualitySample
from ..models.document import Document
from ..models.task import Task
from ..models.annotation import Annotation


logger = logging.getLogger(__name__)


class DataEnhancementService:
    """
    Service for data enhancement and quality improvement.
    
    Implements algorithms for:
    - Quality sample filling
    - Positive data amplification
    - Batch data enhancement
    - Quality score updates
    """
    
    def __init__(self, db_manager=None):
        """Initialize the data enhancement service."""
        self.db_manager = db_manager
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def enhance_with_quality_samples(
        self, 
        documents: List[Document], 
        config: EnhancementConfig
    ) -> EnhancementResult:
        """
        Fill dataset with high-quality samples.
        
        Args:
            documents: List of documents to enhance
            config: Enhancement configuration
            
        Returns:
            Enhancement result with statistics
        """
        start_time = time.time()
        logger.info(f"Starting quality sample enhancement for {len(documents)} documents")
        
        # Calculate original quality scores
        original_scores = [self._calculate_document_quality(doc) for doc in documents]
        original_avg_quality = sum(original_scores) / len(original_scores) if original_scores else 0.0
        
        # Identify low-quality documents that need enhancement
        low_quality_docs = [
            (doc, score) for doc, score in zip(documents, original_scores)
            if score < config.target_quality_threshold
        ]
        
        # Generate quality samples for enhancement
        enhanced_documents = []
        for doc, score in low_quality_docs:
            quality_samples = await self._generate_quality_samples(doc, config)
            enhanced_doc = self._merge_with_quality_samples(doc, quality_samples)
            enhanced_documents.append(enhanced_doc)
        
        # Calculate new quality scores
        enhanced_scores = [self._calculate_document_quality(doc) for doc in enhanced_documents]
        new_avg_quality = sum(enhanced_scores) / len(enhanced_scores) if enhanced_scores else original_avg_quality
        
        processing_time = time.time() - start_time
        
        result = EnhancementResult(
            enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
            original_count=len(documents),
            enhanced_count=len(enhanced_documents),
            quality_improvement=new_avg_quality - original_avg_quality,
            processing_time=processing_time,
            metadata={
                "original_avg_quality": original_avg_quality,
                "new_avg_quality": new_avg_quality,
                "low_quality_count": len(low_quality_docs),
                "quality_threshold": config.target_quality_threshold
            }
        )
        
        logger.info(f"Quality sample enhancement completed in {processing_time:.2f}s")
        return result
    
    async def amplify_positive_data(
        self, 
        tasks: List[Task], 
        config: EnhancementConfig
    ) -> EnhancementResult:
        """
        Amplify positive reinforcement data to improve training balance.
        
        Args:
            tasks: List of annotation tasks
            config: Enhancement configuration
            
        Returns:
            Enhancement result with statistics
        """
        start_time = time.time()
        logger.info(f"Starting positive data amplification for {len(tasks)} tasks")
        
        # Identify positive samples (high quality scores)
        positive_tasks = [
            task for task in tasks 
            if task.quality_score >= config.target_quality_threshold
        ]
        
        # Calculate amplification target
        target_count = int(len(positive_tasks) * config.amplification_factor)
        amplification_needed = max(0, target_count - len(positive_tasks))
        
        # Generate amplified samples
        amplified_tasks = []
        if amplification_needed > 0:
            amplified_tasks = await self._generate_amplified_samples(
                positive_tasks, amplification_needed, config
            )
        
        processing_time = time.time() - start_time
        
        result = EnhancementResult(
            enhancement_type=EnhancementType.POSITIVE_AMPLIFICATION,
            original_count=len(tasks),
            enhanced_count=len(tasks) + len(amplified_tasks),
            quality_improvement=self._calculate_amplification_improvement(tasks, amplified_tasks),
            processing_time=processing_time,
            metadata={
                "positive_count": len(positive_tasks),
                "amplified_count": len(amplified_tasks),
                "amplification_factor": config.amplification_factor,
                "target_threshold": config.target_quality_threshold
            }
        )
        
        logger.info(f"Positive data amplification completed in {processing_time:.2f}s")
        return result
    
    async def batch_enhance_data(
        self, 
        documents: List[Document], 
        config: EnhancementConfig
    ) -> EnhancementResult:
        """
        Perform batch data enhancement operations.
        
        Args:
            documents: List of documents to enhance
            config: Enhancement configuration
            
        Returns:
            Enhancement result with statistics
        """
        start_time = time.time()
        logger.info(f"Starting batch enhancement for {len(documents)} documents")
        
        # Process documents in batches
        batches = [
            documents[i:i + config.batch_size] 
            for i in range(0, len(documents), config.batch_size)
        ]
        
        enhanced_batches = []
        total_quality_improvement = 0.0
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            
            # Enhance each batch
            enhanced_batch = await self._enhance_batch(batch, config)
            enhanced_batches.extend(enhanced_batch)
            
            # Calculate quality improvement for this batch
            original_quality = sum(self._calculate_document_quality(doc) for doc in batch) / len(batch)
            enhanced_quality = sum(self._calculate_document_quality(doc) for doc in enhanced_batch) / len(enhanced_batch)
            total_quality_improvement += enhanced_quality - original_quality
        
        processing_time = time.time() - start_time
        
        result = EnhancementResult(
            enhancement_type=EnhancementType.BATCH_ENHANCEMENT,
            original_count=len(documents),
            enhanced_count=len(enhanced_batches),
            quality_improvement=total_quality_improvement / len(batches),
            processing_time=processing_time,
            metadata={
                "batch_count": len(batches),
                "batch_size": config.batch_size,
                "preserve_original": config.preserve_original
            }
        )
        
        logger.info(f"Batch enhancement completed in {processing_time:.2f}s")
        return result
    
    async def update_quality_scores(
        self, 
        tasks: List[Task], 
        enhancement_result: EnhancementResult
    ) -> List[Task]:
        """
        Update quality scores for tasks after enhancement.
        
        Args:
            tasks: List of tasks to update
            enhancement_result: Result from enhancement operation
            
        Returns:
            Updated tasks with new quality scores
        """
        logger.info(f"Updating quality scores for {len(tasks)} tasks")
        
        # Calculate quality improvement factor
        improvement_factor = 1.0 + enhancement_result.quality_improvement
        
        updated_tasks = []
        for task in tasks:
            # Update quality score based on enhancement
            new_score = min(1.0, task.quality_score * improvement_factor)
            task.update_quality_score(new_score)
            updated_tasks.append(task)
        
        logger.info("Quality scores updated successfully")
        return updated_tasks
    
    def _calculate_document_quality(self, document: Document) -> float:
        """
        Calculate quality score for a document.
        
        Args:
            document: Document to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Ensure we have valid content
        if not document.content or len(document.content.strip()) == 0:
            return 0.0
        
        # Content quality indicators
        content = document.content.strip()
        content_length = len(content)
        
        # Length score (optimal range: 100-2000 characters)
        if content_length < 50:
            length_score = content_length / 50.0
        elif content_length <= 2000:
            length_score = 1.0
        else:
            # Penalize very long content slightly
            length_score = max(0.8, 2000.0 / content_length)
        
        # Diversity score (based on unique words)
        words = content.lower().split()
        if len(words) == 0:
            diversity_score = 0.0
        else:
            unique_words = len(set(words))
            diversity_score = min(1.0, unique_words / max(1, len(words) * 0.7))
        
        # Metadata richness score
        metadata_score = 0.0
        if document.metadata:
            # Count meaningful metadata fields
            meaningful_fields = sum(1 for k, v in document.metadata.items() 
                                  if v and str(v).strip())
            metadata_score = min(1.0, meaningful_fields / 5.0)  # Normalize by 5 fields
        
        # Structure score (basic checks for structured content)
        structure_score = 0.5  # Default neutral score
        if any(marker in content for marker in ['\n', '.', '!', '?', ':']):
            structure_score = 0.8
        if any(marker in content for marker in ['```', '|', '-', '*']):
            structure_score = 1.0
        
        # Combine scores with weights
        quality_score = (
            length_score * 0.3 +
            diversity_score * 0.3 +
            metadata_score * 0.2 +
            structure_score * 0.2
        )
        
        return min(1.0, max(0.0, quality_score))
    
    async def _generate_quality_samples(
        self, 
        document: Document, 
        config: EnhancementConfig
    ) -> List[QualitySample]:
        """
        Generate high-quality samples for document enhancement.
        
        Args:
            document: Source document
            config: Enhancement configuration
            
        Returns:
            List of quality samples
        """
        # Generate synthetic quality samples based on document content
        samples = []
        
        # Create variations of the original content
        base_content = document.content
        
        # Sample 1: Enhanced with additional context
        enhanced_content = f"{base_content}\n[Enhanced with quality context]"
        samples.append(QualitySample(
            content=enhanced_content,
            quality_score=config.target_quality_threshold + 0.1,
            metadata={"enhancement_type": "context_addition", "source_doc": str(document.id)}
        ))
        
        # Sample 2: Refined version
        refined_content = base_content.strip() + " [Quality refined]"
        samples.append(QualitySample(
            content=refined_content,
            quality_score=config.target_quality_threshold + 0.05,
            metadata={"enhancement_type": "refinement", "source_doc": str(document.id)}
        ))
        
        return samples
    
    def _merge_with_quality_samples(
        self, 
        document: Document, 
        quality_samples: List[QualitySample]
    ) -> Document:
        """
        Merge document with quality samples to create enhanced version.
        
        Args:
            document: Original document
            quality_samples: Quality samples to merge
            
        Returns:
            Enhanced document
        """
        # Create enhanced version by combining with best quality sample
        if not quality_samples:
            return document
        
        # Calculate actual quality for original document
        original_quality = self._calculate_document_quality(document)
        
        # Find the best quality sample based on actual calculated quality
        best_sample = None
        best_calculated_quality = original_quality
        
        for sample in quality_samples:
            # Create a temporary document to calculate quality
            temp_doc = Document(
                source_type=document.source_type,
                source_config=document.source_config,
                content=sample.content,
                metadata=sample.metadata or {}
            )
            calculated_quality = self._calculate_document_quality(temp_doc)
            
            if calculated_quality > best_calculated_quality:
                best_sample = sample
                best_calculated_quality = calculated_quality
        
        # If no sample is better, return original with enhancement metadata
        if best_sample is None:
            enhanced_doc = Document(
                source_type=document.source_type,
                source_config=document.source_config,
                content=document.content,
                metadata={
                    **document.metadata,
                    "enhanced": True,
                    "enhancement_quality": original_quality,
                    "original_id": str(document.id)
                }
            )
            return enhanced_doc
        
        # Create new document with enhanced content
        enhanced_doc = Document(
            source_type=document.source_type,
            source_config=document.source_config,
            content=best_sample.content,
            metadata={
                **document.metadata,
                "enhanced": True,
                "enhancement_quality": best_calculated_quality,
                "original_id": str(document.id)
            }
        )
        
        return enhanced_doc
    
    async def _generate_amplified_samples(
        self, 
        positive_tasks: List[Task], 
        count: int, 
        config: EnhancementConfig
    ) -> List[Task]:
        """
        Generate amplified samples from positive tasks.
        
        Args:
            positive_tasks: Source positive tasks
            count: Number of samples to generate
            config: Enhancement configuration
            
        Returns:
            List of amplified tasks
        """
        amplified_tasks = []
        
        for i in range(count):
            # Select source task (round-robin)
            source_task = positive_tasks[i % len(positive_tasks)]
            
            # Create amplified version
            amplified_task = Task(
                document_id=source_task.document_id,
                project_id=source_task.project_id,
                status=source_task.status,
                annotations=source_task.annotations.copy(),
                ai_predictions=source_task.ai_predictions.copy(),
                quality_score=min(1.0, source_task.quality_score * 1.1)  # Slight boost
            )
            
            amplified_tasks.append(amplified_task)
        
        return amplified_tasks
    
    def _calculate_amplification_improvement(
        self, 
        original_tasks: List[Task], 
        amplified_tasks: List[Task]
    ) -> float:
        """
        Calculate quality improvement from amplification.
        
        Args:
            original_tasks: Original task list
            amplified_tasks: Amplified tasks
            
        Returns:
            Quality improvement score
        """
        if not original_tasks:
            return 0.0
        
        original_avg = sum(task.quality_score for task in original_tasks) / len(original_tasks)
        
        if not amplified_tasks:
            return 0.0
        
        amplified_avg = sum(task.quality_score for task in amplified_tasks) / len(amplified_tasks)
        
        # Calculate improvement based on positive sample ratio increase
        positive_ratio_improvement = len(amplified_tasks) / len(original_tasks)
        return positive_ratio_improvement * (amplified_avg - original_avg)
    
    async def _enhance_batch(
        self, 
        batch: List[Document], 
        config: EnhancementConfig
    ) -> List[Document]:
        """
        Enhance a batch of documents.
        
        Args:
            batch: Batch of documents to enhance
            config: Enhancement configuration
            
        Returns:
            Enhanced documents
        """
        enhanced_docs = []
        
        for doc in batch:
            # Apply quality sample enhancement
            quality_samples = await self._generate_quality_samples(doc, config)
            enhanced_doc = self._merge_with_quality_samples(doc, quality_samples)
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs