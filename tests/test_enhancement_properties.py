"""
Property-based tests for data enhancement functionality.

Tests the data enhancement operations to ensure they improve overall data quality
as specified in Requirement 5.2 of the SuperInsight Platform requirements.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis import HealthCheck
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
import json
import asyncio

from src.enhancement.service import DataEnhancementService
from src.enhancement.models import EnhancementConfig, EnhancementResult, EnhancementType, QualitySample
from src.models.document import Document
from src.models.task import Task, TaskStatus


# Hypothesis strategies for generating test data

def document_strategy():
    """Strategy for generating valid Document instances."""
    return st.builds(
        Document,
        id=st.just(uuid4()),
        source_type=st.sampled_from(["database", "file", "api"]),
        source_config=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=50),
            min_size=1, max_size=3
        ),
        content=st.text(min_size=10, max_size=1000),
        metadata=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats()),
            min_size=0, max_size=5
        ),
        created_at=st.just(datetime.now()),
        updated_at=st.just(datetime.now())
    )


def task_strategy():
    """Strategy for generating valid Task instances."""
    return st.builds(
        Task,
        id=st.just(uuid4()),
        document_id=st.just(uuid4()),
        project_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        status=st.sampled_from([TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]),
        annotations=st.lists(
            st.dictionaries(
                st.sampled_from(['id', 'result', 'annotator_id', 'confidence']),
                st.one_of(st.text(), st.integers(), st.floats(min_value=0.0, max_value=1.0))
            ),
            min_size=0, max_size=3
        ),
        ai_predictions=st.lists(st.dictionaries(st.text(), st.text()), min_size=0, max_size=2),
        quality_score=st.floats(min_value=0.0, max_value=1.0),
        created_at=st.just(datetime.now())
    )


def enhancement_config_strategy():
    """Strategy for generating valid EnhancementConfig instances."""
    return st.builds(
        EnhancementConfig,
        enhancement_type=st.sampled_from(list(EnhancementType)),
        target_quality_threshold=st.floats(min_value=0.1, max_value=0.9),
        amplification_factor=st.floats(min_value=1.1, max_value=5.0),
        batch_size=st.integers(min_value=10, max_value=500),
        preserve_original=st.booleans()
    )


def quality_sample_strategy():
    """Strategy for generating valid QualitySample instances."""
    return st.builds(
        QualitySample,
        id=st.just(uuid4()),
        content=st.text(min_size=10, max_size=500),
        quality_score=st.floats(min_value=0.7, max_value=1.0),  # Generate high-quality samples
        metadata=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=50),
            min_size=0, max_size=3
        ),
        created_at=st.just(datetime.now())
    )


class TestDataEnhancementPositivity:
    """
    Property-based tests for data enhancement positivity.
    
    Validates Requirement 5.2:
    - When data enhancement is performed, it should improve overall data quality scores
    - Enhancement operations should have positive impact on data quality metrics
    """
    
    @given(st.lists(document_strategy(), min_size=1, max_size=10), enhancement_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
    def test_quality_sample_enhancement_improves_quality(self, documents: List[Document], config: EnhancementConfig):
        """
        Property 9: Data Enhancement Positivity - Quality Sample Fill
        
        For any list of documents and enhancement configuration, performing
        quality sample enhancement should improve the overall data quality score.
        
        **Validates: Requirement 5.2**
        """
        # Ensure we're testing quality sample enhancement
        config.enhancement_type = EnhancementType.QUALITY_SAMPLE_FILL
        
        # Ensure documents have unique IDs and reasonable content
        for i, doc in enumerate(documents):
            doc.id = uuid4()
            if not doc.content.strip():
                doc.content = f"Sample document content {i}"
        
        # Create enhancement service
        enhancement_service = DataEnhancementService()
        
        # Calculate original quality scores
        original_scores = [enhancement_service._calculate_document_quality(doc) for doc in documents]
        original_avg_quality = sum(original_scores) / len(original_scores)
        
        # Perform enhancement
        async def perform_enhancement():
            result = await enhancement_service.enhance_with_quality_samples(documents, config)
            return result
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            enhancement_result = loop.run_until_complete(perform_enhancement())
            
            # Property: Enhancement should improve quality (positive quality improvement)
            assert enhancement_result.quality_improvement >= 0.0, \
                f"Quality sample enhancement should improve quality, got improvement: {enhancement_result.quality_improvement}"
            
            # Verify enhancement metadata contains quality information
            assert "original_avg_quality" in enhancement_result.metadata
            assert "new_avg_quality" in enhancement_result.metadata
            
            original_quality = enhancement_result.metadata["original_avg_quality"]
            new_quality = enhancement_result.metadata["new_avg_quality"]
            
            # Property: New quality should be >= original quality
            assert new_quality >= original_quality, \
                f"New quality {new_quality} should be >= original quality {original_quality}"
            
            # Property: Quality improvement should match the difference
            expected_improvement = new_quality - original_quality
            assert abs(enhancement_result.quality_improvement - expected_improvement) < 1e-10, \
                f"Quality improvement {enhancement_result.quality_improvement} should match difference {expected_improvement}"
            
            # Verify enhancement processed documents
            assert enhancement_result.original_count == len(documents)
            assert enhancement_result.enhanced_count >= 0
            
        finally:
            loop.close()
    
    @given(st.lists(task_strategy(), min_size=1, max_size=10), enhancement_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
    def test_positive_amplification_improves_quality(self, tasks: List[Task], config: EnhancementConfig):
        """
        Property 9: Data Enhancement Positivity - Positive Amplification
        
        For any list of tasks and enhancement configuration, performing
        positive data amplification should improve the overall quality metrics.
        
        **Validates: Requirement 5.2**
        """
        # Ensure we're testing positive amplification
        config.enhancement_type = EnhancementType.POSITIVE_AMPLIFICATION
        
        # Ensure tasks have unique IDs and valid quality scores
        for i, task in enumerate(tasks):
            task.id = uuid4()
            task.project_id = f"project_{i}"
            # Ensure some tasks have high quality scores (positive samples)
            if i < len(tasks) // 2:
                task.quality_score = max(config.target_quality_threshold, 0.8)
        
        # Create enhancement service
        enhancement_service = DataEnhancementService()
        
        # Calculate original metrics
        original_avg_quality = sum(task.quality_score for task in tasks) / len(tasks)
        original_positive_count = sum(1 for task in tasks if task.quality_score >= config.target_quality_threshold)
        
        # Perform amplification
        async def perform_amplification():
            result = await enhancement_service.amplify_positive_data(tasks, config)
            return result
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            enhancement_result = loop.run_until_complete(perform_amplification())
            
            # Property: Amplification should have positive or neutral impact on quality
            assert enhancement_result.quality_improvement >= 0.0, \
                f"Positive amplification should improve quality, got improvement: {enhancement_result.quality_improvement}"
            
            # Verify amplification metadata
            assert "positive_count" in enhancement_result.metadata
            assert "amplified_count" in enhancement_result.metadata
            
            positive_count = enhancement_result.metadata["positive_count"]
            amplified_count = enhancement_result.metadata["amplified_count"]
            
            # Property: Positive count should match our calculation
            assert positive_count == original_positive_count, \
                f"Positive count {positive_count} should match original {original_positive_count}"
            
            # Property: Enhanced count should be >= original count (amplification adds samples)
            assert enhancement_result.enhanced_count >= enhancement_result.original_count, \
                f"Enhanced count {enhancement_result.enhanced_count} should be >= original count {enhancement_result.original_count}"
            
            # Property: If amplification occurred, enhanced count should be higher
            if amplified_count > 0:
                assert enhancement_result.enhanced_count > enhancement_result.original_count, \
                    f"With {amplified_count} amplified samples, enhanced count should exceed original count"
            
        finally:
            loop.close()
    
    @given(st.lists(document_strategy(), min_size=1, max_size=20), enhancement_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=50)
    def test_batch_enhancement_improves_quality(self, documents: List[Document], config: EnhancementConfig):
        """
        Property 9: Data Enhancement Positivity - Batch Enhancement
        
        For any batch of documents and enhancement configuration, performing
        batch enhancement should improve the overall data quality.
        
        **Validates: Requirement 5.2**
        """
        # Ensure we're testing batch enhancement
        config.enhancement_type = EnhancementType.BATCH_ENHANCEMENT
        
        # Ensure documents have unique IDs and reasonable content
        for i, doc in enumerate(documents):
            doc.id = uuid4()
            if not doc.content.strip():
                doc.content = f"Batch document content {i}"
        
        # Create enhancement service
        enhancement_service = DataEnhancementService()
        
        # Calculate original quality
        original_scores = [enhancement_service._calculate_document_quality(doc) for doc in documents]
        original_avg_quality = sum(original_scores) / len(original_scores)
        
        # Perform batch enhancement
        async def perform_batch_enhancement():
            result = await enhancement_service.batch_enhance_data(documents, config)
            return result
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            enhancement_result = loop.run_until_complete(perform_batch_enhancement())
            
            # Property: Batch enhancement should improve quality
            assert enhancement_result.quality_improvement >= 0.0, \
                f"Batch enhancement should improve quality, got improvement: {enhancement_result.quality_improvement}"
            
            # Verify batch processing metadata
            assert "batch_count" in enhancement_result.metadata
            assert "batch_size" in enhancement_result.metadata
            
            batch_count = enhancement_result.metadata["batch_count"]
            batch_size = enhancement_result.metadata["batch_size"]
            
            # Property: Batch count should be reasonable for document count
            expected_batches = (len(documents) + batch_size - 1) // batch_size  # Ceiling division
            assert batch_count == expected_batches, \
                f"Batch count {batch_count} should match expected {expected_batches}"
            
            # Property: Enhanced count should match original count (batch processing preserves count)
            assert enhancement_result.enhanced_count == enhancement_result.original_count, \
                f"Batch enhancement should preserve document count: {enhancement_result.enhanced_count} vs {enhancement_result.original_count}"
            
        finally:
            loop.close()
    
    @given(st.lists(task_strategy(), min_size=1, max_size=10), enhancement_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
    def test_quality_score_update_improves_scores(self, tasks: List[Task], config: EnhancementConfig):
        """
        Property 9: Data Enhancement Positivity - Quality Score Updates
        
        For any list of tasks and positive enhancement result, updating
        quality scores should improve the individual task quality scores.
        
        **Validates: Requirement 5.2**
        """
        # Ensure tasks have unique IDs and reasonable initial scores
        for i, task in enumerate(tasks):
            task.id = uuid4()
            task.project_id = f"project_{i}"
            # Ensure initial scores are not at maximum (room for improvement)
            task.quality_score = min(0.9, task.quality_score)
        
        # Create enhancement service
        enhancement_service = DataEnhancementService()
        
        # Store original quality scores
        original_scores = [task.quality_score for task in tasks]
        original_avg = sum(original_scores) / len(original_scores)
        
        # Create a positive enhancement result (simulating successful enhancement)
        enhancement_result = EnhancementResult(
            enhancement_type=config.enhancement_type,
            original_count=len(tasks),
            enhanced_count=len(tasks),
            quality_improvement=0.1,  # Positive improvement
            processing_time=1.0,
            metadata={"test": "quality_update"}
        )
        
        # Update quality scores
        async def update_scores():
            updated_tasks = await enhancement_service.update_quality_scores(tasks, enhancement_result)
            return updated_tasks
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            updated_tasks = loop.run_until_complete(update_scores())
            
            # Property: Updated tasks should have improved quality scores
            updated_scores = [task.quality_score for task in updated_tasks]
            updated_avg = sum(updated_scores) / len(updated_scores)
            
            # Property: Average quality should improve
            assert updated_avg >= original_avg, \
                f"Updated average quality {updated_avg} should be >= original {original_avg}"
            
            # Property: Individual scores should improve or stay the same
            for i, (original_score, updated_score) in enumerate(zip(original_scores, updated_scores)):
                assert updated_score >= original_score, \
                    f"Task {i} updated score {updated_score} should be >= original score {original_score}"
                
                # Property: Updated scores should not exceed 1.0
                assert updated_score <= 1.0, \
                    f"Task {i} updated score {updated_score} should not exceed 1.0"
            
            # Property: Number of tasks should be preserved
            assert len(updated_tasks) == len(tasks), \
                f"Number of tasks should be preserved: {len(updated_tasks)} vs {len(tasks)}"
            
        finally:
            loop.close()
    
    @given(st.lists(document_strategy(), min_size=2, max_size=15))
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=50)
    def test_enhancement_consistency_across_operations(self, documents: List[Document]):
        """
        Property 9: Data Enhancement Positivity - Consistency Across Operations
        
        For any set of documents, different enhancement operations should
        consistently improve quality in a predictable manner.
        
        **Validates: Requirement 5.2**
        """
        # Ensure documents have unique IDs and content
        for i, doc in enumerate(documents):
            doc.id = uuid4()
            if not doc.content.strip():
                doc.content = f"Consistency test document {i}"
        
        # Create enhancement service
        enhancement_service = DataEnhancementService()
        
        # Test different enhancement configurations
        configs = [
            EnhancementConfig(
                enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
                target_quality_threshold=0.7,
                batch_size=50
            ),
            EnhancementConfig(
                enhancement_type=EnhancementType.BATCH_ENHANCEMENT,
                target_quality_threshold=0.6,
                batch_size=100
            )
        ]
        
        # Test each configuration
        async def test_multiple_enhancements():
            results = []
            
            for config in configs:
                # Create fresh document copies for each test
                doc_copies = []
                for doc in documents:
                    doc_copy = Document(
                        source_type=doc.source_type,
                        source_config=doc.source_config.copy(),
                        content=doc.content,
                        metadata=doc.metadata.copy() if doc.metadata else {}
                    )
                    doc_copies.append(doc_copy)
                
                # Perform enhancement
                if config.enhancement_type == EnhancementType.QUALITY_SAMPLE_FILL:
                    result = await enhancement_service.enhance_with_quality_samples(doc_copies, config)
                elif config.enhancement_type == EnhancementType.BATCH_ENHANCEMENT:
                    result = await enhancement_service.batch_enhance_data(doc_copies, config)
                
                results.append(result)
            
            return results
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            enhancement_results = loop.run_until_complete(test_multiple_enhancements())
            
            # Property: All enhancement operations should improve quality
            for i, result in enumerate(enhancement_results):
                assert result.quality_improvement >= 0.0, \
                    f"Enhancement operation {i} should improve quality, got: {result.quality_improvement}"
                
                # Property: Processing should complete successfully
                assert result.processing_time >= 0.0, \
                    f"Enhancement operation {i} should have non-negative processing time"
                
                # Property: Document counts should be reasonable
                assert result.original_count == len(documents), \
                    f"Enhancement operation {i} should process all documents"
                assert result.enhanced_count >= 0, \
                    f"Enhancement operation {i} should produce non-negative enhanced count"
            
            # Property: Quality improvements should be consistent (all positive or zero)
            improvements = [result.quality_improvement for result in enhancement_results]
            assert all(improvement >= 0.0 for improvement in improvements), \
                f"All quality improvements should be non-negative: {improvements}"
            
        finally:
            loop.close()
    
    @given(st.lists(quality_sample_strategy(), min_size=1, max_size=5), document_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
    def test_quality_sample_merging_improves_document(self, quality_samples: List[QualitySample], document: Document):
        """
        Property 9: Data Enhancement Positivity - Quality Sample Merging
        
        For any document and list of quality samples, merging with quality
        samples should result in a document with improved quality characteristics.
        
        **Validates: Requirement 5.2**
        """
        # Ensure document has reasonable content
        if not document.content.strip():
            document.content = "Original document content for quality testing"
        
        # Ensure quality samples have higher quality scores
        for i, sample in enumerate(quality_samples):
            sample.quality_score = max(0.7, sample.quality_score)  # Ensure high quality
            if not sample.content.strip():
                sample.content = f"High quality sample content {i}"
        
        # Create enhancement service
        enhancement_service = DataEnhancementService()
        
        # Calculate original document quality
        original_quality = enhancement_service._calculate_document_quality(document)
        
        # Merge with quality samples
        enhanced_document = enhancement_service._merge_with_quality_samples(document, quality_samples)
        
        # Calculate enhanced document quality
        enhanced_quality = enhancement_service._calculate_document_quality(enhanced_document)
        
        # Property: Enhanced document should have quality >= original quality (with tolerance for floating point precision)
        assert enhanced_quality >= original_quality - 1e-10, \
            f"Enhanced document quality {enhanced_quality} should be >= original quality {original_quality}"
        
        # Property: Enhanced document should have enhancement metadata
        assert enhanced_document.metadata.get("enhanced") is True, \
            "Enhanced document should be marked as enhanced"
        
        assert "enhancement_quality" in enhanced_document.metadata, \
            "Enhanced document should contain enhancement quality metadata"
        
        enhancement_quality = enhanced_document.metadata["enhancement_quality"]
        assert enhancement_quality >= 0.0, \
            f"Enhancement quality {enhancement_quality} should be non-negative"
        
        # Property: Enhanced document should reference original
        assert "original_id" in enhanced_document.metadata, \
            "Enhanced document should reference original document ID"
        
        assert enhanced_document.metadata["original_id"] == str(document.id), \
            "Enhanced document should correctly reference original document ID"
        
        # Property: Enhanced document should preserve source information
        assert enhanced_document.source_type == document.source_type, \
            "Enhanced document should preserve source type"
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_concurrent_enhancement_operations_maintain_positivity(self, num_concurrent: int):
        """
        Property 9: Data Enhancement Positivity - Concurrent Operations
        
        For any number of concurrent enhancement operations, each should
        maintain the property of improving data quality without interference.
        
        **Validates: Requirement 5.2**
        """
        # Create enhancement service
        enhancement_service = DataEnhancementService()
        
        # Create concurrent enhancement tasks
        async def perform_concurrent_enhancements():
            tasks = []
            
            for i in range(num_concurrent):
                # Create documents for this enhancement
                documents = [
                    Document(
                        source_type="api",
                        source_config={"source": f"concurrent_test_{i}"},
                        content=f"Concurrent document content {i}_{j}"
                    )
                    for j in range(3)  # 3 documents per enhancement
                ]
                
                # Create enhancement config
                config = EnhancementConfig(
                    enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
                    target_quality_threshold=0.6,
                    batch_size=10
                )
                
                # Create async task for enhancement
                async def enhance_batch(docs=documents, cfg=config, index=i):
                    # Simulate some processing delay
                    await asyncio.sleep(0.01)
                    
                    # Perform enhancement
                    result = await enhancement_service.enhance_with_quality_samples(docs, cfg)
                    return result, index
                
                tasks.append(enhance_batch())
            
            # Run all enhancements concurrently
            results = await asyncio.gather(*tasks)
            return results
        
        # Run the concurrent process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            concurrent_results = loop.run_until_complete(perform_concurrent_enhancements())
            
            # Property: All concurrent enhancements should improve quality
            for result, index in concurrent_results:
                assert result.quality_improvement >= 0.0, \
                    f"Concurrent enhancement {index} should improve quality, got: {result.quality_improvement}"
                
                # Property: Each enhancement should process its documents
                assert result.original_count == 3, \
                    f"Concurrent enhancement {index} should process 3 documents"
                
                # Property: Processing should complete successfully
                assert result.processing_time >= 0.0, \
                    f"Concurrent enhancement {index} should have non-negative processing time"
            
            # Property: All enhancements should complete (no interference)
            assert len(concurrent_results) == num_concurrent, \
                f"All {num_concurrent} concurrent enhancements should complete"
            
            # Property: Quality improvements should all be non-negative
            improvements = [result.quality_improvement for result, _ in concurrent_results]
            assert all(improvement >= 0.0 for improvement in improvements), \
                f"All concurrent quality improvements should be non-negative: {improvements}"
            
        finally:
            loop.close()


if __name__ == "__main__":
    # Run with verbose output and show hypothesis examples
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])