"""
Unit tests for data enhancement functionality.

Tests sample filling algorithms, positive data amplification logic, and batch processing
as specified in Requirements 5.1, 5.2, 5.5.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock

from src.enhancement.service import DataEnhancementService
from src.enhancement.models import (
    EnhancementConfig, 
    EnhancementResult, 
    EnhancementType, 
    QualitySample
)
from src.enhancement.reconstruction import (
    DataReconstructionService,
    ReconstructionConfig,
    ReconstructionType,
    ReconstructionRecord,
    ReconstructionResult
)
from src.models.document import Document
from src.models.task import Task


class TestDataEnhancementService:
    """Unit tests for DataEnhancementService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enhancement_service = DataEnhancementService()
    
    def test_enhancement_service_initialization(self):
        """Test DataEnhancementService initializes correctly."""
        assert self.enhancement_service.db_manager is None
        assert self.enhancement_service.executor is not None
    
    def test_calculate_document_quality(self):
        """Test document quality calculation algorithm."""
        # Test document with good content and metadata (1000+ chars for high content score)
        doc_good = Document(
            source_type="file",
            source_config={"file_path": "test.txt"},
            content="This is a comprehensive document with substantial content that provides good information. " * 15,  # ~1350 chars
            metadata={"author": "test", "category": "important", "tags": ["quality", "test"], "version": "1.0", "reviewed": True}
        )
        
        quality_good = self.enhancement_service._calculate_document_quality(doc_good)
        assert 0.0 <= quality_good <= 1.0
        assert quality_good > 0.8  # Should be high with 1000+ chars and rich metadata
        
        # Test document with minimal content and no metadata
        doc_poor = Document(
            source_type="file",
            source_config={"file_path": "test.txt"},
            content="Short",
            metadata={}
        )
        
        quality_poor = self.enhancement_service._calculate_document_quality(doc_poor)
        assert 0.0 <= quality_poor <= 1.0
        assert quality_poor < quality_good  # Should be lower than good document
        
        # Test document with minimal content (single character to avoid empty validation)
        doc_minimal = Document(
            source_type="file",
            source_config={"file_path": "test.txt"},
            content="x",
            metadata={}
        )
        
        quality_minimal = self.enhancement_service._calculate_document_quality(doc_minimal)
        assert quality_minimal < quality_poor  # Should be even lower
    
    @pytest.mark.asyncio
    async def test_generate_quality_samples_correctness(self):
        """
        Test quality sample generation algorithm correctness.
        
        Validates Requirement 5.1: THE SuperInsight_Platform SHALL 支持填充优质样本数据
        """
        # Create test document
        document = Document(
            source_type="file",
            source_config={"file_path": "test.txt"},
            content="Original document content for testing",
            metadata={"category": "test"}
        )
        
        # Create enhancement config
        config = EnhancementConfig(
            enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
            target_quality_threshold=0.8
        )
        
        # Generate quality samples
        samples = await self.enhancement_service._generate_quality_samples(document, config)
        
        # Verify samples are generated
        assert len(samples) > 0
        assert all(isinstance(sample, QualitySample) for sample in samples)
        
        # Verify sample quality scores meet threshold
        for sample in samples:
            assert sample.quality_score >= config.target_quality_threshold
            assert sample.content != document.content  # Should be enhanced
            assert len(sample.content) > 0
            assert "enhancement_type" in sample.metadata
            assert "source_doc" in sample.metadata
    
    @pytest.mark.asyncio
    async def test_merge_with_quality_samples_correctness(self):
        """
        Test merging document with quality samples.
        
        Validates Requirement 5.1: THE SuperInsight_Platform SHALL 支持填充优质样本数据
        """
        # Create original document
        original_doc = Document(
            source_type="file",
            source_config={"file_path": "test.txt"},
            content="Original content",
            metadata={"category": "test"}
        )
        
        # Create quality samples
        samples = [
            QualitySample(
                content="Enhanced content version 1",
                quality_score=0.85,
                metadata={"enhancement_type": "context_addition"}
            ),
            QualitySample(
                content="Enhanced content version 2",
                quality_score=0.90,
                metadata={"enhancement_type": "refinement"}
            )
        ]
        
        # Merge with quality samples
        enhanced_doc = self.enhancement_service._merge_with_quality_samples(original_doc, samples)
        
        # Verify enhanced document
        assert enhanced_doc.content == "Enhanced content version 2"  # Should use best sample
        assert enhanced_doc.metadata["enhanced"] is True
        assert enhanced_doc.metadata["enhancement_quality"] == 0.90
        assert enhanced_doc.metadata["original_id"] == str(original_doc.id)
        assert enhanced_doc.source_type == original_doc.source_type
        
        # Test with empty samples
        enhanced_empty = self.enhancement_service._merge_with_quality_samples(original_doc, [])
        assert enhanced_empty == original_doc  # Should return original if no samples
    
    @pytest.mark.asyncio
    async def test_enhance_with_quality_samples_algorithm(self):
        """
        Test complete quality sample enhancement algorithm.
        
        Validates Requirement 5.1: THE SuperInsight_Platform SHALL 支持填充优质样本数据
        """
        # Create test documents with varying quality
        documents = [
            Document(
                source_type="file",
                source_config={"file_path": "low_quality.txt"},
                content="Short",  # Low quality content
                metadata={}
            ),
            Document(
                source_type="file",
                source_config={"file_path": "medium_quality.txt"},
                content="Medium length content with some information",
                metadata={"category": "test"}
            )
        ]
        
        # Create enhancement config
        config = EnhancementConfig(
            enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
            target_quality_threshold=0.7
        )
        
        # Perform enhancement
        result = await self.enhancement_service.enhance_with_quality_samples(documents, config)
        
        # Verify enhancement result
        assert isinstance(result, EnhancementResult)
        assert result.enhancement_type == EnhancementType.QUALITY_SAMPLE_FILL
        assert result.original_count == len(documents)
        assert result.quality_improvement >= 0.0  # Should improve or maintain quality
        assert result.processing_time > 0.0
        
        # Verify metadata contains quality information
        assert "original_avg_quality" in result.metadata
        assert "new_avg_quality" in result.metadata
        assert "low_quality_count" in result.metadata
        assert "quality_threshold" in result.metadata
        
        # Verify quality improvement calculation
        original_quality = result.metadata["original_avg_quality"]
        new_quality = result.metadata["new_avg_quality"]
        expected_improvement = new_quality - original_quality
        assert abs(result.quality_improvement - expected_improvement) < 1e-10
    
    @pytest.mark.asyncio
    async def test_positive_data_amplification_logic(self):
        """
        Test positive data amplification logic correctness.
        
        Validates Requirement 5.2: WHEN 进行数据增强时，THE SuperInsight_Platform SHALL 放大正向激励数据占比
        """
        # Create test tasks with varying quality scores
        tasks = [
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.9  # High quality (positive)
            ),
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.85  # High quality (positive)
            ),
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.6   # Low quality (negative)
            ),
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.5   # Low quality (negative)
            )
        ]
        
        # Create amplification config
        config = EnhancementConfig(
            enhancement_type=EnhancementType.POSITIVE_AMPLIFICATION,
            target_quality_threshold=0.8,
            amplification_factor=2.0
        )
        
        # Perform positive amplification
        result = await self.enhancement_service.amplify_positive_data(tasks, config)
        
        # Verify amplification result
        assert isinstance(result, EnhancementResult)
        assert result.enhancement_type == EnhancementType.POSITIVE_AMPLIFICATION
        assert result.original_count == len(tasks)
        assert result.enhanced_count >= result.original_count  # Should add samples
        
        # Verify positive samples were identified correctly
        positive_count = result.metadata["positive_count"]
        assert positive_count == 2  # Only 2 tasks above threshold (0.9, 0.85)
        
        # Verify amplification occurred
        amplified_count = result.metadata["amplified_count"]
        expected_target = int(positive_count * config.amplification_factor)
        expected_amplification = max(0, expected_target - positive_count)
        assert amplified_count == expected_amplification
        
        # Verify quality improvement calculation
        assert result.quality_improvement >= 0.0  # Should improve positive ratio
    
    @pytest.mark.asyncio
    async def test_generate_amplified_samples_logic(self):
        """
        Test amplified sample generation logic.
        
        Validates Requirement 5.2: WHEN 进行数据增强时，THE SuperInsight_Platform SHALL 放大正向激励数据占比
        """
        # Create positive tasks
        positive_tasks = [
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.9,
                annotations=[{"label": "positive", "confidence": 0.9}]
            ),
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.85,
                annotations=[{"label": "positive", "confidence": 0.85}]
            )
        ]
        
        config = EnhancementConfig(
            enhancement_type=EnhancementType.POSITIVE_AMPLIFICATION,
            target_quality_threshold=0.8,
            amplification_factor=2.0
        )
        
        # Generate amplified samples
        amplified_tasks = await self.enhancement_service._generate_amplified_samples(
            positive_tasks, 3, config
        )
        
        # Verify amplified samples
        assert len(amplified_tasks) == 3
        
        for task in amplified_tasks:
            assert isinstance(task, Task)
            assert task.quality_score > 0.8  # Should maintain high quality
            assert task.project_id == "test_project"
            assert len(task.annotations) > 0  # Should copy annotations
    
    @pytest.mark.asyncio
    async def test_batch_processing_functionality(self):
        """
        Test batch data enhancement processing functionality.
        
        Validates Requirement 5.5: THE SuperInsight_Platform SHALL 支持批量数据增强操作
        """
        # Create large batch of documents
        documents = []
        for i in range(25):  # Create 25 documents for batch testing
            doc = Document(
                source_type="file",
                source_config={"file_path": f"batch_doc_{i}.txt"},
                content=f"Batch document content {i} with varying length and quality",
                metadata={"batch_id": "test_batch", "doc_number": i}
            )
            documents.append(doc)
        
        # Create batch enhancement config
        config = EnhancementConfig(
            enhancement_type=EnhancementType.BATCH_ENHANCEMENT,
            batch_size=10,  # Process in batches of 10
            target_quality_threshold=0.7,
            preserve_original=True
        )
        
        # Perform batch enhancement
        result = await self.enhancement_service.batch_enhance_data(documents, config)
        
        # Verify batch processing result
        assert isinstance(result, EnhancementResult)
        assert result.enhancement_type == EnhancementType.BATCH_ENHANCEMENT
        assert result.original_count == len(documents)
        assert result.enhanced_count >= 0
        assert result.processing_time > 0.0
        
        # Verify batch processing metadata
        assert "batch_count" in result.metadata
        assert "batch_size" in result.metadata
        assert "preserve_original" in result.metadata
        
        expected_batches = (len(documents) + config.batch_size - 1) // config.batch_size
        assert result.metadata["batch_count"] == expected_batches
        assert result.metadata["batch_size"] == config.batch_size
        assert result.metadata["preserve_original"] == config.preserve_original
    
    @pytest.mark.asyncio
    async def test_enhance_batch_processing_logic(self):
        """
        Test individual batch enhancement processing logic.
        
        Validates Requirement 5.5: THE SuperInsight_Platform SHALL 支持批量数据增强操作
        """
        # Create batch of documents
        batch = [
            Document(
                source_type="file",
                source_config={"file_path": "batch1.txt"},
                content="Batch document 1",
                metadata={"batch": True}
            ),
            Document(
                source_type="file",
                source_config={"file_path": "batch2.txt"},
                content="Batch document 2",
                metadata={"batch": True}
            )
        ]
        
        config = EnhancementConfig(
            enhancement_type=EnhancementType.BATCH_ENHANCEMENT,
            target_quality_threshold=0.7
        )
        
        # Process batch
        enhanced_batch = await self.enhancement_service._enhance_batch(batch, config)
        
        # Verify batch processing
        assert len(enhanced_batch) == len(batch)
        
        for i, enhanced_doc in enumerate(enhanced_batch):
            assert isinstance(enhanced_doc, Document)
            assert enhanced_doc.metadata["enhanced"] is True
            assert "original_id" in enhanced_doc.metadata
            assert enhanced_doc.source_type == batch[i].source_type
    
    @pytest.mark.asyncio
    async def test_update_quality_scores_functionality(self):
        """
        Test quality score update functionality after enhancement.
        
        Validates Requirement 5.4: WHEN 数据增强完成时，THE SuperInsight_Platform SHALL 更新数据质量评分
        """
        # Create test tasks
        tasks = [
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.6
            ),
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.7
            )
        ]
        
        # Create enhancement result with positive improvement
        enhancement_result = EnhancementResult(
            enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
            original_count=2,
            enhanced_count=2,
            quality_improvement=0.2,  # 20% improvement
            processing_time=1.0
        )
        
        # Update quality scores
        updated_tasks = await self.enhancement_service.update_quality_scores(
            tasks, enhancement_result
        )
        
        # Verify quality score updates
        assert len(updated_tasks) == len(tasks)
        
        # Calculate expected scores
        improvement_factor = 1.0 + enhancement_result.quality_improvement
        expected_scores = [
            min(1.0, 0.6 * improvement_factor),
            min(1.0, 0.7 * improvement_factor)
        ]
        
        for i, task in enumerate(updated_tasks):
            assert task.quality_score == expected_scores[i]
            assert task.quality_score <= 1.0  # Should not exceed maximum
    
    def test_calculate_amplification_improvement_logic(self):
        """
        Test amplification improvement calculation logic.
        
        Validates Requirement 5.2: WHEN 进行数据增强时，THE SuperInsight_Platform SHALL 放大正向激励数据占比
        """
        # Create original tasks
        original_tasks = [
            Task(document_id=uuid4(), project_id="test", quality_score=0.8),
            Task(document_id=uuid4(), project_id="test", quality_score=0.6),
            Task(document_id=uuid4(), project_id="test", quality_score=0.9)
        ]
        
        # Create amplified tasks (high quality)
        amplified_tasks = [
            Task(document_id=uuid4(), project_id="test", quality_score=0.88),
            Task(document_id=uuid4(), project_id="test", quality_score=0.92)
        ]
        
        # Calculate improvement
        improvement = self.enhancement_service._calculate_amplification_improvement(
            original_tasks, amplified_tasks
        )
        
        # Verify improvement calculation
        assert improvement >= 0.0  # Should be positive
        
        # Test with empty lists
        improvement_empty = self.enhancement_service._calculate_amplification_improvement([], [])
        assert improvement_empty == 0.0
        
        improvement_no_amplified = self.enhancement_service._calculate_amplification_improvement(
            original_tasks, []
        )
        assert improvement_no_amplified == 0.0


class TestDataReconstructionService:
    """Unit tests for DataReconstructionService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reconstruction_service = DataReconstructionService()
    
    def test_reconstruction_service_initialization(self):
        """Test DataReconstructionService initializes correctly."""
        assert self.reconstruction_service.db_manager is None
        assert len(self.reconstruction_service.reconstruction_history) == 0
    
    @pytest.mark.asyncio
    async def test_document_structure_transformation(self):
        """Test document structure transformation functionality."""
        # Create test document
        document = Document(
            source_type="file",
            source_config={"file_path": "test.txt"},
            content="Original document content",
            metadata={"category": "test", "version": "1.0"}
        )
        
        # Create transformation config
        config = ReconstructionConfig(
            reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
            source_format="text",
            target_format="enhanced_text",
            transformation_rules={
                "add_prefix": "[ENHANCED] ",
                "add_suffix": " [END]",
                "add_metadata": {"transformed": True, "enhancement_date": "2024-01-01"}
            },
            preserve_metadata=True
        )
        
        # Perform reconstruction
        result = await self.reconstruction_service.reconstruct_document(document, config)
        
        # Verify reconstruction result
        assert result.success is True
        assert isinstance(result.reconstructed_data, Document)
        
        reconstructed_doc = result.reconstructed_data
        assert reconstructed_doc.content == "[ENHANCED] Original document content [END]"
        assert reconstructed_doc.metadata["transformed"] is True
        assert reconstructed_doc.metadata["category"] == "test"  # Preserved
        assert reconstructed_doc.metadata["reconstructed"] is True
        assert "original_id" in reconstructed_doc.metadata
    
    @pytest.mark.asyncio
    async def test_document_format_conversion(self):
        """Test document format conversion functionality."""
        # Create test document
        document = Document(
            source_type="file",
            source_config={"file_path": "test.txt"},
            content="Plain text content",
            metadata={"format": "text"}
        )
        
        # Create format conversion config
        config = ReconstructionConfig(
            reconstruction_type=ReconstructionType.FORMAT_CONVERSION,
            source_format="text",
            target_format="json",
            preserve_metadata=True
        )
        
        # Perform format conversion
        result = await self.reconstruction_service.reconstruct_document(document, config)
        
        # Verify conversion result
        assert result.success is True
        reconstructed_doc = result.reconstructed_data
        assert '{"content": "Plain text content", "format": "json"}' in reconstructed_doc.content
        assert reconstructed_doc.metadata["format_converted"] is True
        assert reconstructed_doc.metadata["source_format"] == "text"
        assert reconstructed_doc.metadata["target_format"] == "json"
    
    @pytest.mark.asyncio
    async def test_batch_reconstruction_functionality(self):
        """
        Test batch reconstruction functionality.
        
        Validates Requirement 5.3: THE SuperInsight_Platform SHALL 提供数据重构接口
        """
        # Create multiple documents and tasks
        documents = [
            Document(
                source_type="file",
                source_config={"file_path": f"doc_{i}.txt"},
                content=f"Document {i} content",
                metadata={"doc_id": i}
            )
            for i in range(3)
        ]
        
        tasks = [
            Task(
                document_id=uuid4(),
                project_id="test_project",
                quality_score=0.8
            )
            for _ in range(2)
        ]
        
        # Mix documents and tasks
        data_items = documents + tasks
        
        # Create reconstruction config
        config = ReconstructionConfig(
            reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
            source_format="mixed",
            target_format="enhanced",
            preserve_metadata=True
        )
        
        # Perform batch reconstruction
        results = await self.reconstruction_service.batch_reconstruct(data_items, config)
        
        # Verify batch results
        assert len(results) == len(data_items)
        
        # Check document reconstructions
        for i in range(3):
            result = results[i]
            assert result.success is True
            assert isinstance(result.reconstructed_data, Document)
        
        # Check task reconstructions
        for i in range(3, 5):
            result = results[i]
            assert result.success is True
            assert isinstance(result.reconstructed_data, Task)
    
    def test_reconstruction_history_tracking(self):
        """
        Test reconstruction history tracking functionality.
        
        Validates Requirement 5.4: WHEN 数据增强完成时，THE SuperInsight_Platform SHALL 更新数据质量评分
        (History tracking supports quality score updates)
        """
        # Initially empty history
        history = self.reconstruction_service.get_reconstruction_history()
        assert len(history) == 0
        
        # Create and add reconstruction records
        record1 = ReconstructionRecord(
            source_id=uuid4(),
            target_id=uuid4(),
            reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
            config=ReconstructionConfig(
                reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
                source_format="text",
                target_format="json"
            ),
            status="completed"
        )
        
        record2 = ReconstructionRecord(
            source_id=uuid4(),
            target_id=uuid4(),
            reconstruction_type=ReconstructionType.FORMAT_CONVERSION,
            config=ReconstructionConfig(
                reconstruction_type=ReconstructionType.FORMAT_CONVERSION,
                source_format="json",
                target_format="xml"
            ),
            status="failed"
        )
        
        self.reconstruction_service.reconstruction_history.extend([record1, record2])
        
        # Test getting all history
        all_history = self.reconstruction_service.get_reconstruction_history()
        assert len(all_history) == 2
        
        # Test filtering by source_id
        filtered_history = self.reconstruction_service.get_reconstruction_history(
            source_id=record1.source_id
        )
        assert len(filtered_history) == 1
        assert filtered_history[0].source_id == record1.source_id
        
        # Test filtering by reconstruction_type
        type_filtered = self.reconstruction_service.get_reconstruction_history(
            reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM
        )
        assert len(type_filtered) == 1
        assert type_filtered[0].reconstruction_type == ReconstructionType.STRUCTURE_TRANSFORM
    
    @pytest.mark.asyncio
    async def test_reconstruction_verification(self):
        """
        Test reconstruction verification functionality.
        
        Validates Requirement 5.4: WHEN 数据增强完成时，THE SuperInsight_Platform SHALL 更新数据质量评分
        (Verification ensures quality of reconstruction results)
        """
        # Create completed reconstruction record
        completed_record = ReconstructionRecord(
            source_id=uuid4(),
            target_id=uuid4(),
            reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
            config=ReconstructionConfig(
                reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
                source_format="text",
                target_format="json"
            ),
            status="completed"
        )
        
        # Verify completed reconstruction
        verification_result = await self.reconstruction_service.verify_reconstruction(completed_record)
        
        assert verification_result["verified"] is True
        assert verification_result["source_id"] == str(completed_record.source_id)
        assert verification_result["target_id"] == str(completed_record.target_id)
        assert "verification_time" in verification_result
        
        # Test verification of incomplete reconstruction
        incomplete_record = ReconstructionRecord(
            source_id=uuid4(),
            target_id=uuid4(),
            reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
            config=ReconstructionConfig(
                reconstruction_type=ReconstructionType.STRUCTURE_TRANSFORM,
                source_format="text",
                target_format="json"
            ),
            status="in_progress"
        )
        
        verification_result = await self.reconstruction_service.verify_reconstruction(incomplete_record)
        
        assert verification_result["verified"] is False
        assert "not completed" in verification_result["reason"]


class TestEnhancementModels:
    """Unit tests for enhancement data models."""
    
    def test_quality_sample_validation(self):
        """Test QualitySample model validation."""
        # Valid quality sample
        sample = QualitySample(
            content="Test content",
            quality_score=0.85,
            metadata={"type": "test"}
        )
        
        assert sample.content == "Test content"
        assert sample.quality_score == 0.85
        assert sample.metadata["type"] == "test"
        assert sample.id is not None
        assert sample.created_at is not None
        
        # Test quality score validation
        with pytest.raises(ValueError, match="quality_score must be between 0.0 and 1.0"):
            QualitySample(content="Test", quality_score=1.5)
        
        with pytest.raises(ValueError, match="quality_score must be between 0.0 and 1.0"):
            QualitySample(content="Test", quality_score=-0.1)
        
        # Test content validation
        with pytest.raises(ValueError, match="content cannot be empty"):
            QualitySample(content="", quality_score=0.8)
        
        with pytest.raises(ValueError, match="content cannot be empty"):
            QualitySample(content="   ", quality_score=0.8)
    
    def test_enhancement_config_validation(self):
        """Test EnhancementConfig model validation."""
        # Valid config
        config = EnhancementConfig(
            enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
            target_quality_threshold=0.8,
            amplification_factor=2.0,
            batch_size=100
        )
        
        assert config.enhancement_type == EnhancementType.QUALITY_SAMPLE_FILL
        assert config.target_quality_threshold == 0.8
        assert config.amplification_factor == 2.0
        assert config.batch_size == 100
        assert config.preserve_original is True  # Default value
        
        # Test quality threshold validation
        with pytest.raises(ValueError, match="target_quality_threshold must be between 0.0 and 1.0"):
            EnhancementConfig(
                enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
                target_quality_threshold=1.5
            )
        
        # Test amplification factor validation
        with pytest.raises(ValueError, match="amplification_factor must be positive"):
            EnhancementConfig(
                enhancement_type=EnhancementType.POSITIVE_AMPLIFICATION,
                amplification_factor=-1.0
            )
        
        # Test batch size validation
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EnhancementConfig(
                enhancement_type=EnhancementType.BATCH_ENHANCEMENT,
                batch_size=0
            )
    
    def test_enhancement_result_serialization(self):
        """Test EnhancementResult serialization and deserialization."""
        # Create enhancement result
        result = EnhancementResult(
            enhancement_type=EnhancementType.QUALITY_SAMPLE_FILL,
            original_count=10,
            enhanced_count=15,
            quality_improvement=0.2,
            processing_time=5.5,
            metadata={"test": "data"}
        )
        
        # Test to_dict serialization
        result_dict = result.to_dict()
        
        assert result_dict["enhancement_type"] == "quality_sample_fill"
        assert result_dict["original_count"] == 10
        assert result_dict["enhanced_count"] == 15
        assert result_dict["quality_improvement"] == 0.2
        assert result_dict["processing_time"] == 5.5
        assert result_dict["metadata"]["test"] == "data"
        assert "id" in result_dict
        assert "created_at" in result_dict
        
        # Test from_dict deserialization
        restored_result = EnhancementResult.from_dict(result_dict)
        
        assert restored_result.enhancement_type == result.enhancement_type
        assert restored_result.original_count == result.original_count
        assert restored_result.enhanced_count == result.enhanced_count
        assert restored_result.quality_improvement == result.quality_improvement
        assert restored_result.processing_time == result.processing_time
        assert restored_result.metadata == result.metadata
        assert restored_result.id == result.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])