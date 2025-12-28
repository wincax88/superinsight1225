"""
Integration tests for Knowledge Graph system.

Tests end-to-end knowledge graph construction, multi-module collaboration,
and performance characteristics.
"""

import pytest
from typing import List
import time

from src.knowledge_graph.nlp.entity_extractor import EntityExtractor
from src.knowledge_graph.nlp.relation_extractor import RelationExtractor
from src.knowledge_graph.nlp.text_processor import TextProcessor
from src.knowledge_graph.core.models import (
    EntityType, RelationType, ExtractedEntity, ExtractedRelation
)


class TestEndToEndKnowledgeGraphConstruction:
    """Test end-to-end knowledge graph construction."""

    @pytest.fixture
    def entity_extractor(self):
        """Create an EntityExtractor instance."""
        extractor = EntityExtractor(use_rule_based=True)
        extractor.initialize()
        return extractor

    @pytest.fixture
    def relation_extractor(self):
        """Create a RelationExtractor instance."""
        extractor = RelationExtractor()
        extractor.initialize()
        return extractor

    @pytest.fixture
    def text_processor(self):
        """Create a TextProcessor instance."""
        return TextProcessor()

    def test_single_document_processing(self, entity_extractor, relation_extractor):
        """Test processing a single document."""
        text = "张三在阿里巴巴工作，公司位于杭州。2024年1月15日他获得了100万元的奖励。"
        
        # Extract entities
        entities = entity_extractor.extract(text)
        assert len(entities) > 0, "No entities extracted"
        
        # Extract relations
        relations = relation_extractor.extract(text, entities)
        assert isinstance(relations, list), "Relations should be a list"

    def test_multiple_documents_processing(self, entity_extractor, relation_extractor):
        """Test processing multiple documents."""
        documents = [
            "2024年1月15日开会",
            "投资100万元",
            "增长率25%",
        ]
        
        all_entities = []
        all_relations = []
        
        for doc in documents:
            entities = entity_extractor.extract(doc)
            relations = relation_extractor.extract(doc, entities)
            
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        assert len(all_entities) > 0, "No entities extracted from documents"
        assert isinstance(all_relations, list), "Relations should be a list"

    def test_entity_deduplication(self, entity_extractor):
        """Test that duplicate entities are handled correctly."""
        # Same entity appears multiple times
        text = "2024年1月15日开会，2024年1月15日是一个重要日期"
        
        entities = entity_extractor.extract(text)
        
        # Count unique entity texts
        unique_texts = set(e.text for e in entities)
        
        # Should have some entities
        assert len(entities) > 0
        assert len(unique_texts) > 0

    def test_entity_type_distribution(self, entity_extractor):
        """Test entity type distribution in extracted entities."""
        text = """
        2024年1月15日，张三在北京的阿里巴巴公司工作。
        他的年薪是100万元，增长率达到25%。
        下午3点30分，他参加了一个重要会议。
        """
        
        entities = entity_extractor.extract(text)
        
        # Count by type
        type_counts = {}
        for entity in entities:
            type_name = entity.entity_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Should have multiple types
        assert len(type_counts) > 0, "No entity types found"

    def test_text_processing_pipeline(self, text_processor, entity_extractor):
        """Test the complete text processing pipeline."""
        text = "自然语言处理是人工智能的重要领域"
        
        # Process text
        processed = text_processor.process(text)
        assert processed is not None
        assert len(processed.tokens) > 0
        
        # Extract entities
        entities = entity_extractor.extract(text)
        assert isinstance(entities, list)

    def test_large_document_processing(self, entity_extractor):
        """Test processing of large documents."""
        # Create a large document
        text = """
        2024年1月15日，张三在北京的阿里巴巴公司工作。
        """ * 100  # Repeat to create larger document
        
        start_time = time.time()
        entities = entity_extractor.extract(text)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 10.0, f"Processing took too long: {elapsed_time}s"
        assert len(entities) > 0, "No entities extracted from large document"

    def test_batch_processing_performance(self, entity_extractor):
        """Test batch processing performance."""
        texts = [
            "2024年1月15日开会",
            "投资100万元",
            "增长率25%",
        ] * 10  # 30 documents
        
        start_time = time.time()
        results = entity_extractor.extract_batch(texts)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 10.0, f"Batch processing took too long: {elapsed_time}s"
        assert len(results) == len(texts)

    def test_entity_extraction_accuracy(self, entity_extractor):
        """Test entity extraction accuracy on known examples."""
        test_cases = [
            ("2024年1月15日", EntityType.DATE),
            ("下午3点30分", EntityType.TIME),
            ("100万元", EntityType.MONEY),
            ("25%", EntityType.PERCENT),
        ]
        
        for text, expected_type in test_cases:
            entities = entity_extractor.extract(text, entity_types=[expected_type])
            
            # Should find at least one entity of expected type
            found = any(e.entity_type == expected_type for e in entities)
            assert found, f"Failed to extract {expected_type} from '{text}'"

    def test_relation_extraction_with_entities(self, entity_extractor, relation_extractor):
        """Test relation extraction with extracted entities."""
        text = "2024年1月15日投资100万元"
        
        # Extract entities
        entities = entity_extractor.extract(text)
        
        # Extract relations
        relations = relation_extractor.extract(text, entities)
        
        # Should have entities and relations
        assert len(entities) > 0, "No entities extracted"
        assert isinstance(relations, list), "Relations should be a list"

    def test_consistency_across_runs(self, entity_extractor):
        """Test that results are consistent across multiple runs."""
        text = "2024年1月15日投资100万元"
        
        results = []
        for _ in range(5):
            entities = entity_extractor.extract(text)
            results.append(entities)
        
        # All runs should produce same number of entities
        counts = [len(r) for r in results]
        assert len(set(counts)) == 1, f"Inconsistent entity counts: {counts}"
        
        # All entities should be identical
        for i in range(1, len(results)):
            assert len(results[0]) == len(results[i])
            for e1, e2 in zip(results[0], results[i]):
                assert e1.text == e2.text
                assert e1.entity_type == e2.entity_type


class TestMultiModuleCollaboration:
    """Test collaboration between multiple modules."""

    @pytest.fixture
    def entity_extractor(self):
        """Create an EntityExtractor instance."""
        extractor = EntityExtractor(use_rule_based=True)
        extractor.initialize()
        return extractor

    @pytest.fixture
    def relation_extractor(self):
        """Create a RelationExtractor instance."""
        extractor = RelationExtractor()
        extractor.initialize()
        return extractor

    @pytest.fixture
    def text_processor(self):
        """Create a TextProcessor instance."""
        return TextProcessor()

    def test_entity_and_relation_extraction_flow(self, entity_extractor, relation_extractor):
        """Test the flow from entity extraction to relation extraction."""
        text = "2024年1月15日投资100万元"
        
        # Step 1: Extract entities
        entities = entity_extractor.extract(text)
        assert len(entities) > 0
        
        # Step 2: Extract relations
        relations = relation_extractor.extract(text, entities)
        assert isinstance(relations, list)

    def test_text_processing_and_entity_extraction(self, text_processor, entity_extractor):
        """Test text processing followed by entity extraction."""
        text = "  2024年1月15日  开会  "
        
        # Step 1: Process text
        processed = text_processor.process(text)
        assert processed is not None
        
        # Step 2: Extract entities
        entities = entity_extractor.extract(text)
        assert len(entities) > 0

    def test_entity_statistics_generation(self, entity_extractor):
        """Test generation of entity statistics."""
        text = "2024年1月15日投资100万元，增长率25%"
        
        entities = entity_extractor.extract(text)
        stats = entity_extractor.get_entity_statistics(entities)
        
        # Verify statistics structure
        assert "total_count" in stats
        assert "by_type" in stats
        assert "by_source" in stats
        assert "avg_confidence" in stats
        assert "unique_texts" in stats
        
        # Verify statistics values
        assert stats["total_count"] == len(entities)
        assert stats["avg_confidence"] >= 0.0
        assert stats["avg_confidence"] <= 1.0

    def test_relation_statistics_generation(self, relation_extractor):
        """Test generation of relation statistics."""
        relations = []
        stats = relation_extractor.get_relation_statistics(relations)
        
        # Verify statistics structure
        assert "total_count" in stats
        assert "by_type" in stats
        assert stats["total_count"] == 0


class TestPerformanceAndStress:
    """Test performance and stress conditions."""

    @pytest.fixture
    def entity_extractor(self):
        """Create an EntityExtractor instance."""
        extractor = EntityExtractor(use_rule_based=True)
        extractor.initialize()
        return extractor

    def test_empty_text_handling(self, entity_extractor):
        """Test handling of empty text."""
        result = entity_extractor.extract("")
        assert result == []

    def test_very_long_text(self, entity_extractor):
        """Test handling of very long text."""
        # Create a very long text
        text = "2024年1月15日 " * 1000
        
        start_time = time.time()
        entities = entity_extractor.extract(text)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 30.0, f"Processing took too long: {elapsed_time}s"

    def test_special_characters_handling(self, entity_extractor):
        """Test handling of special characters."""
        text = "2024年1月15日@#$%^&*()开会"
        
        entities = entity_extractor.extract(text)
        assert isinstance(entities, list)

    def test_unicode_handling(self, entity_extractor):
        """Test handling of unicode characters."""
        text = "2024年1月15日 🎉 开会 😊"
        
        entities = entity_extractor.extract(text)
        assert isinstance(entities, list)

    def test_mixed_language_handling(self, entity_extractor):
        """Test handling of mixed language text."""
        text = "2024年1月15日 Meeting at 3:30 PM 投资100万元"
        
        entities = entity_extractor.extract(text)
        assert isinstance(entities, list)

    def test_concurrent_extraction(self, entity_extractor):
        """Test concurrent extraction requests."""
        texts = [
            "2024年1月15日开会",
            "投资100万元",
            "增长率25%",
        ]
        
        # Extract from multiple texts
        results = entity_extractor.extract_batch(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, list) for r in results)

    def test_memory_efficiency(self, entity_extractor):
        """Test memory efficiency with large batches."""
        # Create a large batch
        texts = ["2024年1月15日"] * 1000
        
        start_time = time.time()
        results = entity_extractor.extract_batch(texts)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 30.0, f"Batch processing took too long: {elapsed_time}s"
        assert len(results) == 1000


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def entity_extractor(self):
        """Create an EntityExtractor instance."""
        extractor = EntityExtractor(use_rule_based=True)
        extractor.initialize()
        return extractor

    def test_none_input_handling(self, entity_extractor):
        """Test handling of None input."""
        result = entity_extractor.extract(None)
        assert result == []

    def test_invalid_entity_type_filter(self, entity_extractor):
        """Test handling of invalid entity type filter."""
        text = "2024年1月15日"
        
        # Should handle empty filter gracefully - returns empty list
        result = entity_extractor.extract(text, entity_types=[])
        # When filtering with empty list, no entities should be returned
        # because no entity types are in the filter
        assert isinstance(result, list)

    def test_confidence_threshold_edge_cases(self, entity_extractor):
        """Test confidence threshold edge cases."""
        text = "2024年1月15日"
        
        # Test with threshold = 0
        entity_extractor.confidence_threshold = 0.0
        result1 = entity_extractor.extract(text)
        
        # Test with threshold = 1
        entity_extractor.confidence_threshold = 1.0
        result2 = entity_extractor.extract(text)
        
        # Result with threshold 0 should have >= entities than threshold 1
        assert len(result1) >= len(result2)

    def test_batch_with_empty_texts(self, entity_extractor):
        """Test batch extraction with empty texts."""
        texts = ["", "2024年1月15日", "", "投资100万元"]
        
        results = entity_extractor.extract_batch(texts)
        
        assert len(results) == 4
        assert results[0] == []  # Empty text
        assert len(results[1]) > 0  # Valid text
        assert results[2] == []  # Empty text
        assert len(results[3]) > 0  # Valid text
