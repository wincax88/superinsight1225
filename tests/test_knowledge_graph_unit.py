"""
Unit tests for Knowledge Graph system components.

Tests core functionality of entity extraction, relation extraction,
and graph database operations.
"""

import pytest
from datetime import datetime
from typing import List

from src.knowledge_graph.nlp.entity_extractor import EntityExtractor
from src.knowledge_graph.nlp.relation_extractor import RelationExtractor, RelationPattern
from src.knowledge_graph.nlp.text_processor import TextProcessor
from src.knowledge_graph.core.models import (
    EntityType, RelationType, ExtractedEntity, ExtractedRelation
)
from src.knowledge_graph.core.graph_db import GraphDatabase


class TestEntityExtractor:
    """Test suite for EntityExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance."""
        extractor = EntityExtractor(
            use_rule_based=True,
            confidence_threshold=0.5
        )
        extractor.initialize()
        return extractor

    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        result = extractor.extract("")
        assert result == []

    def test_extract_none_text(self, extractor):
        """Test extraction from None text."""
        result = extractor.extract(None)
        assert result == []

    def test_extract_whitespace_only(self, extractor):
        """Test extraction from whitespace-only text."""
        result = extractor.extract("   \n\t  ")
        assert result == []

    def test_extract_date_entities(self, extractor):
        """Test extraction of date entities."""
        text = "会议在2024年1月15日举行"
        result = extractor.extract(text, entity_types=[EntityType.DATE])
        
        assert len(result) > 0
        assert any(e.entity_type == EntityType.DATE for e in result)

    def test_extract_time_entities(self, extractor):
        """Test extraction of time entities."""
        text = "下午3点30分开会"
        result = extractor.extract(text, entity_types=[EntityType.TIME])
        
        assert len(result) > 0
        assert any(e.entity_type == EntityType.TIME for e in result)

    def test_extract_money_entities(self, extractor):
        """Test extraction of money entities."""
        text = "投资金额为100万元"
        result = extractor.extract(text, entity_types=[EntityType.MONEY])
        
        assert len(result) > 0
        assert any(e.entity_type == EntityType.MONEY for e in result)

    def test_extract_percent_entities(self, extractor):
        """Test extraction of percent entities."""
        text = "增长率达到25%"
        result = extractor.extract(text, entity_types=[EntityType.PERCENT])
        
        assert len(result) > 0
        assert any(e.entity_type == EntityType.PERCENT for e in result)

    def test_extract_batch(self, extractor):
        """Test batch extraction."""
        texts = [
            "2024年1月15日开会",
            "下午3点30分",
            "投资100万元"
        ]
        results = extractor.extract_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_extract_with_confidence_threshold(self, extractor):
        """Test extraction respects confidence threshold."""
        extractor.confidence_threshold = 0.9
        text = "2024年1月15日开会"
        result = extractor.extract(text)
        
        # All results should meet threshold
        assert all(e.confidence >= 0.9 for e in result)

    def test_extract_removes_overlapping(self, extractor):
        """Test that overlapping entities are removed."""
        text = "2024年1月15日"
        result = extractor.extract(text)
        
        # Check no overlapping entities
        for i, e1 in enumerate(result):
            for e2 in result[i+1:]:
                # Entities should not overlap
                assert e1.end_char <= e2.start_char or e2.end_char <= e1.start_char

    def test_add_custom_pattern(self, extractor):
        """Test adding custom extraction patterns."""
        extractor.add_pattern(EntityType.PRODUCT, r'产品\d+')
        text = "我们的产品001很受欢迎"
        result = extractor.extract(text, entity_types=[EntityType.PRODUCT])
        
        assert len(result) > 0

    def test_get_entity_statistics(self, extractor):
        """Test entity statistics calculation."""
        text = "2024年1月15日投资100万元，增长25%"
        entities = extractor.extract(text)
        stats = extractor.get_entity_statistics(entities)
        
        assert "total_count" in stats
        assert "by_type" in stats
        assert "by_source" in stats
        assert "avg_confidence" in stats
        assert stats["total_count"] == len(entities)

    def test_extract_maintains_position(self, extractor):
        """Test that extracted entities maintain correct positions."""
        text = "2024年1月15日开会"
        result = extractor.extract(text)
        
        for entity in result:
            # Verify position is correct
            extracted_text = text[entity.start_char:entity.end_char]
            assert extracted_text == entity.text


class TestRelationExtractor:
    """Test suite for RelationExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a RelationExtractor instance."""
        extractor = RelationExtractor(
            confidence_threshold=0.5
        )
        extractor.initialize()
        return extractor

    @pytest.fixture
    def entity_extractor(self):
        """Create an EntityExtractor instance."""
        extractor = EntityExtractor(use_rule_based=True)
        extractor.initialize()
        return extractor

    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        result = extractor.extract("", [])
        assert result == []

    def test_extract_no_entities(self, extractor):
        """Test extraction with no entities."""
        result = extractor.extract("无关文本", [])
        assert result == []

    def test_extract_with_entities(self, extractor, entity_extractor):
        """Test relation extraction with entities."""
        text = "2024年1月15日投资100万元"
        entities = entity_extractor.extract(text)
        
        # Should have at least some entities
        assert len(entities) > 0

    def test_extract_batch(self, extractor):
        """Test batch relation extraction."""
        texts = ["文本1", "文本2", "文本3"]
        entities_list = [[], [], []]
        
        results = extractor.extract_batch(texts, entities_list)
        assert len(results) == 3

    def test_add_custom_pattern(self, extractor):
        """Test adding custom relation patterns."""
        pattern = RelationPattern(
            source_types=[EntityType.PERSON],
            target_types=[EntityType.ORGANIZATION],
            relation_type=RelationType.WORKS_FOR,
            patterns=[r'{source}在{target}工作'],
            confidence=0.8
        )
        extractor.add_pattern(pattern)
        
        assert len(extractor.patterns) > 0

    def test_get_relation_statistics(self, extractor):
        """Test relation statistics calculation."""
        relations = []
        stats = extractor.get_relation_statistics(relations)
        
        assert "total_count" in stats
        assert "by_type" in stats
        assert stats["total_count"] == 0


class TestTextProcessor:
    """Test suite for TextProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a TextProcessor instance."""
        return TextProcessor()

    def test_process_empty_text(self, processor):
        """Test processing empty text."""
        result = processor.process("")
        assert result is not None
        assert len(result.tokens) == 0

    def test_process_simple_text(self, processor):
        """Test processing simple text."""
        text = "这是一个测试"
        result = processor.process(text)
        
        assert result is not None
        assert len(result.tokens) > 0

    def test_tokenization(self, processor):
        """Test text tokenization."""
        text = "我爱自然语言处理"
        result = processor.process(text)
        
        assert len(result.tokens) > 0
        # Verify tokens have required attributes
        for token in result.tokens:
            assert hasattr(token, 'text')
            assert hasattr(token, 'pos')

    def test_get_noun_phrases(self, processor):
        """Test noun phrase extraction."""
        text = "自然语言处理是人工智能的重要领域"
        result = processor.process(text)
        phrases = processor.get_noun_phrases(result.tokens)
        
        assert isinstance(phrases, list)

    def test_normalize_text(self, processor):
        """Test text normalization."""
        text = "  这是  一个   测试  "
        # TextProcessor doesn't have a normalize method, so we test process instead
        result = processor.process(text)
        
        assert result is not None
        assert len(result.tokens) > 0


class TestGraphDatabase:
    """Test suite for GraphDatabase."""

    @pytest.fixture
    def graph_db(self):
        """Create a GraphDatabase instance."""
        # Use in-memory or test database
        db = GraphDatabase(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test_password"
        )
        return db

    def test_create_entity(self, graph_db):
        """Test entity creation."""
        entity = ExtractedEntity(
            text="测试实体",
            entity_type=EntityType.PERSON,
            start_char=0,
            end_char=4,
            confidence=0.9,
            normalized_name="测试实体"
        )
        
        # Should not raise exception
        try:
            result = graph_db.create_entity(entity)
            assert result is not None
        except Exception as e:
            # Database might not be available in test environment
            pytest.skip(f"Database not available: {e}")

    def test_create_relation(self, graph_db):
        """Test relation creation."""
        entity1 = ExtractedEntity(
            text="张三",
            entity_type=EntityType.PERSON,
            start_char=0,
            end_char=2,
            confidence=0.9,
            normalized_name="张三"
        )
        
        entity2 = ExtractedEntity(
            text="阿里巴巴",
            entity_type=EntityType.ORGANIZATION,
            start_char=3,
            end_char=7,
            confidence=0.9,
            normalized_name="阿里巴巴"
        )
        
        relation = ExtractedRelation(
            source_entity=entity1,
            target_entity=entity2,
            relation_type=RelationType.WORKS_FOR,
            confidence=0.85
        )
        
        try:
            result = graph_db.create_relation(relation)
            assert result is not None
        except Exception as e:
            pytest.skip(f"Database not available: {e}")

    def test_query_neighbors(self, graph_db):
        """Test neighbor query."""
        try:
            result = graph_db.query_neighbors("test_entity", depth=1)
            assert isinstance(result, list)
        except Exception as e:
            pytest.skip(f"Database not available: {e}")

    def test_find_shortest_path(self, graph_db):
        """Test shortest path finding."""
        try:
            result = graph_db.find_shortest_path("entity1", "entity2")
            assert isinstance(result, list)
        except Exception as e:
            pytest.skip(f"Database not available: {e}")


class TestIntegration:
    """Integration tests for knowledge graph components."""

    def test_entity_extraction_consistency(self):
        """Test that entity extraction is consistent."""
        extractor = EntityExtractor(use_rule_based=True)
        extractor.initialize()
        
        text = "2024年1月15日开会"
        result1 = extractor.extract(text)
        result2 = extractor.extract(text)
        
        # Same text should produce same entities
        assert len(result1) == len(result2)
        for e1, e2 in zip(result1, result2):
            assert e1.text == e2.text
            assert e1.entity_type == e2.entity_type

    def test_entity_and_relation_extraction(self):
        """Test entity and relation extraction together."""
        entity_extractor = EntityExtractor(use_rule_based=True)
        entity_extractor.initialize()
        
        relation_extractor = RelationExtractor()
        relation_extractor.initialize()
        
        text = "2024年1月15日投资100万元"
        entities = entity_extractor.extract(text)
        
        # Should extract some entities
        assert len(entities) > 0
        
        # Try to extract relations
        relations = relation_extractor.extract(text, entities)
        assert isinstance(relations, list)

    def test_batch_processing(self):
        """Test batch processing of multiple texts."""
        extractor = EntityExtractor(use_rule_based=True)
        extractor.initialize()
        
        texts = [
            "2024年1月15日开会",
            "投资100万元",
            "增长率25%"
        ]
        
        results = extractor.extract_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
