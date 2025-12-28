"""
Property-based tests for Knowledge Graph entity extraction.

Tests correctness properties using Hypothesis for property-based testing.
Feature: knowledge-graph, Property 1: Entity extraction consistency
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime

from src.knowledge_graph.nlp.entity_extractor import EntityExtractor
from src.knowledge_graph.core.models import EntityType, ExtractedEntity


# Create a global extractor instance for property tests
_extractor = None

def get_extractor():
    """Get or create global extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor(
            use_rule_based=True,
            confidence_threshold=0.5
        )
        _extractor.initialize()
    return _extractor


# Strategies for generating test data
@st.composite
def entity_texts(draw):
    """Generate realistic entity text samples."""
    entity_types = [
        "2024年1月15日",  # Date
        "下午3点30分",     # Time
        "100万元",         # Money
        "25%",             # Percent
        "张三",            # Person name
        "阿里巴巴",        # Organization
        "北京",            # Location
    ]
    return draw(st.sampled_from(entity_types))


@st.composite
def text_with_entities(draw):
    """Generate text containing entities."""
    entity = draw(entity_texts())
    prefix = draw(st.text(min_size=0, max_size=20))
    suffix = draw(st.text(min_size=0, max_size=20))
    return f"{prefix}{entity}{suffix}"


class TestEntityExtractionConsistency:
    """
    Property 1: Entity extraction consistency
    
    For any input text, the same entities should be extracted consistently
    across multiple extractions with identical configuration.
    
    Validates: Requirements 2.1, 2.2
    """

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_entity_extraction_idempotent(self, text):
        """
        Property: Entity extraction is idempotent.
        
        For any text, extracting entities multiple times should produce
        identical results (same entities, same positions, same types).
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        # Extract entities multiple times
        result1 = extractor.extract(text)
        result2 = extractor.extract(text)
        result3 = extractor.extract(text)
        
        # All results should have same length
        assert len(result1) == len(result2) == len(result3), \
            f"Extraction count mismatch for text: {text}"
        
        # All entities should match exactly
        for e1, e2, e3 in zip(result1, result2, result3):
            assert e1.text == e2.text == e3.text
            assert e1.entity_type == e2.entity_type == e3.entity_type
            assert e1.start_char == e2.start_char == e3.start_char
            assert e1.end_char == e2.end_char == e3.end_char

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_entity_positions_valid(self, text):
        """
        Property: Entity positions are valid.
        
        For any extracted entity, the start and end positions should be
        valid indices within the text, and the extracted substring should
        match the entity text.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        entities = extractor.extract(text)
        
        for entity in entities:
            # Positions should be valid
            assert 0 <= entity.start_char <= len(text)
            assert 0 <= entity.end_char <= len(text)
            assert entity.start_char <= entity.end_char
            
            # Extracted text should match
            extracted = text[entity.start_char:entity.end_char]
            assert extracted == entity.text

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_entity_confidence_valid(self, text):
        """
        Property: Entity confidence scores are valid.
        
        For any extracted entity, the confidence score should be between
        0 and 1, and should be >= the configured threshold.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        entities = extractor.extract(text)
        
        for entity in entities:
            # Confidence should be in valid range
            assert 0.0 <= entity.confidence <= 1.0
            
            # Confidence should meet threshold
            assert entity.confidence >= extractor.confidence_threshold

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_no_overlapping_entities(self, text):
        """
        Property: No overlapping entities.
        
        For any text, extracted entities should not overlap. Each character
        position should belong to at most one entity.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        entities = extractor.extract(text)
        
        # Check for overlaps
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Entities should not overlap
                assert e1.end_char <= e2.start_char or e2.end_char <= e1.start_char

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_entity_type_valid(self, text):
        """
        Property: Entity types are valid.
        
        For any extracted entity, the entity type should be a valid
        EntityType enum value.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        entities = extractor.extract(text)
        
        for entity in entities:
            # Entity type should be valid
            assert isinstance(entity.entity_type, EntityType)

    @given(
        text=text_with_entities(),
        entity_types=st.lists(
            st.sampled_from(list(EntityType)),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_entity_type_filtering(self, text, entity_types):
        """
        Property: Entity type filtering works correctly.
        
        For any text and entity type filter, all returned entities should
        have types in the specified filter list.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        entities = extractor.extract(text, entity_types=entity_types)
        
        # All entities should match filter
        for entity in entities:
            assert entity.entity_type in entity_types

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_batch_extraction_consistency(self, text):
        """
        Property: Batch extraction is consistent with single extraction.
        
        For any text, extracting it as part of a batch should produce
        the same results as extracting it individually.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        # Single extraction
        single_result = extractor.extract(text)
        
        # Batch extraction
        batch_results = extractor.extract_batch([text])
        batch_result = batch_results[0]
        
        # Results should match
        assert len(single_result) == len(batch_result)
        
        for e1, e2 in zip(single_result, batch_result):
            assert e1.text == e2.text
            assert e1.entity_type == e2.entity_type
            assert e1.start_char == e2.start_char
            assert e1.end_char == e2.end_char

    @given(
        text=text_with_entities(),
        threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_confidence_threshold_respected(self, text, threshold):
        """
        Property: Confidence threshold is respected.
        
        For any text and confidence threshold, all returned entities should
        have confidence >= threshold.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = EntityExtractor(
            use_rule_based=True,
            confidence_threshold=threshold
        )
        extractor.initialize()
        entities = extractor.extract(text)
        
        # All entities should meet threshold
        for entity in entities:
            assert entity.confidence >= threshold

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_entity_metadata_present(self, text):
        """
        Property: Entity metadata is present.
        
        For any extracted entity, metadata should be present and contain
        at least a 'source' field.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        entities = extractor.extract(text)
        
        for entity in entities:
            # Metadata should be present
            assert entity.metadata is not None
            assert isinstance(entity.metadata, dict)
            assert "source" in entity.metadata

    @given(text=text_with_entities())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_normalized_name_present(self, text):
        """
        Property: Normalized entity name is present.
        
        For any extracted entity, a normalized name should be present
        and should be a non-empty string.
        
        Feature: knowledge-graph, Property 1: Entity extraction consistency
        Validates: Requirements 2.1, 2.2
        """
        extractor = get_extractor()
        entities = extractor.extract(text)
        
        for entity in entities:
            # Normalized name should be present
            assert entity.normalized_name is not None
            assert isinstance(entity.normalized_name, str)
            assert len(entity.normalized_name) > 0
