"""
Data Transformer.

Provides data transformation capabilities including field mapping,
type conversion, value transformation, and custom scripts.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from src.sync.connectors.base import DataBatch, DataRecord

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Transformation type enumeration."""
    FIELD_MAPPING = "field_mapping"
    TYPE_CONVERSION = "type_conversion"
    VALUE_TRANSFORM = "value_transform"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    ENRICHMENT = "enrichment"
    NORMALIZATION = "normalization"
    CUSTOM = "custom"


@dataclass
class TransformationRule:
    """Data transformation rule."""
    name: str
    rule_type: TransformationType
    enabled: bool = True
    priority: int = 0
    source_field: Optional[str] = None
    target_field: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    error_handling: str = "skip"  # skip, fail, default

    def __post_init__(self):
        if not self.target_field:
            self.target_field = self.source_field


@dataclass
class TransformResult:
    """Result of transformation operation."""
    success: bool
    records_transformed: int = 0
    records_skipped: int = 0
    records_failed: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


class BaseTransformer(ABC):
    """Abstract base for transformers."""

    @abstractmethod
    def transform(
        self,
        record: DataRecord,
        rule: TransformationRule
    ) -> Optional[DataRecord]:
        """Transform a single record."""
        pass


class FieldMappingTransformer(BaseTransformer):
    """Field mapping transformer."""

    def transform(
        self,
        record: DataRecord,
        rule: TransformationRule
    ) -> Optional[DataRecord]:
        """Map field from source to target."""
        if not rule.source_field or rule.source_field not in record.data:
            return record

        value = record.data[rule.source_field]
        target = rule.target_field or rule.source_field

        record.data[target] = value

        # Remove source if different from target
        if target != rule.source_field and rule.config.get("remove_source", False):
            del record.data[rule.source_field]

        return record


class TypeConversionTransformer(BaseTransformer):
    """Type conversion transformer."""

    TYPE_CONVERTERS = {
        "string": str,
        "int": int,
        "float": float,
        "bool": lambda x: bool(x) if x not in ("false", "False", "0", "") else False,
        "datetime": lambda x: datetime.fromisoformat(x) if isinstance(x, str) else x,
        "date": lambda x: datetime.fromisoformat(x).date() if isinstance(x, str) else x,
        "json": lambda x: __import__("json").loads(x) if isinstance(x, str) else x,
    }

    def transform(
        self,
        record: DataRecord,
        rule: TransformationRule
    ) -> Optional[DataRecord]:
        """Convert field type."""
        if not rule.source_field or rule.source_field not in record.data:
            return record

        target_type = rule.config.get("target_type", "string")
        converter = self.TYPE_CONVERTERS.get(target_type, str)

        try:
            value = record.data[rule.source_field]
            if value is not None:
                record.data[rule.source_field] = converter(value)
        except (ValueError, TypeError) as e:
            if rule.error_handling == "fail":
                raise
            elif rule.error_handling == "default":
                record.data[rule.source_field] = rule.config.get("default_value")
            # else skip - keep original value

        return record


class ValueTransformTransformer(BaseTransformer):
    """Value transformation transformer."""

    def transform(
        self,
        record: DataRecord,
        rule: TransformationRule
    ) -> Optional[DataRecord]:
        """Transform field value."""
        if not rule.source_field or rule.source_field not in record.data:
            return record

        value = record.data[rule.source_field]
        transform_type = rule.config.get("transform")

        if transform_type == "uppercase":
            value = str(value).upper()
        elif transform_type == "lowercase":
            value = str(value).lower()
        elif transform_type == "trim":
            value = str(value).strip()
        elif transform_type == "replace":
            pattern = rule.config.get("pattern", "")
            replacement = rule.config.get("replacement", "")
            value = str(value).replace(pattern, replacement)
        elif transform_type == "regex_replace":
            pattern = rule.config.get("pattern", "")
            replacement = rule.config.get("replacement", "")
            value = re.sub(pattern, replacement, str(value))
        elif transform_type == "format":
            template = rule.config.get("template", "{value}")
            value = template.format(value=value, **record.data)
        elif transform_type == "concat":
            fields = rule.config.get("fields", [])
            separator = rule.config.get("separator", "")
            values = [str(record.data.get(f, "")) for f in fields]
            value = separator.join(values)
        elif transform_type == "split":
            separator = rule.config.get("separator", ",")
            index = rule.config.get("index", 0)
            parts = str(value).split(separator)
            value = parts[index] if index < len(parts) else ""
        elif transform_type == "hash":
            import hashlib
            algorithm = rule.config.get("algorithm", "sha256")
            hasher = getattr(hashlib, algorithm)()
            hasher.update(str(value).encode())
            value = hasher.hexdigest()
        elif transform_type == "mask":
            mask_char = rule.config.get("mask_char", "*")
            visible_start = rule.config.get("visible_start", 0)
            visible_end = rule.config.get("visible_end", 0)
            value_str = str(value)
            if len(value_str) > visible_start + visible_end:
                masked = (
                    value_str[:visible_start] +
                    mask_char * (len(value_str) - visible_start - visible_end) +
                    value_str[-visible_end:] if visible_end else ""
                )
                value = masked

        target = rule.target_field or rule.source_field
        record.data[target] = value

        return record


class EnrichmentTransformer(BaseTransformer):
    """Enrichment transformer for adding derived fields."""

    def transform(
        self,
        record: DataRecord,
        rule: TransformationRule
    ) -> Optional[DataRecord]:
        """Add enrichment fields."""
        enrichment_type = rule.config.get("enrichment_type")

        if enrichment_type == "timestamp":
            record.data[rule.target_field] = datetime.utcnow().isoformat()

        elif enrichment_type == "uuid":
            from uuid import uuid4
            record.data[rule.target_field] = str(uuid4())

        elif enrichment_type == "constant":
            record.data[rule.target_field] = rule.config.get("value")

        elif enrichment_type == "computed":
            expression = rule.config.get("expression", "")
            # Simple expression evaluation (use with caution)
            try:
                # Only allow safe operations
                safe_dict = {k: v for k, v in record.data.items() if isinstance(v, (int, float, str))}
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                record.data[rule.target_field] = result
            except Exception:
                pass

        elif enrichment_type == "lookup":
            lookup_table = rule.config.get("lookup_table", {})
            lookup_field = rule.config.get("lookup_field")
            if lookup_field and lookup_field in record.data:
                key = str(record.data[lookup_field])
                record.data[rule.target_field] = lookup_table.get(
                    key, rule.config.get("default_value")
                )

        return record


class NormalizationTransformer(BaseTransformer):
    """Normalization transformer for standardizing values."""

    def transform(
        self,
        record: DataRecord,
        rule: TransformationRule
    ) -> Optional[DataRecord]:
        """Normalize field values."""
        if not rule.source_field or rule.source_field not in record.data:
            return record

        value = record.data[rule.source_field]
        normalization_type = rule.config.get("normalization_type")

        if normalization_type == "phone":
            # Normalize phone number
            value = re.sub(r"[^\d+]", "", str(value))
            if not value.startswith("+"):
                country_code = rule.config.get("country_code", "+86")
                value = country_code + value

        elif normalization_type == "email":
            # Normalize email
            value = str(value).lower().strip()

        elif normalization_type == "date":
            # Normalize date format
            from datetime import datetime
            input_format = rule.config.get("input_format", "%Y-%m-%d")
            output_format = rule.config.get("output_format", "%Y-%m-%d")
            try:
                dt = datetime.strptime(str(value), input_format)
                value = dt.strftime(output_format)
            except ValueError:
                pass

        elif normalization_type == "boolean":
            # Normalize boolean values
            true_values = rule.config.get("true_values", ["true", "yes", "1", "on"])
            value = str(value).lower() in true_values

        elif normalization_type == "numeric":
            # Normalize numeric values
            try:
                value = float(re.sub(r"[^\d.-]", "", str(value)))
                if rule.config.get("as_integer", False):
                    value = int(value)
            except ValueError:
                value = rule.config.get("default_value", 0)

        target = rule.target_field or rule.source_field
        record.data[target] = value

        return record


class TransformationPipeline:
    """Pipeline for executing multiple transformations."""

    TRANSFORMERS = {
        TransformationType.FIELD_MAPPING: FieldMappingTransformer(),
        TransformationType.TYPE_CONVERSION: TypeConversionTransformer(),
        TransformationType.VALUE_TRANSFORM: ValueTransformTransformer(),
        TransformationType.ENRICHMENT: EnrichmentTransformer(),
        TransformationType.NORMALIZATION: NormalizationTransformer(),
    }

    def __init__(self, rules: Optional[List[TransformationRule]] = None):
        self._rules = rules or []
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: TransformationRule) -> None:
        """Add a transformation rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                del self._rules[i]
                return True
        return False

    def transform_record(self, record: DataRecord) -> Optional[DataRecord]:
        """Transform a single record through the pipeline."""
        for rule in self._rules:
            if not rule.enabled:
                continue

            transformer = self.TRANSFORMERS.get(rule.rule_type)
            if not transformer:
                continue

            try:
                record = transformer.transform(record, rule)
                if record is None:
                    return None  # Record filtered out
            except Exception as e:
                if rule.error_handling == "fail":
                    raise
                logger.warning(f"Transform error in {rule.name}: {e}")

        return record

    def transform_batch(self, batch: DataBatch) -> TransformResult:
        """Transform a batch of records."""
        import time
        start_time = time.time()

        result = TransformResult(success=True)
        transformed_records = []

        for record in batch.records:
            try:
                transformed = self.transform_record(record)
                if transformed:
                    transformed_records.append(transformed)
                    result.records_transformed += 1
                else:
                    result.records_skipped += 1
            except Exception as e:
                result.records_failed += 1
                result.errors.append({
                    "record_id": record.id,
                    "error": str(e)
                })

        batch.records = transformed_records
        result.duration_seconds = time.time() - start_time

        return result


class DataTransformer:
    """
    Data Transformer service.

    Provides high-level API for data transformation operations.
    """

    def __init__(self):
        self._pipelines: Dict[str, TransformationPipeline] = {}

    def create_pipeline(
        self,
        name: str,
        rules: List[TransformationRule]
    ) -> TransformationPipeline:
        """Create a named transformation pipeline."""
        pipeline = TransformationPipeline(rules)
        self._pipelines[name] = pipeline
        return pipeline

    def get_pipeline(self, name: str) -> Optional[TransformationPipeline]:
        """Get a pipeline by name."""
        return self._pipelines.get(name)

    def transform(
        self,
        batch: DataBatch,
        pipeline_name: Optional[str] = None,
        rules: Optional[List[TransformationRule]] = None
    ) -> TransformResult:
        """
        Transform a data batch.

        Args:
            batch: DataBatch to transform
            pipeline_name: Name of pre-configured pipeline
            rules: List of rules for ad-hoc transformation

        Returns:
            TransformResult with statistics
        """
        if pipeline_name:
            pipeline = self.get_pipeline(pipeline_name)
            if not pipeline:
                raise ValueError(f"Pipeline not found: {pipeline_name}")
        elif rules:
            pipeline = TransformationPipeline(rules)
        else:
            # No transformation
            return TransformResult(
                success=True,
                records_transformed=len(batch.records)
            )

        return pipeline.transform_batch(batch)


# Global transformer instance
data_transformer = DataTransformer()
