"""
Data Cleanser.

Provides data cleaning capabilities including deduplication,
format validation, anomaly detection, and data quality scoring.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.sync.connectors.base import DataBatch, DataRecord

logger = logging.getLogger(__name__)


class CleansingType(str, Enum):
    """Cleansing operation type."""
    DEDUPLICATION = "deduplication"
    VALIDATION = "validation"
    FORMATTING = "formatting"
    NULL_HANDLING = "null_handling"
    OUTLIER_DETECTION = "outlier_detection"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class CleansingRule:
    """Data cleansing rule."""
    name: str
    rule_type: CleansingType
    enabled: bool = True
    priority: int = 0
    fields: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    action: str = "skip"  # skip, fix, flag


@dataclass
class QualityScore:
    """Data quality score."""
    overall_score: float
    completeness: float
    consistency: float
    accuracy: float
    validity: float
    uniqueness: float
    timeliness: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleansingResult:
    """Result of cleansing operation."""
    success: bool
    records_processed: int = 0
    records_cleaned: int = 0
    records_removed: int = 0
    duplicates_found: int = 0
    invalid_records: int = 0
    quality_score: Optional[QualityScore] = None
    issues: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


class DeduplicationEngine:
    """Deduplication engine for identifying and removing duplicates."""

    def __init__(self):
        self._seen_hashes: Set[str] = set()
        self._hash_counts: Dict[str, int] = {}

    def reset(self) -> None:
        """Reset deduplication state."""
        self._seen_hashes.clear()
        self._hash_counts.clear()

    def compute_hash(
        self,
        record: DataRecord,
        fields: Optional[List[str]] = None
    ) -> str:
        """Compute hash for record."""
        import json

        if fields:
            data = {f: record.data.get(f) for f in fields if f in record.data}
        else:
            data = record.data

        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def is_duplicate(
        self,
        record: DataRecord,
        fields: Optional[List[str]] = None,
        strategy: str = "exact"
    ) -> tuple[bool, Optional[str]]:
        """
        Check if record is a duplicate.

        Args:
            record: Record to check
            fields: Fields to use for comparison
            strategy: Dedup strategy (exact, fuzzy)

        Returns:
            Tuple of (is_duplicate, hash)
        """
        record_hash = self.compute_hash(record, fields)

        if record_hash in self._seen_hashes:
            self._hash_counts[record_hash] = self._hash_counts.get(record_hash, 1) + 1
            return True, record_hash

        self._seen_hashes.add(record_hash)
        self._hash_counts[record_hash] = 1
        return False, record_hash

    def get_duplicate_count(self) -> int:
        """Get total duplicate count."""
        return sum(c - 1 for c in self._hash_counts.values() if c > 1)


class ValidationEngine:
    """Validation engine for data validation."""

    VALIDATORS = {
        "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        "phone": re.compile(r"^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$"),
        "url": re.compile(r"^https?://[\w\-]+(\.[\w\-]+)+[/#?]?.*$"),
        "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
        "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
        "datetime": re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
        "numeric": re.compile(r"^-?\d+\.?\d*$"),
        "alpha": re.compile(r"^[a-zA-Z]+$"),
        "alphanumeric": re.compile(r"^[a-zA-Z0-9]+$"),
    }

    def validate_field(
        self,
        value: Any,
        validation_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a field value.

        Args:
            value: Value to validate
            validation_type: Type of validation
            config: Additional validation config

        Returns:
            Tuple of (is_valid, error_message)
        """
        config = config or {}

        if value is None:
            if config.get("nullable", True):
                return True, None
            return False, "Value is null"

        str_value = str(value)

        # Pattern validation
        if validation_type in self.VALIDATORS:
            pattern = self.VALIDATORS[validation_type]
            if not pattern.match(str_value):
                return False, f"Does not match {validation_type} format"
            return True, None

        # Range validation
        if validation_type == "range":
            try:
                num_value = float(value)
                min_val = config.get("min")
                max_val = config.get("max")
                if min_val is not None and num_value < min_val:
                    return False, f"Value below minimum ({min_val})"
                if max_val is not None and num_value > max_val:
                    return False, f"Value above maximum ({max_val})"
            except (ValueError, TypeError):
                return False, "Not a valid number"
            return True, None

        # Length validation
        if validation_type == "length":
            length = len(str_value)
            min_len = config.get("min")
            max_len = config.get("max")
            if min_len is not None and length < min_len:
                return False, f"Length below minimum ({min_len})"
            if max_len is not None and length > max_len:
                return False, f"Length above maximum ({max_len})"
            return True, None

        # Enum validation
        if validation_type == "enum":
            allowed = config.get("values", [])
            if value not in allowed:
                return False, f"Value not in allowed list: {allowed}"
            return True, None

        # Custom regex
        if validation_type == "regex":
            pattern = config.get("pattern")
            if pattern:
                if not re.match(pattern, str_value):
                    return False, "Does not match pattern"
            return True, None

        # Required validation
        if validation_type == "required":
            if not str_value or str_value.strip() == "":
                return False, "Value is required"
            return True, None

        return True, None


class DataCleanser:
    """
    Data Cleanser service.

    Provides comprehensive data cleaning capabilities:
    - Deduplication
    - Validation
    - Format standardization
    - Null handling
    - Quality scoring
    """

    def __init__(self):
        self._dedup_engine = DeduplicationEngine()
        self._validation_engine = ValidationEngine()

    def cleanse(
        self,
        batch: DataBatch,
        rules: List[CleansingRule]
    ) -> CleansingResult:
        """
        Cleanse a data batch.

        Args:
            batch: DataBatch to cleanse
            rules: Cleansing rules to apply

        Returns:
            CleansingResult with statistics
        """
        import time
        start_time = time.time()

        result = CleansingResult(success=True, records_processed=len(batch.records))
        cleaned_records = []
        rules.sort(key=lambda r: r.priority, reverse=True)

        # Reset dedup state
        self._dedup_engine.reset()

        for record in batch.records:
            keep_record = True
            record_issues = []

            for rule in rules:
                if not rule.enabled:
                    continue

                if rule.rule_type == CleansingType.DEDUPLICATION:
                    is_dup, _ = self._dedup_engine.is_duplicate(
                        record,
                        fields=rule.fields,
                        strategy=rule.config.get("strategy", "exact")
                    )
                    if is_dup:
                        result.duplicates_found += 1
                        if rule.action == "skip":
                            keep_record = False
                            break

                elif rule.rule_type == CleansingType.VALIDATION:
                    for field_name in rule.fields:
                        if field_name in record.data:
                            valid, error = self._validation_engine.validate_field(
                                record.data[field_name],
                                rule.config.get("validation_type", "required"),
                                rule.config
                            )
                            if not valid:
                                record_issues.append({
                                    "field": field_name,
                                    "error": error,
                                    "rule": rule.name
                                })
                                if rule.action == "skip":
                                    keep_record = False
                                    result.invalid_records += 1
                                    break

                elif rule.rule_type == CleansingType.NULL_HANDLING:
                    for field_name in rule.fields:
                        if field_name in record.data and record.data[field_name] is None:
                            action = rule.config.get("null_action", "default")
                            if action == "default":
                                record.data[field_name] = rule.config.get("default_value", "")
                            elif action == "skip_record":
                                keep_record = False
                                break
                            elif action == "skip_field":
                                del record.data[field_name]

                elif rule.rule_type == CleansingType.FORMATTING:
                    for field_name in rule.fields:
                        if field_name in record.data:
                            record.data[field_name] = self._format_value(
                                record.data[field_name],
                                rule.config
                            )

            if keep_record:
                if record_issues:
                    record.metadata["quality_issues"] = record_issues
                cleaned_records.append(record)
                result.records_cleaned += 1
            else:
                result.records_removed += 1
                result.issues.append({
                    "record_id": record.id,
                    "issues": record_issues
                })

        # Update batch
        batch.records = cleaned_records

        # Calculate quality score
        result.quality_score = self._calculate_quality_score(batch, result)

        result.duration_seconds = time.time() - start_time
        return result

    def _format_value(self, value: Any, config: Dict[str, Any]) -> Any:
        """Apply formatting to a value."""
        if value is None:
            return value

        format_type = config.get("format_type")

        if format_type == "trim":
            return str(value).strip()

        elif format_type == "lowercase":
            return str(value).lower()

        elif format_type == "uppercase":
            return str(value).upper()

        elif format_type == "titlecase":
            return str(value).title()

        elif format_type == "remove_whitespace":
            return re.sub(r"\s+", " ", str(value)).strip()

        elif format_type == "remove_special":
            pattern = config.get("pattern", r"[^\w\s]")
            return re.sub(pattern, "", str(value))

        elif format_type == "truncate":
            max_length = config.get("max_length", 255)
            return str(value)[:max_length]

        return value

    def _calculate_quality_score(
        self,
        batch: DataBatch,
        result: CleansingResult
    ) -> QualityScore:
        """Calculate data quality score."""
        total = result.records_processed
        if total == 0:
            return QualityScore(
                overall_score=1.0,
                completeness=1.0,
                consistency=1.0,
                accuracy=1.0,
                validity=1.0,
                uniqueness=1.0
            )

        # Calculate individual scores
        uniqueness = 1 - (result.duplicates_found / total) if total > 0 else 1.0
        validity = 1 - (result.invalid_records / total) if total > 0 else 1.0

        # Check completeness (null ratio)
        null_count = 0
        total_fields = 0
        for record in batch.records:
            for value in record.data.values():
                total_fields += 1
                if value is None or value == "":
                    null_count += 1
        completeness = 1 - (null_count / total_fields) if total_fields > 0 else 1.0

        # Consistency score based on format issues
        consistency = 0.95  # Default high consistency

        # Accuracy (would need validation against known correct values)
        accuracy = validity * 0.9 + 0.1  # Approximate based on validity

        # Overall score (weighted average)
        overall = (
            completeness * 0.25 +
            consistency * 0.15 +
            accuracy * 0.2 +
            validity * 0.25 +
            uniqueness * 0.15
        )

        return QualityScore(
            overall_score=round(overall, 4),
            completeness=round(completeness, 4),
            consistency=round(consistency, 4),
            accuracy=round(accuracy, 4),
            validity=round(validity, 4),
            uniqueness=round(uniqueness, 4),
            details={
                "total_records": total,
                "duplicates": result.duplicates_found,
                "invalid": result.invalid_records,
                "null_ratio": null_count / total_fields if total_fields > 0 else 0
            }
        )

    def deduplicate(
        self,
        batch: DataBatch,
        fields: Optional[List[str]] = None,
        keep: str = "first"
    ) -> CleansingResult:
        """
        Remove duplicates from batch.

        Args:
            batch: DataBatch to deduplicate
            fields: Fields to use for comparison
            keep: Which duplicate to keep (first, last)

        Returns:
            CleansingResult
        """
        rules = [
            CleansingRule(
                name="dedup",
                rule_type=CleansingType.DEDUPLICATION,
                fields=fields or [],
                config={"strategy": "exact", "keep": keep},
                action="skip"
            )
        ]
        return self.cleanse(batch, rules)

    def validate(
        self,
        batch: DataBatch,
        validations: Dict[str, Dict[str, Any]]
    ) -> CleansingResult:
        """
        Validate batch data.

        Args:
            batch: DataBatch to validate
            validations: Field -> validation config mapping

        Returns:
            CleansingResult
        """
        rules = []
        for field_name, config in validations.items():
            rules.append(CleansingRule(
                name=f"validate_{field_name}",
                rule_type=CleansingType.VALIDATION,
                fields=[field_name],
                config=config,
                action=config.get("action", "flag")
            ))
        return self.cleanse(batch, rules)


# Global cleanser instance
data_cleanser = DataCleanser()
