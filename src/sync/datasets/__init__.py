"""
Datasets Module.

Provides industry dataset integration for data enrichment and noise dilution.
"""

from src.sync.datasets.dataset_discovery import (
    DatasetDiscoveryService,
    DatasetRegistry,
    DatasetMetadata,
    DatasetSearchQuery,
    DatasetQualityMetrics,
    DatasetSource,
    DatasetDomain,
    DatasetFormat,
    DatasetStatus,
    DatasetConnector,
    HuggingFaceConnector,
    LocalConnector,
    get_discovery_service,
)

from src.sync.datasets.data_dilution import (
    DataDilutionEngine,
    DataMerger,
    DataSample,
    DilutionConfig,
    DilutionMetrics,
    DilutionStrategy,
    SampleQuality,
    QualityAssessor,
    SampleSelector,
    get_dilution_engine,
    get_data_merger,
)

__all__ = [
    # Dataset Discovery
    "DatasetDiscoveryService",
    "DatasetRegistry",
    "DatasetMetadata",
    "DatasetSearchQuery",
    "DatasetQualityMetrics",
    "DatasetSource",
    "DatasetDomain",
    "DatasetFormat",
    "DatasetStatus",
    "DatasetConnector",
    "HuggingFaceConnector",
    "LocalConnector",
    "get_discovery_service",
    # Data Dilution
    "DataDilutionEngine",
    "DataMerger",
    "DataSample",
    "DilutionConfig",
    "DilutionMetrics",
    "DilutionStrategy",
    "SampleQuality",
    "QualityAssessor",
    "SampleSelector",
    "get_dilution_engine",
    "get_data_merger",
]
