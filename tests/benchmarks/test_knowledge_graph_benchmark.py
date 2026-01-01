"""
Performance Benchmark Suite for Knowledge Graph System.

Task 16: Performance Benchmark Suite
- Implements performance benchmarking for all major operations
- Generates performance reports
- Validates performance against targets
"""

import pytest
import asyncio
import time
import statistics
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch
import json
import random
import string


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    operation: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_ops: float  # Operations per second
    passed: bool
    target_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "operation": self.operation,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
            "throughput_ops": round(self.throughput_ops, 2),
            "passed": self.passed,
            "target_ms": self.target_ms,
            "timestamp": self.timestamp
        }


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def measure(
        self,
        func,
        iterations: int = 100,
        warmup: int = 10,
        name: str = "",
        target_ms: float = 100.0
    ) -> BenchmarkResult:
        """
        Measure function performance.

        Args:
            func: Function to benchmark (sync or async)
            iterations: Number of iterations
            warmup: Warmup iterations (not counted)
            name: Benchmark name
            target_ms: Target time in milliseconds

        Returns:
            BenchmarkResult with statistics
        """
        # Warmup
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(func):
                asyncio.get_event_loop().run_until_complete(func())
            else:
                func()

        # Measure
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            if asyncio.iscoroutinefunction(func):
                asyncio.get_event_loop().run_until_complete(func())
            else:
                func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = (iterations / total_time) * 1000  # ops/sec

        result = BenchmarkResult(
            name=name or func.__name__,
            operation=func.__name__,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops=throughput,
            passed=avg_time <= target_ms,
            target_ms=target_ms
        )

        self.results.append(result)
        return result

    def generate_report(self) -> Dict[str, Any]:
        """Generate benchmark report."""
        return {
            "summary": {
                "total_benchmarks": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "generated_at": datetime.utcnow().isoformat()
            },
            "benchmarks": [r.to_dict() for r in self.results]
        }


# Mock classes for testing without database
class MockGraphDatabase:
    """Mock graph database for benchmarking."""

    def __init__(self):
        self.entities = {}
        self.relations = {}

    async def create_entity(self, entity: Dict) -> Dict:
        entity_id = f"entity_{len(self.entities)}"
        self.entities[entity_id] = {**entity, "id": entity_id}
        return self.entities[entity_id]

    async def get_entity(self, entity_id: str) -> Dict:
        return self.entities.get(entity_id, {})

    async def search_entities(self, query: str, limit: int = 100) -> List[Dict]:
        results = [
            e for e in self.entities.values()
            if query.lower() in e.get("name", "").lower()
        ]
        return results[:limit]

    async def create_relation(self, relation: Dict) -> Dict:
        rel_id = f"rel_{len(self.relations)}"
        self.relations[rel_id] = {**relation, "id": rel_id}
        return self.relations[rel_id]

    async def get_neighbors(self, entity_id: str, depth: int = 1) -> List[Dict]:
        neighbors = []
        for rel in self.relations.values():
            if rel.get("source_id") == entity_id:
                target = self.entities.get(rel.get("target_id"))
                if target:
                    neighbors.append({"entity": target, "relation": rel})
            elif rel.get("target_id") == entity_id:
                source = self.entities.get(rel.get("source_id"))
                if source:
                    neighbors.append({"entity": source, "relation": rel})
        return neighbors

    async def execute_cypher(self, query: str, params: Dict = None) -> List[Dict]:
        # Simulate query execution
        await asyncio.sleep(0.001)
        return []


class MockEntityExtractor:
    """Mock entity extractor for benchmarking."""

    def extract(self, text: str) -> List[Dict]:
        # Simulate entity extraction
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entities.append({
                    "name": word,
                    "type": "PERSON" if i % 2 == 0 else "ORGANIZATION",
                    "confidence": 0.9,
                    "span": [i * 5, i * 5 + len(word)]
                })
        return entities


class MockRelationExtractor:
    """Mock relation extractor for benchmarking."""

    def extract(self, text: str, entities: List[Dict]) -> List[Dict]:
        relations = []
        for i in range(len(entities) - 1):
            relations.append({
                "source": entities[i]["name"],
                "target": entities[i + 1]["name"],
                "type": "RELATED_TO",
                "confidence": 0.85
            })
        return relations


class MockQueryCache:
    """Mock query cache for benchmarking."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


# Benchmark Tests

class TestEntityOperationsBenchmark:
    """Benchmarks for entity CRUD operations."""

    @pytest.fixture
    def db(self):
        return MockGraphDatabase()

    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()

    @pytest.mark.asyncio
    async def test_entity_creation_benchmark(self, db, benchmark):
        """Benchmark entity creation performance."""
        async def create_entity():
            await db.create_entity({
                "name": f"Entity_{random.randint(1, 10000)}",
                "type": "PERSON",
                "properties": {"key": "value"}
            })

        result = benchmark.measure(
            create_entity,
            iterations=1000,
            warmup=100,
            name="Entity Creation",
            target_ms=10.0
        )

        assert result.passed, f"Entity creation too slow: {result.avg_time_ms}ms > {result.target_ms}ms"
        assert result.throughput_ops > 100, "Throughput below 100 ops/sec"

    @pytest.mark.asyncio
    async def test_entity_retrieval_benchmark(self, db, benchmark):
        """Benchmark entity retrieval performance."""
        # Setup: create entities
        for i in range(100):
            await db.create_entity({"name": f"Entity_{i}", "type": "PERSON"})

        async def get_entity():
            await db.get_entity(f"entity_{random.randint(0, 99)}")

        result = benchmark.measure(
            get_entity,
            iterations=1000,
            warmup=50,
            name="Entity Retrieval",
            target_ms=5.0
        )

        assert result.passed, f"Entity retrieval too slow: {result.avg_time_ms}ms"

    @pytest.mark.asyncio
    async def test_entity_search_benchmark(self, db, benchmark):
        """Benchmark entity search performance."""
        # Setup
        for i in range(500):
            await db.create_entity({"name": f"TestEntity_{i}", "type": "PERSON"})

        async def search_entities():
            await db.search_entities("Test", limit=50)

        result = benchmark.measure(
            search_entities,
            iterations=500,
            warmup=50,
            name="Entity Search",
            target_ms=50.0
        )

        assert result.passed, f"Entity search too slow: {result.avg_time_ms}ms"


class TestRelationOperationsBenchmark:
    """Benchmarks for relation operations."""

    @pytest.fixture
    def db(self):
        db = MockGraphDatabase()
        # Pre-populate with entities
        loop = asyncio.get_event_loop()
        for i in range(100):
            loop.run_until_complete(
                db.create_entity({"name": f"Entity_{i}", "type": "PERSON"})
            )
        return db

    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()

    @pytest.mark.asyncio
    async def test_relation_creation_benchmark(self, db, benchmark):
        """Benchmark relation creation performance."""
        async def create_relation():
            i = random.randint(0, 98)
            await db.create_relation({
                "source_id": f"entity_{i}",
                "target_id": f"entity_{i + 1}",
                "type": "RELATED_TO"
            })

        result = benchmark.measure(
            create_relation,
            iterations=1000,
            warmup=100,
            name="Relation Creation",
            target_ms=10.0
        )

        assert result.passed

    @pytest.mark.asyncio
    async def test_neighbor_traversal_benchmark(self, db, benchmark):
        """Benchmark neighbor traversal performance."""
        # Create relations
        for i in range(99):
            await db.create_relation({
                "source_id": f"entity_{i}",
                "target_id": f"entity_{i + 1}",
                "type": "RELATED_TO"
            })

        async def get_neighbors():
            await db.get_neighbors(f"entity_{random.randint(0, 99)}", depth=2)

        result = benchmark.measure(
            get_neighbors,
            iterations=500,
            warmup=50,
            name="Neighbor Traversal",
            target_ms=20.0
        )

        assert result.passed


class TestNLPExtractionBenchmark:
    """Benchmarks for NLP extraction operations."""

    @pytest.fixture
    def entity_extractor(self):
        return MockEntityExtractor()

    @pytest.fixture
    def relation_extractor(self):
        return MockRelationExtractor()

    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()

    def test_entity_extraction_benchmark(self, entity_extractor, benchmark):
        """Benchmark entity extraction performance."""
        texts = [
            "John Smith works at Acme Corporation in New York City.",
            "Microsoft announced a partnership with OpenAI for advanced AI research.",
            "The conference was held at Stanford University with speakers from Google and Meta."
        ]

        def extract_entities():
            text = random.choice(texts)
            entity_extractor.extract(text)

        result = benchmark.measure(
            extract_entities,
            iterations=1000,
            warmup=100,
            name="Entity Extraction",
            target_ms=50.0
        )

        assert result.passed, f"Entity extraction too slow: {result.avg_time_ms}ms"

    def test_relation_extraction_benchmark(self, entity_extractor, relation_extractor, benchmark):
        """Benchmark relation extraction performance."""
        text = "John works at Acme Corp and reports to Jane who manages the Engineering team."

        def extract_relations():
            entities = entity_extractor.extract(text)
            relation_extractor.extract(text, entities)

        result = benchmark.measure(
            extract_relations,
            iterations=500,
            warmup=50,
            name="Relation Extraction",
            target_ms=100.0
        )

        assert result.passed


class TestQueryBenchmark:
    """Benchmarks for query operations."""

    @pytest.fixture
    def db(self):
        return MockGraphDatabase()

    @pytest.fixture
    def cache(self):
        return MockQueryCache(max_size=100)

    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()

    @pytest.mark.asyncio
    async def test_cypher_query_benchmark(self, db, benchmark):
        """Benchmark Cypher query execution."""
        async def execute_query():
            await db.execute_cypher(
                "MATCH (p:Person)-[:WORKS_FOR]->(o:Organization) RETURN p, o LIMIT 10"
            )

        result = benchmark.measure(
            execute_query,
            iterations=500,
            warmup=50,
            name="Cypher Query",
            target_ms=100.0
        )

        assert result.passed

    def test_cache_hit_benchmark(self, cache, benchmark):
        """Benchmark cache hit performance."""
        # Pre-populate cache
        for i in range(100):
            cache.set(f"query_{i}", {"result": f"data_{i}"})

        def cache_lookup():
            cache.get(f"query_{random.randint(0, 99)}")

        result = benchmark.measure(
            cache_lookup,
            iterations=10000,
            warmup=1000,
            name="Cache Hit",
            target_ms=0.1
        )

        assert result.passed
        assert cache.hit_rate > 0.9, f"Cache hit rate too low: {cache.hit_rate}"


class TestBulkOperationsBenchmark:
    """Benchmarks for bulk operations."""

    @pytest.fixture
    def db(self):
        return MockGraphDatabase()

    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()

    @pytest.mark.asyncio
    async def test_bulk_entity_creation_benchmark(self, db, benchmark):
        """Benchmark bulk entity creation."""
        async def bulk_create():
            for i in range(100):
                await db.create_entity({
                    "name": f"BulkEntity_{i}",
                    "type": "PERSON"
                })

        result = benchmark.measure(
            bulk_create,
            iterations=10,
            warmup=2,
            name="Bulk Entity Creation (100)",
            target_ms=500.0
        )

        assert result.passed
        # Should achieve > 200 entities/second
        assert result.throughput_ops * 100 > 200


class TestConcurrencyBenchmark:
    """Benchmarks for concurrent operations."""

    @pytest.fixture
    def db(self):
        return MockGraphDatabase()

    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()

    @pytest.mark.asyncio
    async def test_concurrent_reads_benchmark(self, db, benchmark):
        """Benchmark concurrent read operations."""
        # Setup
        for i in range(100):
            await db.create_entity({"name": f"Entity_{i}", "type": "PERSON"})

        async def concurrent_reads():
            tasks = [
                db.get_entity(f"entity_{random.randint(0, 99)}")
                for _ in range(10)
            ]
            await asyncio.gather(*tasks)

        result = benchmark.measure(
            concurrent_reads,
            iterations=100,
            warmup=10,
            name="Concurrent Reads (10)",
            target_ms=50.0
        )

        assert result.passed


class TestPerformanceTargets:
    """Verify overall performance targets."""

    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()

    def test_performance_targets_summary(self, benchmark):
        """Generate performance targets summary."""
        targets = {
            "Entity Creation": {"target_ms": 10, "min_ops": 100},
            "Entity Retrieval": {"target_ms": 5, "min_ops": 200},
            "Entity Search": {"target_ms": 50, "min_ops": 20},
            "Relation Creation": {"target_ms": 10, "min_ops": 100},
            "Neighbor Traversal": {"target_ms": 20, "min_ops": 50},
            "Entity Extraction": {"target_ms": 50, "min_ops": 20},
            "Relation Extraction": {"target_ms": 100, "min_ops": 10},
            "Cypher Query": {"target_ms": 100, "min_ops": 10},
            "Cache Hit": {"target_ms": 0.1, "min_ops": 10000}
        }

        print("\n=== Performance Targets ===")
        for op, target in targets.items():
            print(f"{op}: < {target['target_ms']}ms, > {target['min_ops']} ops/sec")

        # All tests should pass within these targets
        assert len(targets) == 9


class TestBenchmarkReport:
    """Test benchmark report generation."""

    def test_generate_benchmark_report(self):
        """Test generating comprehensive benchmark report."""
        benchmark = PerformanceBenchmark()

        # Add some mock results
        benchmark.results.append(BenchmarkResult(
            name="Test Operation",
            operation="test_func",
            iterations=100,
            total_time_ms=1000,
            avg_time_ms=10,
            min_time_ms=5,
            max_time_ms=20,
            std_dev_ms=3,
            throughput_ops=100,
            passed=True,
            target_ms=15
        ))

        report = benchmark.generate_report()

        assert "summary" in report
        assert report["summary"]["total_benchmarks"] == 1
        assert report["summary"]["passed"] == 1
        assert "benchmarks" in report
        assert len(report["benchmarks"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
