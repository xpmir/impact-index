#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ir-datasets",
#     "ir-measures",
#     "torch",
#     "transformers",
#     "numpy",
#     "tqdm",
#     "psutil",
#     "impact-index",
# ]
# ///
"""
SPLADE Indexing and Search Benchmark

This example demonstrates:
1. Loading a SPLADE model from HuggingFace
2. Encoding documents and queries using SPLADE
3. Building impact-index, BMP, and compressed indices
4. Searching with various algorithms (WAND, MaxScore, BMP)
5. Measuring performance (CPU time, wall time, index size)
6. Computing IR metrics (MRR, NDCG, Recall)

Usage:
    uv run examples/splade_benchmark.py [--dataset msmarco-passage/dev/small] [--model naver/splade-cocondenser-ensembledistil]

Or with pip-installed dependencies:
    python examples/splade_benchmark.py [options]

Compressed index examples:
    # Regular compressed index with 16-bit quantization
    --compressed-index 'nbits=16 block-size=128'

    # Split compressed index (for MaxScore optimization)
    --compressed-index 'split=0.9 nbits=8'

    # Multiple configurations
    --compressed-index 'nbits=8' --compressed-index 'split=0.8,0.95 nbits=16'

Approximate search grids (each build searched at every operating point):
    --seismic '' --seismic 'n-postings=3500' --seismic-search 'query-cut=10 heap-factor=0.7'
    --bmp-search 'alpha=1 beta=1' --bmp-search 'alpha=0.9 beta=0.8'

Reusing an encoded index across runs (encoding MS MARCO takes hours):
    --index-dir /path/to/encoded-index [--mmap-index]
Every run is saved as a TREC run file under OUTPUT_DIR/runs/.
"""

import argparse
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Import the index libraries
import impact_index
from impact_index import (
    BitPackingCompressor,
    BmpSearcher,
    QuantizedBitPackedCompressor,
    CompressionTransform,
    SplitIndexTransform,
)

# Seismic is an opt-in build feature (`--features seismic`)
SeismicSearcher = getattr(impact_index, "SeismicSearcher", None)


@dataclass
class CompressedIndexConfig:
    """Configuration for a compressed index."""

    nbits: int = 8
    block_size: int = 128
    split_quantiles: Optional[List[float]] = None

    @classmethod
    def parse(cls, config_str: str) -> "CompressedIndexConfig":
        """Parse a configuration string like 'split=0.8,0.95 nbits=16 block-size=128'."""
        config = cls()

        # Parse key=value pairs
        for part in config_str.split():
            if "=" not in part:
                raise ValueError(
                    f"Invalid config part: {part}. Expected key=value format."
                )
            key, value = part.split("=", 1)
            key = key.strip().lower().replace("-", "_")

            if key == "nbits":
                config.nbits = int(value)
            elif key == "block_size":
                config.block_size = int(value)
            elif key == "split":
                config.split_quantiles = [float(q) for q in value.split(",")]
            else:
                raise ValueError(f"Unknown config key: {key}")

        return config

    def get_dir_name(self) -> str:
        """Generate a unique directory name based on configuration."""
        parts = []
        if self.split_quantiles:
            quantiles_str = "_".join(
                f"{q:.2f}".replace(".", "") for q in self.split_quantiles
            )
            parts.append(f"split_{quantiles_str}")
        parts.append(f"nb{self.nbits}")
        parts.append(f"bs{self.block_size}")
        return "_".join(parts)

    def get_display_name(self) -> str:
        """Get a human-readable name for display."""
        if self.split_quantiles:
            return f"Split({','.join(f'{q:.2f}' for q in self.split_quantiles)}) nb={self.nbits} bs={self.block_size}"
        return f"Compressed nb={self.nbits} bs={self.block_size}"

    def is_split(self) -> bool:
        """Check if this is a split index configuration."""
        return self.split_quantiles is not None


def get_best_device() -> str:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB using psutil."""
    import psutil

    return psutil.Process().memory_info().rss / 1024 / 1024


def get_current_rss_mb() -> float:
    """Get current RSS (Resident Set Size) in MB using psutil."""
    import psutil

    return psutil.Process().memory_info().rss / 1024 / 1024


@dataclass
class PeakMemoryResult:
    """Result of peak memory measurement."""

    peak_mb: float
    start_mb: float
    end_mb: float
    samples: int

    @property
    def delta_mb(self) -> float:
        return self.peak_mb - self.start_mb

    def to_dict(self) -> dict:
        return {
            "peak_mb": self.peak_mb,
            "start_mb": self.start_mb,
            "end_mb": self.end_mb,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PeakMemoryResult":
        return cls(
            peak_mb=d["peak_mb"],
            start_mb=d["start_mb"],
            end_mb=d["end_mb"],
            samples=d.get("samples", 0),
        )


@dataclass
class BuildStats:
    """Statistics from building an index."""

    wall_time: float
    cpu_time: float
    memory: Optional[PeakMemoryResult]

    def to_dict(self) -> dict:
        return {
            "wall_time": self.wall_time,
            "cpu_time": self.cpu_time,
            "memory": self.memory.to_dict() if self.memory else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BuildStats":
        return cls(
            wall_time=d["wall_time"],
            cpu_time=d["cpu_time"],
            memory=PeakMemoryResult.from_dict(d["memory"]) if d.get("memory") else None,
        )


class PeakMemoryTracker:
    """Track peak memory usage during an operation using background sampling."""

    def __init__(self, interval_ms: int = 50):
        self.interval_ms = interval_ms
        self.peak_mb = 0.0
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.samples = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample_loop(self):
        """Background thread that samples memory usage."""
        while not self._stop_event.wait(self.interval_ms / 1000.0):
            current = get_current_rss_mb()
            self.peak_mb = max(self.peak_mb, current)
            self.samples += 1

    def start(self):
        """Start tracking memory."""
        self.start_mb = get_current_rss_mb()
        self.peak_mb = self.start_mb
        self.samples = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> PeakMemoryResult:
        """Stop tracking and return results."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.end_mb = get_current_rss_mb()
        # Final sample
        self.peak_mb = max(self.peak_mb, self.end_mb)
        return PeakMemoryResult(
            peak_mb=self.peak_mb,
            start_mb=self.start_mb,
            end_mb=self.end_mb,
            samples=self.samples,
        )


@contextmanager
def track_peak_memory(interval_ms: int = 50):
    """Context manager to track peak memory usage during an operation.

    Usage:
        with track_peak_memory() as tracker:
            # ... do work ...
        print(f"Peak memory: {tracker.result.peak_mb:.2f} MB")
    """
    tracker = PeakMemoryTracker(interval_ms=interval_ms)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.result = tracker.stop()


def get_file_size_mb(path: Path) -> float:
    """Get total size of files in a directory in MB."""
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1024 / 1024


class IndexBuilder:
    """Base class for index builders with caching and statistics tracking."""

    STATS_FILE = "build_stats.json"

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.build_stats: Optional[BuildStats] = None

    @property
    def done_file(self) -> Path:
        """Path to the marker file indicating the build is complete."""
        if self.output_path.suffix:
            return self.output_path.with_suffix(".done")
        return self.output_path / ".done"

    @property
    def stats_file(self) -> Path:
        """Path to the build statistics file."""
        if self.output_path.suffix:
            return self.output_path.with_suffix(".stats.json")
        return self.output_path / self.STATS_FILE

    def is_built(self) -> bool:
        """Check if the index has already been built."""
        return self.done_file.exists()

    def save_stats(self, stats: BuildStats):
        """Save build statistics to disk."""
        with open(self.stats_file, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        self.build_stats = stats

    def load_stats(self) -> Optional[BuildStats]:
        """Load build statistics from disk if available."""
        if self.stats_file.exists():
            with open(self.stats_file, "r") as f:
                self.build_stats = BuildStats.from_dict(json.load(f))
                return self.build_stats
        return None

    def get_size_mb(self) -> float:
        """Get the index size in MB."""
        return get_file_size_mb(self.output_path)

    def display_name(self) -> str:
        """Human-readable name for display."""
        raise NotImplementedError

    def _do_build(self, source_index: impact_index.Index):
        """Perform the actual build. Must be implemented by subclasses."""
        raise NotImplementedError

    def _load_index(self):
        """Load the built index. Must be implemented by subclasses."""
        raise NotImplementedError

    def build(self, source_index: impact_index.Index, load: bool = True):
        """Build the index with caching, timing, and memory tracking.

        Returns the loaded index, or None when `load` is False."""
        print(f"\n=== Building {self.display_name()} ===")

        if self.is_built():
            print("  Index already built, loading from disk...")
            stats = self.load_stats()
            if stats:
                print(
                    f"  (Previous build: {stats.wall_time:.2f}s wall, "
                    f"{stats.cpu_time:.2f}s CPU",
                    end="",
                )
                if stats.memory:
                    print(f", peak mem: {stats.memory.peak_mb:.2f} MB", end="")
                print(")")
            print(f"  Index size: {self.get_size_mb():.2f} MB")
            return self._load_index() if load else None

        # Create output directory if needed
        if not self.output_path.suffix:
            self.output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        start_cpu = time.process_time()

        with track_peak_memory() as mem_tracker:
            self._do_build(source_index)

        end_time = time.time()
        end_cpu = time.process_time()
        mem_result = mem_tracker.result

        # Save statistics
        stats = BuildStats(
            wall_time=end_time - start_time,
            cpu_time=end_cpu - start_cpu,
            memory=mem_result,
        )
        self.save_stats(stats)

        # Mark as done
        self.done_file.touch()

        print(f"  Wall time: {stats.wall_time:.2f}s")
        print(f"  CPU time: {stats.cpu_time:.2f}s")
        print(
            f"  Peak memory: {mem_result.peak_mb:.2f} MB (delta: {mem_result.delta_mb:.2f} MB)"
        )
        print(f"  Index size: {self.get_size_mb():.2f} MB")

        # Load and return the built index
        return self._load_index() if load else None


class BmpIndexBuilder(IndexBuilder):
    """Builder for BMP index."""

    def __init__(
        self,
        output_path: Path,
        bsize: int = 64,
        compress_range: bool = True,
        use_streaming: bool = True,
    ):
        super().__init__(output_path)
        self.bsize = bsize
        self.compress_range = compress_range
        self.use_streaming = use_streaming

    def display_name(self) -> str:
        method = "streaming" if self.use_streaming else "legacy"
        return f"BMP Index ({method})"

    def _do_build(self, source_index: impact_index.Index):
        if self.use_streaming:
            source_index.to_bmp_streaming(
                str(self.output_path), self.bsize, self.compress_range
            )
        else:
            source_index.to_bmp(str(self.output_path), self.bsize, self.compress_range)

    def _load_index(self) -> BmpSearcher:
        return BmpSearcher(str(self.output_path))

    def build(
        self, source_index: impact_index.Index, load: bool = True
    ) -> Optional[BmpSearcher]:
        return super().build(source_index, load=load)


class CompressedIndexBuilder(IndexBuilder):
    """Builder for compressed (and optionally split) index."""

    def __init__(self, output_path: Path, config: CompressedIndexConfig):
        super().__init__(output_path)
        self.config = config

    def display_name(self) -> str:
        return self.config.get_display_name()

    def _do_build(self, source_index: impact_index.Index):
        if self.config.is_split():
            # Split requires the full transform pipeline
            doc_ids_compressor = BitPackingCompressor()
            impact_compressor = QuantizedBitPackedCompressor(self.config.nbits)
            compression_transform = CompressionTransform(
                self.config.block_size, doc_ids_compressor, impact_compressor
            )
            transform = SplitIndexTransform(
                self.config.split_quantiles, compression_transform
            )
            transform.process(str(self.output_path), source_index)
        else:
            # Simple compression: use Index.compress()
            source_index.compress(
                str(self.output_path),
                block_size=self.config.block_size,
                nbits=self.config.nbits,
            )

    def _load_index(self) -> impact_index.Index:
        return impact_index.Index.load(str(self.output_path), in_memory=True)


@dataclass
class SeismicBuildConfig:
    """Build parameters of a Seismic index (see `Index.to_seismic`)."""

    n_postings: int = 6000
    centroid_fraction: float = 0.1
    summary_energy: float = 0.4
    max_fraction: float = 1.5
    knn: int = 0

    @classmethod
    def parse(cls, config_str: str) -> "SeismicBuildConfig":
        """Parse e.g. 'n-postings=4000 summary-energy=0.4'."""
        config = cls()
        for part in config_str.split():
            key, value = part.split("=", 1)
            key = key.strip().lower().replace("-", "_")
            if key not in cls.__dataclass_fields__:
                raise ValueError(f"Unknown Seismic build key: {key}")
            setattr(config, key, type(getattr(config, key))(value))
        return config

    def get_dir_name(self) -> str:
        return (
            f"seismic_np{self.n_postings}_cf{self.centroid_fraction}"
            f"_se{self.summary_energy}_mf{self.max_fraction}_knn{self.knn}"
        )

    def get_display_name(self) -> str:
        name = f"Seismic np={self.n_postings} se={self.summary_energy} cf={self.centroid_fraction}"
        return name + (f" knn={self.knn}" if self.knn else "")


@dataclass
class SeismicSearchConfig:
    """Query-time parameters of a Seismic search."""

    query_cut: int = 10
    heap_factor: float = 0.7
    n_knn: int = 0

    @classmethod
    def parse(cls, config_str: str) -> "SeismicSearchConfig":
        """Parse e.g. 'query-cut=10 heap-factor=0.7'."""
        config = cls()
        for part in config_str.split():
            key, value = part.split("=", 1)
            key = key.strip().lower().replace("-", "_")
            if key not in cls.__dataclass_fields__:
                raise ValueError(f"Unknown Seismic search key: {key}")
            setattr(config, key, type(getattr(config, key))(value))
        return config

    def get_display_name(self) -> str:
        name = f"qc={self.query_cut} hf={self.heap_factor}"
        return name + (f" knn={self.n_knn}" if self.n_knn else "")


@dataclass
class BmpSearchConfig:
    """Query-time parameters of a BMP search (alpha = beta = 1 is safe)."""

    alpha: float = 1.0  # stop when threshold > alpha * block upper bound
    beta: float = 1.0  # fraction of the (heaviest) query terms kept

    @classmethod
    def parse(cls, config_str: str) -> "BmpSearchConfig":
        """Parse e.g. 'alpha=0.9 beta=0.8'."""
        config = cls()
        for part in config_str.split():
            key, value = part.split("=", 1)
            key = key.strip().lower()
            if key not in cls.__dataclass_fields__:
                raise ValueError(f"Unknown BMP search key: {key}")
            setattr(config, key, float(value))
        return config

    def get_display_name(self) -> str:
        return f"BMP a={self.alpha} b={self.beta}"


class SeismicIndexBuilder(IndexBuilder):
    """Builder for a Seismic index."""

    def __init__(self, output_path: Path, config: SeismicBuildConfig):
        super().__init__(output_path)
        self.config = config

    def display_name(self) -> str:
        return self.config.get_display_name()

    def _do_build(self, source_index: impact_index.Index):
        c = self.config
        source_index.to_seismic(
            str(self.output_path),
            n_postings=c.n_postings,
            centroid_fraction=c.centroid_fraction,
            summary_energy=c.summary_energy,
            max_fraction=c.max_fraction,
            knn=c.knn,
        )

    def _load_index(self):
        return SeismicSearcher(str(self.output_path))


class SpladeEncoder:
    """SPLADE sparse encoder using HuggingFace transformers."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = None,
        multi_gpu: bool = False,
        fp16: bool = False,
    ):
        self.device = device or get_best_device()
        # fp16 autocast for the transformer (CUDA only); pooling stays in fp32
        self.fp16 = fp16 and self.device == "cuda"
        print(f"Loading SPLADE model '{model_name}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Check for multi-GPU setup (only if explicitly enabled)
        self.multi_gpu = False
        self.models = []  # List of (model, device) for multi-GPU

        if multi_gpu and self.device == "cuda" and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            print(f"  Loading model on {num_gpus} GPUs for parallel inference...")
            # Load a separate model copy on each GPU for true parallelism
            for gpu_id in range(num_gpus):
                device = f"cuda:{gpu_id}"
                model = AutoModelForMaskedLM.from_pretrained(model_name)
                model.to(device)
                model.eval()
                self.models.append((model, device))
                print(f"    GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            self.multi_gpu = True
            self.model = self.models[0][0]  # Keep reference for compatibility
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            if self.device == "cuda":
                print("  Using single GPU (CUDA device 0)")
            self.model.to(self.device)
            self.model.eval()

        # Get vocabulary for term mapping
        self.vocab = self.tokenizer.get_vocab()
        self.id2token = {v: k for k, v in self.vocab.items()}

    def _run_model_on_device(
        self, inputs_chunk: dict, model, device: str
    ) -> List[Dict[int, float]]:
        """Run model inference on pre-tokenized inputs on a specific device."""
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs_chunk.items()}

        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=self.fp16
        ):
            outputs = model(**inputs)
            logits = outputs.logits

            # SPLADE aggregation: max_t log1p(relu(x_t)) * mask_t. Since
            # log1p(relu(.)) is non-decreasing (and >= 0), this equals
            # log1p(relu(max_t x_t)) with padded positions zeroed, which
            # avoids materializing (batch, seq_len, vocab) temporaries.
            # Shape: (batch, seq_len, vocab_size) -> (batch, vocab_size)
            logits.masked_fill_(inputs["attention_mask"].unsqueeze(-1) == 0, 0)
            weights = torch.log1p(torch.relu(logits.max(dim=1).values.float()))

        # One device->host copy per batch (a per-term .item() would force a
        # GPU sync for every non-zero weight)
        weights = weights.cpu().numpy()
        results = []
        for row in weights:
            nonzero = np.flatnonzero(row > 0)
            results.append(dict(zip(nonzero.tolist(), row[nonzero].tolist())))

        return results

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 256) -> List[Dict[int, float]]:
        """
        Encode texts into sparse SPLADE representations.

        Returns list of dicts mapping term_id -> weight.
        """
        # Tokenize all texts upfront on CPU (avoids GIL contention in threads)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        if not self.multi_gpu:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return self._run_model_on_device(inputs, self.model, self.device)

        # Multi-GPU: distribute pre-tokenized inputs across GPUs using threads
        import concurrent.futures

        num_gpus = len(self.models)
        batch_size = inputs["input_ids"].shape[0]
        chunk_size = (batch_size + num_gpus - 1) // num_gpus

        # Split tokenized inputs into chunks
        input_chunks = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = min(start + chunk_size, batch_size)
            if start < batch_size:
                chunk = {k: v[start:end] for k, v in inputs.items()}
                input_chunks.append(chunk)

        results = [None] * len(input_chunks)

        def run_on_gpu(idx, inputs_chunk, model, device):
            if inputs_chunk["input_ids"].shape[0] == 0:
                return idx, []
            return idx, self._run_model_on_device(inputs_chunk, model, device)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, chunk in enumerate(input_chunks):
                if i < len(self.models):
                    model, device = self.models[i]
                    futures.append(executor.submit(run_on_gpu, i, chunk, model, device))

            for future in concurrent.futures.as_completed(futures):
                idx, chunk_results = future.result()
                results[idx] = chunk_results

        # Flatten results maintaining order
        return [item for sublist in results if sublist for item in sublist]

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        desc: str = "Encoding",
        unit: str = "item",
    ) -> List[Dict[int, float]]:
        """Encode texts in batches with progress bar (shows individual items, not batches).

        In multi-GPU mode, batch_size is per GPU, so total batch = batch_size * num_gpus.
        """
        # In multi-GPU mode, batch_size is per GPU
        effective_batch_size = (
            batch_size * len(self.models) if self.multi_gpu else batch_size
        )

        results = []
        with tqdm(total=len(texts), desc=desc, unit=unit) as pbar:
            for i in range(0, len(texts), effective_batch_size):
                batch = texts[i : i + effective_batch_size]
                batch_results = self.encode(batch, max_length)
                results.extend(batch_results)
                pbar.update(len(batch))
        return results


def load_queries_and_qrels(dataset_name: str, max_queries: int = 0):
    """Load queries and qrels from dataset (lightweight, kept in memory).

    Args:
        dataset_name: ir_datasets dataset name
        max_queries: Maximum number of queries to load (0 = all)
    """
    import ir_datasets

    print(f"Loading dataset '{dataset_name}'...")
    dataset = ir_datasets.load(dataset_name)

    # Load queries
    queries = []
    query_ids = []
    query_ids_set = set()
    print("Loading queries...")
    for i, query in enumerate(dataset.queries_iter()):
        if max_queries > 0 and i >= max_queries:
            break
        queries.append(query.text if hasattr(query, "text") else str(query))
        query_ids.append(query.query_id)
        query_ids_set.add(query.query_id)

    print(
        f"  Loaded {len(queries)} queries"
        + (f" (limited from max_queries={max_queries})" if max_queries > 0 else "")
    )

    # Load qrels (only for loaded queries)
    qrels = defaultdict(dict)
    print("Loading qrels...")
    total_qrels = 0
    for qrel in dataset.qrels_iter():
        if qrel.query_id in query_ids_set:
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            total_qrels += 1

    print(f"  Loaded {total_qrels} qrels for {len(qrels)} queries")

    return dataset, queries, query_ids, qrels


def doc_id_mapping_path(index_dir: Path) -> Path:
    """Path to store doc_id mapping for evaluation."""
    return index_dir / "doc_ids.json"


def _document_prefetcher(
    dataset, max_docs: int, batch_size: int, queue, resume_from: int
):
    """Background thread to prefetch document batches.

    Sends tuples of (batch_doc_ids, batch_texts) to the queue.
    Sends None when done.
    """
    batch_texts = []
    batch_doc_ids = []
    skipped_doc_ids = []
    current_doc_idx = 0

    for doc in dataset.docs_iter():
        if max_docs > 0 and current_doc_idx >= max_docs:
            break

        doc_id = doc.doc_id

        # Collect doc_ids before resume point (needed for evaluation)
        if current_doc_idx < resume_from:
            skipped_doc_ids.append(doc_id)
            current_doc_idx += 1
            continue

        # Send skipped doc_ids once we start processing
        if skipped_doc_ids:
            queue.put((skipped_doc_ids, None))  # None texts = skipped batch
            skipped_doc_ids = []

        doc_text = doc.text if hasattr(doc, "text") else doc.body
        batch_texts.append(doc_text)
        batch_doc_ids.append(doc_id)

        if len(batch_texts) >= batch_size:
            queue.put((list(batch_doc_ids), list(batch_texts)))
            batch_texts = []
            batch_doc_ids = []

        current_doc_idx += 1

    # Send any remaining skipped doc_ids (if no documents to process)
    if skipped_doc_ids:
        queue.put((skipped_doc_ids, None))

    # Send remaining batch
    if batch_texts:
        queue.put((list(batch_doc_ids), list(batch_texts)))

    queue.put(None)  # Signal done


def build_impact_index_streaming(
    encoder: SpladeEncoder,
    dataset,
    index_dir: Path,
    batch_size: int = 32,
    max_docs: int = None,
    checkpoint_frequency: int = 0,
    in_memory: bool = True,
) -> Tuple[impact_index.Index, List[str]]:
    """Build impact-index from documents using streaming (no full doc list in memory)."""
    print("\n=== Building Impact Index (Streaming) ===")

    done_file = index_dir / ".done"
    doc_ids_file = doc_id_mapping_path(index_dir)

    # Check if index already exists
    if done_file.exists() and doc_ids_file.exists():
        print("  Index already built, loading from disk...")
        index = impact_index.Index.load(str(index_dir), in_memory=in_memory)
        with open(doc_ids_file, "r") as f:
            doc_ids = json.load(f)
        print(f"  Index size: {get_file_size_mb(index_dir):.2f} MB")
        print(f"  Documents indexed: {len(doc_ids)}")
        return index, doc_ids

    options = impact_index.BuilderOptions()
    if checkpoint_frequency > 0:
        options.checkpoint_frequency = checkpoint_frequency
    indexer = impact_index.IndexBuilder(str(index_dir), options)

    # Check for checkpoint to resume from
    checkpoint_doc_id = indexer.get_checkpoint_doc_id()
    resume_from = checkpoint_doc_id + 1 if checkpoint_doc_id is not None else 0
    if resume_from > 0:
        print(f"  Resuming from checkpoint at doc {resume_from}")

    start_time = time.time()
    start_cpu = time.process_time()

    # Get total document count from ir_datasets if available
    total_docs = None
    if hasattr(dataset, "docs_count"):
        total_docs = dataset.docs_count()
        if max_docs > 0:
            total_docs = min(total_docs, max_docs)
    elif max_docs > 0:
        total_docs = max_docs

    # Use prefetching with a queue to overlap I/O and GPU computation
    import queue
    import threading

    # Effective batch size (accounts for multi-GPU)
    effective_batch_size = (
        batch_size * len(encoder.models) if encoder.multi_gpu else batch_size
    )

    # Queue with limited size to control memory usage (2 batches ahead)
    prefetch_queue = queue.Queue(maxsize=2)

    # Start prefetcher thread
    prefetch_thread = threading.Thread(
        target=_document_prefetcher,
        args=(dataset, max_docs, effective_batch_size, prefetch_queue, resume_from),
        daemon=True,
    )
    prefetch_thread.start()

    doc_ids = []
    current_doc_idx = resume_from

    print("  Streaming documents with prefetching...")
    with tqdm(
        total=total_docs, desc="Indexing", unit="doc", initial=resume_from
    ) as pbar:
        while True:
            item = prefetch_queue.get()

            if item is None:  # Done
                break

            batch_doc_ids, batch_texts = item

            # Add doc_ids to the list
            doc_ids.extend(batch_doc_ids)

            # Skip batch (texts is None) - just collecting doc_ids for skipped docs
            if batch_texts is None:
                continue

            # Encode and index the batch
            encodings = encoder.encode(batch_texts)
            for j, sparse_vec in enumerate(encodings):
                doc_idx = current_doc_idx + j
                if sparse_vec:
                    terms = np.array(list(sparse_vec.keys()), dtype=np.uint64)
                    values = np.array(list(sparse_vec.values()), dtype=np.float32)
                    indexer.add(doc_idx, terms, values)

            pbar.update(len(batch_texts))
            current_doc_idx += len(batch_texts)

    prefetch_thread.join()

    # Build the index
    index = indexer.build(in_memory=True)

    end_time = time.time()
    end_cpu = time.process_time()

    # Save doc_ids mapping for evaluation
    with open(doc_ids_file, "w") as f:
        json.dump(doc_ids, f)

    # Mark as done
    done_file.touch()

    print(f"  Documents indexed: {len(doc_ids)}")
    print(f"  Wall time: {end_time - start_time:.2f}s")
    print(f"  CPU time: {end_cpu - start_cpu:.2f}s")
    print(f"  Index size: {get_file_size_mb(index_dir):.2f} MB")

    return index, doc_ids


def search_impact_index(
    index: impact_index.Index,
    query_encodings: List[Dict[int, float]],
    top_k: int = 100,
    method: str = "wand",
) -> Tuple[List[List[Tuple[int, float]]], float, float]:
    """Search using impact-index with WAND or MaxScore algorithm."""
    results = []

    start_time = time.time()
    start_cpu = time.process_time()

    search_fn = index.search_wand if method == "wand" else index.search_maxscore
    desc = f"Searching ({method.upper()})"

    for sparse_vec in tqdm(query_encodings, desc=desc, unit="query"):
        if sparse_vec:
            hits = search_fn(sparse_vec, top_k)
            results.append([(h.docid, h.score) for h in hits])
        else:
            results.append([])

    end_time = time.time()
    end_cpu = time.process_time()

    wall_time = end_time - start_time
    cpu_time = end_cpu - start_cpu

    return results, wall_time, cpu_time


def search_bmp_index(
    searcher: BmpSearcher,
    query_encodings: List[Dict[int, float]],
    top_k: int = 100,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[List[List[Tuple[int, float]]], float, float]:
    """Search using BMP index."""
    results = []

    start_time = time.time()
    start_cpu = time.process_time()

    for sparse_vec in tqdm(query_encodings, desc="Searching (BMP)"):
        if sparse_vec:
            # BMP expects string keys
            str_query = {str(k): v for k, v in sparse_vec.items()}
            doc_ids, scores = searcher.search(
                str_query, k=top_k, alpha=alpha, beta=beta
            )
            # Convert string doc_ids back to int
            results.append([(int(d), s) for d, s in zip(doc_ids, scores)])
        else:
            results.append([])

    end_time = time.time()
    end_cpu = time.process_time()

    wall_time = end_time - start_time
    cpu_time = end_cpu - start_cpu

    return results, wall_time, cpu_time


def search_seismic_index(
    searcher,
    query_encodings: List[Dict[int, float]],
    config: SeismicSearchConfig,
    top_k: int = 100,
) -> Tuple[List[List[Tuple[int, float]]], float, float]:
    """Search using a Seismic index."""
    results = []

    start_time = time.time()
    start_cpu = time.process_time()

    for sparse_vec in tqdm(
        query_encodings, desc=f"Searching (Seismic {config.get_display_name()})"
    ):
        if sparse_vec:
            hits = searcher.search(
                sparse_vec,
                top_k,
                query_cut=config.query_cut,
                heap_factor=config.heap_factor,
                n_knn=config.n_knn,
            )
            results.append([(h.docid, h.score) for h in hits])
        else:
            results.append([])

    end_time = time.time()
    end_cpu = time.process_time()

    return results, end_time - start_time, end_cpu - start_cpu


def exact_recall(
    results: List[List[Tuple[int, float]]],
    reference: List[List[Tuple[int, float]]],
    k: int = 10,
) -> float:
    """Fraction of the exact (reference) top-k documents that an
    approximate method retrieves in its own top-k, averaged over queries."""
    fractions = []
    for hits, ref in zip(results, reference):
        ref_docs = {d for d, _ in ref[:k]}
        if ref_docs:
            fractions.append(len(ref_docs & {d for d, _ in hits[:k]}) / len(ref_docs))
    return float(np.mean(fractions)) if fractions else 0.0


def save_run(path: Path, results, query_ids: List[str], doc_ids: List[str], tag: str):
    """Write results as a TREC run file (so they can be re-evaluated later)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for query_id, hits in zip(query_ids, results):
            for rank, (doc_idx, score) in enumerate(hits, 1):
                f.write(f"{query_id} Q0 {doc_ids[doc_idx]} {rank} {score} {tag}\n")


def compute_metrics(
    results: List[List[Tuple[int, float]]],
    query_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    doc_ids: List[str],
    k_values: List[int] = [10, 100],
) -> Dict[str, float]:
    """Compute IR metrics using ir-measures: MRR, NDCG, Recall."""
    import ir_measures
    from ir_measures import Qrel, ScoredDoc, nDCG, RR, Recall

    # Convert results to ir-measures run format (iterator of ScoredDoc namedtuples)
    def run_iter():
        for query_id, hits in zip(query_ids, results):
            for doc_idx, score in hits:
                yield ScoredDoc(query_id, doc_ids[doc_idx], score)

    # Convert qrels to ir-measures format (iterator of Qrel namedtuples)
    def qrels_iter():
        for query_id, doc_rels in qrels.items():
            for doc_id, relevance in doc_rels.items():
                yield Qrel(query_id, doc_id, relevance)

    # Define metrics to compute
    measures = []
    for k in k_values:
        measures.extend([nDCG @ k, RR @ k, Recall @ k])

    # Compute metrics
    results_dict = ir_measures.calc_aggregate(measures, qrels_iter(), run_iter())

    # Convert to our output format
    metrics = {}
    for measure, value in results_dict.items():
        # Convert measure name to our format (e.g., "nDCG@10" -> "NDCG@10")
        name = str(measure)
        if name.startswith("nDCG"):
            name = name.replace("nDCG", "NDCG")
        elif name.startswith("RR"):
            name = name.replace("RR", "MRR")
        metrics[name] = value

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SPLADE Indexing and Search Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco-passage/dev/small",
        help="ir_datasets dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="naver/splade_v2_max",
        help="HuggingFace SPLADE model name",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum number of documents to index (0 = all)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Maximum number of queries to evaluate (0 = all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for encoding"
    )
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Number of results to retrieve"
    )
    parser.add_argument("--bsize", type=int, default=64, help="BMP block size")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: temp)"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Encoded (raw) impact index directory, reused when already built, so "
        "several benchmark runs can share one encoding (default: OUTPUT_DIR/impact_index)",
    )
    parser.add_argument(
        "--compare-bmp-methods",
        action="store_true",
        help="Compare legacy vs streaming BMP conversion",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use DataParallel for multi-GPU encoding (experimental)",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=0,
        help="Checkpoint frequency for indexing (0 = no checkpointing)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warn",
        choices=["error", "warn", "info", "debug", "trace"],
        help="Rust log level (default: warn)",
    )
    parser.add_argument(
        "--compressed-index",
        type=str,
        action="append",
        dest="compressed_indices",
        metavar="CONFIG",
        help=(
            "Add a compressed index configuration. Can be specified multiple times. "
            "Format: 'key=value key=value ...'. Available keys: "
            "nbits (default: 8), block-size (default: 128), split (comma-separated quantiles). "
            "Examples: 'nbits=16 block-size=128', 'split=0.9 nbits=8', 'split=0.8,0.95 nbits=16'"
        ),
    )
    parser.add_argument(
        "--seismic",
        type=str,
        action="append",
        dest="seismic_indices",
        metavar="CONFIG",
        help=(
            "Add a Seismic index build configuration (repeatable; needs impact_index built "
            "with --features seismic). Keys: n-postings, centroid-fraction, summary-energy, "
            "max-fraction, knn. Use '' for the defaults."
        ),
    )
    parser.add_argument(
        "--seismic-search",
        type=str,
        action="append",
        dest="seismic_searches",
        metavar="CONFIG",
        help=(
            "Seismic query-time operating point (repeatable), e.g. 'query-cut=10 heap-factor=0.7'. "
            "Keys: query-cut, heap-factor, n-knn. Default: a small grid."
        ),
    )
    parser.add_argument(
        "--bmp-search",
        type=str,
        action="append",
        dest="bmp_searches",
        metavar="CONFIG",
        help=(
            "BMP query-time operating point (repeatable), e.g. 'alpha=0.9 beta=0.8'. "
            "alpha < 1 stops early, beta < 1 drops the lightest query terms. "
            "Default: the safe 'alpha=1 beta=1' only."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the SPLADE encoder under fp16 autocast (CUDA only)",
    )
    parser.add_argument(
        "--mmap-index",
        action="store_true",
        help="Memory-map an already built raw index instead of loading it in RAM",
    )
    parser.add_argument(
        "--skip-compressed",
        action="store_true",
        help="Do not build/search any compressed index (ignores --compressed-index)",
    )
    parser.add_argument(
        "--skip-wand",
        action="store_true",
        help="Do not run WAND on the raw index (MaxScore gives the same results "
        "and stays the exact reference)",
    )
    parser.add_argument(
        "--skip-bmp",
        action="store_true",
        help="Do not build/search the BMP index",
    )
    args = parser.parse_args()

    args.seismic_build_configs = [
        SeismicBuildConfig.parse(c) for c in (args.seismic_indices or [])
    ]
    args.seismic_search_configs = [
        SeismicSearchConfig.parse(c) for c in (args.seismic_searches or [])
    ] or [
        SeismicSearchConfig(query_cut=qc, heap_factor=hf)
        for qc, hf in [(5, 0.9), (10, 0.8), (10, 0.7), (20, 0.7), (20, 0.6)]
    ]
    args.bmp_search_configs = [
        BmpSearchConfig.parse(c) for c in (args.bmp_searches or [])
    ] or [BmpSearchConfig()]
    if args.seismic_build_configs and SeismicSearcher is None:
        parser.error(
            "--seismic needs impact_index built with the `seismic` cargo feature"
        )

    # Parse compressed index configurations
    args.index_configs = []
    if args.skip_compressed:
        pass
    elif args.compressed_indices:
        for config_str in args.compressed_indices:
            try:
                config = CompressedIndexConfig.parse(config_str)
                args.index_configs.append(config)
            except ValueError as e:
                parser.error(f"Invalid --compressed-index config '{config_str}': {e}")
    else:
        # Default: always test with a compressed index (block_size=128, nbits=8)
        args.index_configs.append(CompressedIndexConfig(nbits=8, block_size=128))

    # Setup Rust logging via RUST_LOG environment variable
    os.environ["RUST_LOG"] = args.log_level
    # Initialize Python logging as well
    logging.basicConfig(
        level=getattr(
            logging, args.log_level.upper() if args.log_level != "trace" else "DEBUG"
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)
        cleanup = True

    try:
        # Initialize encoder first (needed for both queries and documents)
        encoder = SpladeEncoder(args.model, multi_gpu=args.multi_gpu, fp16=args.fp16)

        # Load dataset for streaming
        import ir_datasets

        print(f"Loading dataset '{args.dataset}'...")
        dataset = ir_datasets.load(args.dataset)

        # Build impact-index using streaming (documents not loaded into memory)
        # Do this first since it's the most time-consuming step
        index_dir = (
            Path(args.index_dir) if args.index_dir else output_dir / "impact_index"
        )
        index_dir.mkdir(parents=True, exist_ok=True)
        impact_idx, doc_ids = build_impact_index_streaming(
            encoder,
            dataset,
            index_dir,
            batch_size=args.batch_size,
            max_docs=args.max_docs,
            checkpoint_frequency=args.checkpoint_frequency,
            in_memory=not args.mmap_index,
        )

        # Load queries and qrels (small, kept in memory)
        dataset, queries, query_ids, qrels = load_queries_and_qrels(
            args.dataset, max_queries=args.max_queries
        )

        # Encode queries
        print("\n=== Encoding Queries ===")
        query_encodings = encoder.encode_batch(
            queries, batch_size=args.batch_size, desc="Encoding queries", unit="query"
        )

        # Save query encodings for debugging
        query_encodings_file = output_dir / "query_encodings.json"
        with open(query_encodings_file, "w") as f:
            json.dump([{str(k): v for k, v in q.items()} for q in query_encodings], f)
        print(f"  Saved query encodings to {query_encodings_file}")

        # Build BMP index (streaming)
        bmp_path = output_dir / "index_streaming.bmp"
        bmp_builder = BmpIndexBuilder(bmp_path, bsize=args.bsize, use_streaming=True)
        run_bmp = not args.skip_bmp
        if run_bmp:
            # Build (or reuse) the file now, but only load it for the BMP
            # searches: the loaded searcher is large (~40 GB on MS MARCO) and
            # would evict the raw index used by the exact searches
            bmp_builder.build(impact_idx, load=False)

        # Optionally compare with legacy BMP conversion
        if args.compare_bmp_methods:
            bmp_legacy_path = output_dir / "index_legacy.bmp"
            bmp_legacy_builder = BmpIndexBuilder(
                bmp_legacy_path, bsize=args.bsize, use_streaming=False
            )
            bmp_legacy_builder.build(impact_idx, load=False)

        # Search with WAND
        run_wand = not args.skip_wand
        if run_wand:
            print("\n=== Searching with WAND ===")
            wand_results, wand_wall, wand_cpu = search_impact_index(
                impact_idx, query_encodings, top_k=args.top_k, method="wand"
            )
            print(f"  Wall time: {wand_wall:.2f}s ({len(queries) / wand_wall:.1f} q/s)")
            print(f"  CPU time: {wand_cpu:.2f}s")

        # Search with MaxScore
        print("\n=== Searching with MaxScore ===")
        maxscore_results, maxscore_wall, maxscore_cpu = search_impact_index(
            impact_idx, query_encodings, top_k=args.top_k, method="maxscore"
        )
        print(
            f"  Wall time: {maxscore_wall:.2f}s ({len(queries) / maxscore_wall:.1f} q/s)"
        )
        print(f"  CPU time: {maxscore_cpu:.2f}s")

        if run_wand:
            # Compare WAND vs MaxScore results directly
            print("\n=== Comparing WAND vs MaxScore (same index, same run) ===")
            doc_set_diffs = 0
            ranking_diffs_top10 = 0
            total_only_wand = 0
            total_only_maxscore = 0
            for i, (wand_hits, maxscore_hits) in enumerate(
                zip(wand_results, maxscore_results)
            ):
                wand_docs = set(d for d, s in wand_hits)
                maxscore_docs = set(d for d, s in maxscore_hits)
                only_wand = wand_docs - maxscore_docs
                only_maxscore = maxscore_docs - wand_docs
                if only_wand or only_maxscore:
                    doc_set_diffs += 1
                    total_only_wand += len(only_wand)
                    total_only_maxscore += len(only_maxscore)
                wand_top10 = [d for d, s in wand_hits[:10]]
                maxscore_top10 = [d for d, s in maxscore_hits[:10]]
                if wand_top10 != maxscore_top10:
                    ranking_diffs_top10 += 1
            print(f"  Queries with TOP 10 ranking differences: {ranking_diffs_top10}")
            print(f"  Queries with document set differences: {doc_set_diffs}")
            print(f"  Total docs only in WAND: {total_only_wand}")
            print(f"  Total docs only in MaxScore: {total_only_maxscore}")

        # Search with BMP
        # (one run per operating point, sequentially so timings do not overlap)
        bmp_runs = []
        if run_bmp:
            bmp_searcher = bmp_builder._load_index()
            for search_config in args.bmp_search_configs:
                name = search_config.get_display_name()
                print(f"\n=== Searching {name} ===")
                b_results, b_wall, b_cpu = search_bmp_index(
                    bmp_searcher,
                    query_encodings,
                    top_k=args.top_k,
                    alpha=search_config.alpha,
                    beta=search_config.beta,
                )
                print(f"  Wall time: {b_wall:.2f}s ({len(queries) / b_wall:.1f} q/s)")
                print(f"  CPU time: {b_cpu:.2f}s")
                bmp_runs.append(
                    {
                        "name": name,
                        "search": search_config,
                        "results": b_results,
                        "wall": b_wall,
                        "cpu": b_cpu,
                    }
                )
            # Free it before the other builds (MS MARCO-sized indices are large)
            bmp_searcher = None

        # Seismic indices, each searched at every operating point
        seismic_runs = []
        for build_config in args.seismic_build_configs:
            seismic_dir = output_dir / f"index_{build_config.get_dir_name()}"
            seismic_builder = SeismicIndexBuilder(seismic_dir, build_config)
            seismic_searcher = seismic_builder.build(impact_idx)
            for search_config in args.seismic_search_configs:
                name = f"{build_config.get_display_name()} {search_config.get_display_name()}"
                print(f"\n=== Searching {name} ===")
                s_results, s_wall, s_cpu = search_seismic_index(
                    seismic_searcher, query_encodings, search_config, top_k=args.top_k
                )
                print(f"  Wall time: {s_wall:.2f}s ({len(queries) / s_wall:.1f} q/s)")
                print(f"  CPU time: {s_cpu:.2f}s")
                seismic_runs.append(
                    {
                        "name": name,
                        "build": build_config,
                        "search": search_config,
                        "builder": seismic_builder,
                        "results": s_results,
                        "wall": s_wall,
                        "cpu": s_cpu,
                    }
                )
            del seismic_searcher

        # Compressed index benchmarks (from --compressed-index configurations)
        # Each entry contains: builder, index, and search results
        configured_indices = []

        for config in args.index_configs:
            index_dir_name = config.get_dir_name()
            config_dir = output_dir / f"index_{index_dir_name}"

            # Build the index using the builder class
            builder = CompressedIndexBuilder(config_dir, config)
            configured_idx = builder.build(impact_idx)

            # Determine which search methods to use
            # Split indices only use MaxScore (it's designed for that)
            # Non-split indices use both WAND and MaxScore
            results = {
                "config": config,
                "builder": builder,
                "dir": config_dir,
                "index": configured_idx,
            }

            if config.is_split():
                # Split index: only MaxScore
                print(f"\n=== Searching {config.get_display_name()} with MaxScore ===")
                split_results, wall, cpu = search_impact_index(
                    configured_idx, query_encodings, top_k=args.top_k, method="maxscore"
                )
                print(f"  Wall time: {wall:.2f}s ({len(queries) / wall:.1f} q/s)")
                print(f"  CPU time: {cpu:.2f}s")
                results["maxscore"] = {
                    "results": split_results,
                    "wall": wall,
                    "cpu": cpu,
                }
            else:
                # Regular compressed index: WAND and MaxScore
                print(f"\n=== Searching {config.get_display_name()} with WAND ===")
                wand_results_cfg, wand_wall_cfg, wand_cpu_cfg = search_impact_index(
                    configured_idx, query_encodings, top_k=args.top_k, method="wand"
                )
                print(
                    f"  Wall time: {wand_wall_cfg:.2f}s ({len(queries) / wand_wall_cfg:.1f} q/s)"
                )
                print(f"  CPU time: {wand_cpu_cfg:.2f}s")
                results["wand"] = {
                    "results": wand_results_cfg,
                    "wall": wand_wall_cfg,
                    "cpu": wand_cpu_cfg,
                }

                print(f"\n=== Searching {config.get_display_name()} with MaxScore ===")
                maxscore_results_cfg, maxscore_wall_cfg, maxscore_cpu_cfg = (
                    search_impact_index(
                        configured_idx,
                        query_encodings,
                        top_k=args.top_k,
                        method="maxscore",
                    )
                )
                print(
                    f"  Wall time: {maxscore_wall_cfg:.2f}s ({len(queries) / maxscore_wall_cfg:.1f} q/s)"
                )
                print(f"  CPU time: {maxscore_cpu_cfg:.2f}s")
                results["maxscore"] = {
                    "results": maxscore_results_cfg,
                    "wall": maxscore_wall_cfg,
                    "cpu": maxscore_cpu_cfg,
                }

            # Free the index once searched (MS MARCO-sized indices are large)
            del results["index"], configured_idx
            configured_indices.append(results)

        # Compute metrics (relevance at 10, 100 and the retrieval depth;
        # agreement with the exact top-k at 10 and the retrieval depth)
        metric_ks = sorted({10, 100, args.top_k})
        exact_ks = sorted({10, args.top_k})
        print("\n=== IR Metrics ===")
        if run_wand:
            print("\nWAND:")
            wand_metrics = compute_metrics(
                wand_results, query_ids, qrels, doc_ids, metric_ks
            )
            for metric, value in sorted(wand_metrics.items()):
                print(f"  {metric}: {value:.4f}")

        print("\nMaxScore:")
        maxscore_metrics = compute_metrics(
            maxscore_results, query_ids, qrels, doc_ids, metric_ks
        )
        for metric, value in sorted(maxscore_metrics.items()):
            print(f"  {metric}: {value:.4f}")

        for run in bmp_runs + seismic_runs:
            run["metrics"] = compute_metrics(
                run["results"], query_ids, qrels, doc_ids, metric_ks
            )
            for k in exact_ks:
                run["metrics"][f"ExactRecall@{k}"] = exact_recall(
                    run["results"], maxscore_results, k
                )
            print(f"\n{run['name']}:")
            for metric, value in sorted(run["metrics"].items()):
                print(f"  {metric}: {value:.4f}")

        # Compute metrics for configured indices
        for entry in configured_indices:
            config = entry["config"]
            if "wand" in entry:
                entry["wand"]["metrics"] = compute_metrics(
                    entry["wand"]["results"], query_ids, qrels, doc_ids, metric_ks
                )
                print(f"\n{config.get_display_name()} WAND:")
                for metric, value in sorted(entry["wand"]["metrics"].items()):
                    print(f"  {metric}: {value:.4f}")

            if "maxscore" in entry:
                entry["maxscore"]["metrics"] = compute_metrics(
                    entry["maxscore"]["results"], query_ids, qrels, doc_ids, metric_ks
                )
                print(f"\n{config.get_display_name()} MaxScore:")
                for metric, value in sorted(entry["maxscore"]["metrics"].items()):
                    print(f"  {metric}: {value:.4f}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\nDataset: {args.dataset}")
        print(f"Documents: {len(doc_ids)}")
        print(f"Queries: {len(queries)}")
        print(f"Model: {args.model}")

        print("\nIndex Sizes:")
        print(f"  Standard Index: {get_file_size_mb(index_dir):.2f} MB")
        if run_bmp:
            print(f"  BMP Index: {get_file_size_mb(bmp_path):.2f} MB")
        for build_config in args.seismic_build_configs:
            seismic_dir = output_dir / f"index_{build_config.get_dir_name()}"
            print(
                f"  {build_config.get_display_name()}: {get_file_size_mb(seismic_dir):.2f} MB"
            )
        for entry in configured_indices:
            config = entry["config"]
            print(
                f"  {config.get_display_name()}: {get_file_size_mb(entry['dir']):.2f} MB"
            )

        print("\nSearch Performance:")
        if run_wand:
            print(
                f"  WAND:     {wand_wall:.2f}s wall, {wand_cpu:.2f}s CPU ({len(queries) / wand_wall:.1f} q/s)"
            )
        print(
            f"  MaxScore: {maxscore_wall:.2f}s wall, {maxscore_cpu:.2f}s CPU ({len(queries) / maxscore_wall:.1f} q/s)"
        )
        for run in bmp_runs + seismic_runs:
            print(
                f"  {run['name']}: {run['wall']:.2f}s wall, {run['cpu']:.2f}s CPU "
                f"({len(queries) / run['wall']:.1f} q/s)"
            )
        for entry in configured_indices:
            config = entry["config"]
            if "wand" in entry:
                w = entry["wand"]
                print(
                    f"  {config.get_display_name()} WAND: {w['wall']:.2f}s wall, {w['cpu']:.2f}s CPU ({len(queries) / w['wall']:.1f} q/s)"
                )
            if "maxscore" in entry:
                m = entry["maxscore"]
                print(
                    f"  {config.get_display_name()} MaxScore: {m['wall']:.2f}s wall, {m['cpu']:.2f}s CPU ({len(queries) / m['wall']:.1f} q/s)"
                )

        # Save every run (TREC format) for later re-evaluation
        runs_dir = output_dir / "runs"
        named_runs = [("wand", wand_results)] if run_wand else []
        named_runs.append(("maxscore", maxscore_results))
        named_runs += [(run["name"], run["results"]) for run in bmp_runs + seismic_runs]
        for entry in configured_indices:
            for method in ("wand", "maxscore"):
                if method in entry:
                    named_runs.append(
                        (
                            f"{entry['config'].get_display_name()} {method}",
                            entry[method]["results"],
                        )
                    )
        for name, run_results in named_runs:
            tag = re.sub(r"[^A-Za-z0-9=.,_-]+", "_", name)
            save_run(runs_dir / f"{tag}.trec", run_results, query_ids, doc_ids, tag)
        print(f"\nRuns saved to: {runs_dir}")

        # Save results to JSON
        results_path = output_dir / "benchmark_results.json"
        index_sizes = {
            "standard_index": get_file_size_mb(index_dir),
        }
        search_times = {"maxscore": {"wall": maxscore_wall, "cpu": maxscore_cpu}}
        metrics = {"maxscore": maxscore_metrics}
        if run_wand:
            search_times["wand"] = {"wall": wand_wall, "cpu": wand_cpu}
            metrics["wand"] = wand_metrics
        if run_bmp:
            index_sizes["bmp_index"] = get_file_size_mb(bmp_path)
        bmp_results = [
            {
                "name": run["name"],
                "search": vars(run["search"]),
                "wall": run["wall"],
                "cpu": run["cpu"],
                "metrics": run["metrics"],
            }
            for run in bmp_runs
        ]

        seismic_results = []
        for run in seismic_runs:
            entry = {
                "name": run["name"],
                "build": vars(run["build"]),
                "search": vars(run["search"]),
                "index_size_mb": run["builder"].get_size_mb(),
                "wall": run["wall"],
                "cpu": run["cpu"],
                "metrics": run["metrics"],
            }
            if run["builder"].build_stats:
                entry["build_stats"] = run["builder"].build_stats.to_dict()
            seismic_results.append(entry)

        # Add configured index results
        configured_results = []
        for entry in configured_indices:
            config = entry["config"]
            builder: CompressedIndexBuilder = entry["builder"]
            config_result = {
                "config": {
                    "nbits": config.nbits,
                    "block_size": config.block_size,
                    "split_quantiles": config.split_quantiles,
                },
                "display_name": config.get_display_name(),
                "index_size_mb": builder.get_size_mb(),
            }
            if builder.build_stats:
                config_result["build_stats"] = builder.build_stats.to_dict()
            if "wand" in entry:
                config_result["wand"] = {
                    "wall": entry["wand"]["wall"],
                    "cpu": entry["wand"]["cpu"],
                    "metrics": entry["wand"]["metrics"],
                }
            if "maxscore" in entry:
                config_result["maxscore"] = {
                    "wall": entry["maxscore"]["wall"],
                    "cpu": entry["maxscore"]["cpu"],
                    "metrics": entry["maxscore"]["metrics"],
                }
            configured_results.append(config_result)

        # Add BMP build stats
        bmp_build_info = {"index_size_mb": bmp_builder.get_size_mb()} if run_bmp else {}
        if run_bmp and bmp_builder.build_stats:
            bmp_build_info["build_stats"] = bmp_builder.build_stats.to_dict()

        results = {
            "config": {
                k: v
                for k, v in vars(args).items()
                if k
                not in (
                    "index_configs",
                    "seismic_build_configs",
                    "seismic_search_configs",
                    "bmp_search_configs",
                )
            },
            "dataset_stats": {
                "num_docs": len(doc_ids),
                "num_queries": len(queries),
            },
            "index_sizes_mb": index_sizes,
            "search_time_seconds": search_times,
            "metrics": metrics,
            "bmp_build": bmp_build_info,
            "bmp": bmp_results,
            "configured_indices": configured_results,
            "seismic": seismic_results,
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    finally:
        if cleanup:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
