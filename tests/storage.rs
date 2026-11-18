//! Loading indexes from in-memory files and building them in memory (the
//! WASM paths), and the portable Elias-Fano codec used on 32-bit targets.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use helpers::index::{check_same_index, TestIndex};
use impact_index::base::{load_index, load_index_from_files, DocId, ImpactValue, TermIndex};
use impact_index::builder::{BuilderOptions, Indexer, MemoryIndexer};
use impact_index::compress::docid::{BitPackingCompressor, EliasFanoCompressor, PForCompressor};
use impact_index::compress::elias_fano;
use impact_index::compress::impact::{
    Identity, QuantizationRange, QuantizerCodec, RangedQuantizerFactory,
};
use impact_index::compress::{
    CompressionTransform, DocIdCompressorFactory, ImpactCompressorFactory,
};
use impact_index::index::SparseIndex;
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::storage::{IndexFiles, MemoryFiles, MemoryOutput};
use impact_index::transforms::split::SplitIndexTransform;
use impact_index::transforms::IndexTransform;
use sucds::{EliasFanoBuilder, Searial};
use temp_dir::TempDir;

/// Deterministic PRNG
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

fn sucds_bytes(values: &[u64], universe: u64) -> Vec<u8> {
    let mut b = EliasFanoBuilder::new(universe as usize, values.len()).unwrap();
    for &v in values {
        b.push(v as usize).unwrap();
    }
    let mut out = Vec::new();
    b.build().serialize_into(&mut out).unwrap();
    out
}

#[test]
fn test_elias_fano_matches_sucds() {
    let mut rng = Rng(7);
    // (length, maximum gap): dense, sparse, sequences longer than a DArray
    // block (1024 ones), and gaps large enough for overflow blocks
    let shapes = [
        (1, 1),
        (1, 1000),
        (5, 1),
        (100, 3),
        (128, 50),
        (1500, 2),
        (3000, 100),
        (2100, 1000),
        (64, 1 << 20),
    ];
    for (len, max_gap) in shapes {
        let mut values = Vec::with_capacity(len);
        let mut x = rng.below(max_gap);
        for _ in 0..len {
            values.push(x);
            // Gaps of 0 give duplicates, which Elias-Fano allows
            x += rng.below(max_gap + 1);
        }
        let universe = values.last().unwrap() + 1 + rng.below(10);

        let expected = sucds_bytes(&values, universe);
        let mut got = Vec::new();
        elias_fano::write(&mut got, &values, universe).unwrap();
        assert_eq!(
            got, expected,
            "bytes differ for {len} values, gap {max_gap}"
        );

        let ef = elias_fano::EliasFano::read(&expected).unwrap();
        assert_eq!(ef.len(), len as u64);
        assert_eq!(ef.iter().collect::<Vec<_>>(), values);
    }
}

#[test]
fn test_elias_fano_rejects_garbage() {
    let mut bytes = Vec::new();
    elias_fano::write(&mut bytes, &[1, 5, 9], 10).unwrap();
    for cut in [0, 7, bytes.len() / 2, bytes.len() - 1] {
        assert!(elias_fano::EliasFano::read(&bytes[..cut]).is_err());
    }
}

/// Reads every file under `dir` into memory, keyed by relative path
fn read_dir(dir: &Path) -> MemoryFiles {
    fn walk(root: &Path, dir: &Path, files: &mut MemoryFiles) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, files);
            } else {
                let name = path.strip_prefix(root).unwrap().to_str().unwrap();
                files.insert(name.replace('\\', "/"), std::fs::read(&path).unwrap());
            }
        }
    }
    let mut files = MemoryFiles::new();
    walk(dir, dir, &mut files);
    files
}

fn test_index() -> TestIndex {
    TestIndex::new(
        100,
        2_000,
        5.,
        10,
        Some(42),
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 10,
            checkpoint_flush_ratio: 0.5,
            positions: false,
        },
        &HashSet::<DocId>::new(),
    )
}

fn compression() -> CompressionTransform {
    CompressionTransform {
        max_block_size: 64,
        doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
        impacts_compressor_factory: Box::new(Identity {}),
        positions_codec: None,
    }
}

/// Same postings and same search results
fn assert_same(a: &dyn SparseIndex, b: &dyn SparseIndex) {
    assert_eq!(a.len(), b.len());
    for term_ix in 0..a.len() {
        check_same_index(
            a.block_iterator(term_ix).as_mut(),
            b.block_iterator(term_ix).as_mut(),
            0.,
        );
    }
    let mut rng = Rng(3);
    for _ in 0..20 {
        let query: HashMap<TermIndex, ImpactValue> = (0..4)
            .map(|_| (rng.below(a.len() as u64) as TermIndex, 1.))
            .collect();
        let options = MaxScoreOptions::default();
        let ra = search_maxscore(a, &query, 10, options.clone());
        let rb = search_maxscore(b, &query, 10, options);
        let key = |r: &[impact_index::search::ScoredDocument]| -> Vec<(DocId, f32)> {
            r.iter().map(|d| (d.docid, d.score)).collect()
        };
        assert_eq!(key(&ra), key(&rb));
    }
}

#[test]
fn test_load_forward_index_from_files() {
    let data = test_index();
    let dir = data.dir.path();
    let from_disk = load_index(dir, true);
    let from_memory = load_index_from_files(&read_dir(dir), true).unwrap();
    assert_same(&*from_disk, &*from_memory);
}

#[test]
fn test_load_compressed_index_from_files() {
    let mut data = test_index();
    let index = data.indexer.to_index(true);
    let out = TempDir::new().unwrap();
    compression().process(out.path(), &index).unwrap();

    let from_disk = load_index(out.path(), true);
    let files = read_dir(out.path());
    let from_memory = load_index_from_files(&files, false).unwrap();
    assert_same(&*from_disk, &*from_memory);
}

#[test]
fn test_load_split_index_from_files() {
    let mut data = test_index();
    let index = data.indexer.to_index(true);
    let out = TempDir::new().unwrap();
    SplitIndexTransform {
        sink: Box::new(compression()),
        quantiles: vec![0.5, 0.9],
    }
    .process(out.path(), &index)
    .unwrap();

    let from_disk = load_index(out.path(), true);
    let files = read_dir(out.path());
    assert!(files.exists("inner/index.cbor"));
    let from_memory = load_index_from_files(&files, true).unwrap();
    assert_same(&*from_disk, &*from_memory);
}

#[test]
fn test_load_from_files_reports_errors() {
    let mut data = test_index();
    let index = data.indexer.to_index(true);
    let out = TempDir::new().unwrap();
    compression().process(out.path(), &index).unwrap();
    let complete = read_dir(out.path());

    // A missing data file is an error, not a panic
    let partial: MemoryFiles = complete
        .names()
        .filter(|n| *n != "impacts.dat")
        .map(|n| (n.to_string(), complete.read_all(n).unwrap()))
        .collect();
    let err = load_index_from_files(&partial, true).err().unwrap();
    assert!(err.to_string().contains("impacts.dat"), "{err}");

    // So is a corrupt metadata file
    let mut corrupt = complete.clone();
    corrupt.insert("index.cbor", vec![0xff, 0x00, 0x13]);
    assert!(load_index_from_files(&corrupt, true).is_err());

    // And a manifest from a newer format version
    let mut newer = complete.clone();
    let manifest = String::from_utf8(complete.read_all("manifest.json").unwrap()).unwrap();
    let manifest = manifest.replace("\"format_version\": ", "\"format_version\": 9");
    newer.insert("manifest.json", manifest.into_bytes());
    assert!(load_index_from_files(&newer, true).is_err());
}

/// Documents with increasing ids (with gaps and empty documents)
fn random_docs(rng: &mut Rng, n: usize) -> Vec<(DocId, Vec<TermIndex>, Vec<ImpactValue>)> {
    let mut docid = 0;
    (0..n)
        .map(|_| {
            docid += 1 + rng.below(3);
            let mut terms: Vec<TermIndex> = (0..rng.below(30))
                .map(|_| (rng.below(400) as f64).sqrt() as TermIndex * 3)
                .collect();
            terms.sort();
            terms.dedup();
            let values = terms
                .iter()
                .map(|_| 0.01 + rng.below(3000) as f32 / 1000.)
                .collect();
            (docid, terms, values)
        })
        .collect()
}

#[test]
fn test_memory_build_matches_disk_build() {
    let mut rng = Rng(11);
    let docs = random_docs(&mut rng, 3000);

    let tmp = TempDir::new().unwrap();
    let mut indexer = Indexer::<f32>::new(&tmp.path().join("raw"), &BuilderOptions::default());
    let mut memory = MemoryIndexer::new();
    for (docid, terms, values) in &docs {
        let t: ndarray::Array1<TermIndex> = terms.iter().copied().collect();
        let v: ndarray::Array1<f32> = values.iter().copied().collect();
        indexer.add(*docid, &t, &v).unwrap();
        memory.add(*docid, terms, values).unwrap();
    }
    indexer.build().unwrap();
    let raw = indexer.to_index(true);

    let docids: Vec<fn() -> Box<dyn DocIdCompressorFactory>> = vec![
        || Box::new(EliasFanoCompressor {}),
        || Box::new(PForCompressor {}),
        || Box::new(BitPackingCompressor {}),
    ];
    let impacts: Vec<fn() -> Box<dyn ImpactCompressorFactory>> = vec![
        || Box::new(Identity {}),
        || {
            Box::new(RangedQuantizerFactory {
                nbits: 8,
                range: QuantizationRange::PerIndex,
                codec: QuantizerCodec::BitPacked,
            })
        },
        || {
            Box::new(RangedQuantizerFactory {
                nbits: 10,
                range: QuantizationRange::Fixed { min: 0., max: 2. },
                codec: QuantizerCodec::Plain,
            })
        },
    ];
    for (i, docid) in docids.iter().enumerate() {
        for (j, impact) in impacts.iter().enumerate() {
            let transform = || CompressionTransform {
                max_block_size: 32,
                doc_ids_compressor_factory: docid(),
                impacts_compressor_factory: impact(),
                positions_codec: None,
            };
            let dir = tmp.path().join(format!("c{i}{j}"));
            let disk_report = transform().compress(&dir, &raw).unwrap();
            let out = MemoryOutput::new();
            let memory_report = transform().compress_to(&out, &memory).unwrap();
            assert_eq!(disk_report.clamped_impacts, memory_report.clamped_impacts);

            let on_disk = read_dir(&dir);
            let in_memory = out.into_files();
            let mut names: Vec<&str> = in_memory.names().collect();
            names.sort();
            let mut disk_names: Vec<&str> = on_disk.names().collect();
            disk_names.sort();
            assert_eq!(names, disk_names);
            for name in names {
                if name == "manifest.json" {
                    // Differs by its creation time
                    continue;
                }
                assert!(
                    in_memory.get_bytes(name) == on_disk.get_bytes(name),
                    "{name} differs (docids #{i}, impacts #{j})"
                );
            }

            let a = load_index(&dir, true);
            let b = load_index_from_files(&in_memory, true).unwrap();
            assert_same(&*a, &*b);
        }
    }
}

#[test]
fn test_memory_indexer_rejects_invalid_documents() {
    let mut memory = MemoryIndexer::new();
    memory.add(3, &[1, 2], &[0.5, 1.]).unwrap();
    // Not increasing
    assert!(memory.add(3, &[1], &[1.]).is_err());
    assert!(memory.add(2, &[1], &[1.]).is_err());
    // Lengths, repeated terms, non-positive values
    assert!(memory.add(4, &[1, 2], &[1.]).is_err());
    assert!(memory.add(4, &[1, 1], &[1., 2.]).is_err());
    assert!(memory.add(4, &[1], &[0.]).is_err());
    assert!(memory.add(4, &[1], &[f32::NAN]).is_err());
    // Nothing was added by the rejected documents
    assert_eq!(memory.num_postings(), 2);
    memory.add(4, &[7], &[1.]).unwrap();
    assert_eq!(memory.num_postings(), 3);
}

#[test]
fn test_memory_indexer_docid_gaps() {
    use impact_index::index::SparseIndexView;
    // Gaps of every varint length, from document 0 up to u64::MAX
    let docids: Vec<DocId> = vec![
        0,
        1,
        127,
        128,
        255,
        1 << 14,
        (1 << 21) + 3,
        1 << 32,
        (1 << 35) + 1,
        u64::MAX - 1,
        u64::MAX,
    ];
    let mut memory = MemoryIndexer::new();
    for (i, &docid) in docids.iter().enumerate() {
        memory
            .add(docid, &[0, 1 + i % 2], &[1., i as f32 + 1.])
            .unwrap();
    }
    let all: Vec<DocId> = memory.iterator(0).map(|p| p.docid).collect();
    assert_eq!(all, docids);
    let odd: Vec<(DocId, f32)> = memory.iterator(2).map(|p| (p.docid, p.value)).collect();
    let expected: Vec<(DocId, f32)> = docids
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(i, &d)| (d, i as f32 + 1.))
        .collect();
    assert_eq!(odd, expected);
    assert_eq!(memory.num_postings(), 2 * docids.len());
    assert_eq!(memory.max_doc_id(), u64::MAX);
}

/// Compressed files of `index`, without the manifest (creation time)
fn compressed_files(
    index: &dyn impact_index::index::SparseIndexView,
) -> std::io::Result<Vec<(String, Vec<u8>)>> {
    let transform = CompressionTransform {
        max_block_size: 32,
        doc_ids_compressor_factory: Box::new(PForCompressor {}),
        impacts_compressor_factory: Box::new(RangedQuantizerFactory {
            nbits: 8,
            range: QuantizationRange::PerIndex,
            codec: QuantizerCodec::BitPacked,
        }),
        positions_codec: None,
    };
    let out = MemoryOutput::new();
    transform.compress_to(&out, index)?;
    let files = out.into_files();
    let mut result: Vec<(String, Vec<u8>)> = files
        .names()
        .filter(|n| *n != "manifest.json")
        .map(|n| (n.to_string(), files.get_bytes(n).unwrap().to_vec()))
        .collect();
    result.sort();
    Ok(result)
}

#[test]
fn test_memory_indexer_spill() {
    let mut rng = Rng(5);
    let docs = random_docs(&mut rng, 3000);
    let mut reference = MemoryIndexer::new();
    for (docid, terms, values) in &docs {
        reference.add(*docid, terms, values).unwrap();
    }
    let expected = compressed_files(&reference).unwrap();

    // Spilling down to `target` whenever over `budget`, as the WASM builder
    for (budget, target) in [(0, 0), (500, 250), (20_000, 10_000), (1 << 30, 0)] {
        let mut memory = MemoryIndexer::new();
        let mut scratch: Vec<u8> = Vec::new();
        let mut max_memory = 0;
        for (docid, terms, values) in &docs {
            memory.add(*docid, terms, values).unwrap();
            if memory.memory_bytes() > budget {
                memory.spill(target, &mut scratch).unwrap();
                assert!(memory.memory_bytes() <= target);
            }
            max_memory = max_memory.max(memory.memory_bytes());
        }
        if budget == 1 << 30 {
            memory.spill(0, &mut scratch).unwrap();
        }
        assert!(memory.is_spilled());
        assert_eq!(memory.spilled_bytes(), scratch.len() as u64);
        assert_eq!(memory.num_postings(), reference.num_postings());
        for read_ahead in [0, 100, 5000, 1 << 30] {
            let spilled = memory.with_spill(&scratch, read_ahead);
            assert!(
                compressed_files(&spilled).unwrap() == expected,
                "budget {budget}, read ahead {read_ahead}"
            );
            spilled.check().unwrap();
        }

        // Read errors are reported, not panics
        for bad in [
            scratch[..scratch.len() / 2].to_vec(),
            scratch.iter().map(|_| 0x80).collect(),
        ] {
            for read_ahead in [0, 5000] {
                let spilled = memory.with_spill(&bad, read_ahead);
                let _ = compressed_files(&spilled);
                assert!(spilled.check().is_err(), "budget {budget}");
            }
        }
    }
}

/// A [`RangeReader`] over files in memory, counting reads, failing on demand
struct TestReader {
    files: MemoryFiles,
    reads: std::sync::atomic::AtomicUsize,
    fail: std::sync::atomic::AtomicBool,
}

impl impact_index::storage::RangeReader for TestReader {
    fn size(&self, name: &str) -> std::io::Result<Option<u64>> {
        Ok(self.files.get_bytes(name).map(|b| b.len() as u64))
    }

    fn read(&self, name: &str, offset: u64, length: usize) -> std::io::Result<Vec<u8>> {
        use std::sync::atomic::Ordering;
        if self.fail.load(Ordering::SeqCst) {
            return Err(std::io::Error::other("file deleted"));
        }
        self.reads.fetch_add(1, Ordering::SeqCst);
        let bytes = self.files.get_bytes(name).unwrap();
        Ok(bytes[offset as usize..offset as usize + length].to_vec())
    }
}

fn check_lazy_index(files: MemoryFiles) {
    use impact_index::storage::{RangeCache, ReaderFiles};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let reference = load_index_from_files(&files, true).unwrap();
    let reader = Arc::new(TestReader {
        files,
        reads: Default::default(),
        fail: Default::default(),
    });
    let capacity = 4000;
    let cache = Arc::new(RangeCache::new(capacity));
    let lazy =
        load_index_from_files(&ReaderFiles::new(reader.clone(), cache.clone()), true).unwrap();

    let mut rng = Rng(17);
    for _ in 0..200 {
        let terms: Vec<TermIndex> = (0..4)
            .map(|_| rng.below(reference.len() as u64) as TermIndex)
            .collect();
        let query: HashMap<TermIndex, ImpactValue> = terms.iter().map(|&t| (t, 1.)).collect();
        cache.start_query();
        lazy.prefetch(&terms).unwrap();
        cache.evict();
        let reads = reader.reads.load(Ordering::SeqCst);
        let expected = search_maxscore(&*reference, &query, 10, MaxScoreOptions::default());
        let got = search_maxscore(&*lazy, &query, 10, MaxScoreOptions::default());
        // Nothing read during the search
        assert_eq!(reader.reads.load(Ordering::SeqCst), reads);
        let key = |r: &[impact_index::search::ScoredDocument]| -> Vec<(DocId, f32)> {
            r.iter().map(|d| (d.docid, d.score)).collect()
        };
        assert_eq!(key(&got), key(&expected));
    }
    // Bounded cache, and repeated queries hit it
    assert!(
        cache.bytes() <= capacity + 4 * 2 * 2000,
        "{}",
        cache.bytes()
    );
    let terms = [0, 1];
    cache.start_query();
    lazy.prefetch(&terms).unwrap();
    cache.evict();
    let reads = reader.reads.load(Ordering::SeqCst);
    for _ in 0..2 {
        cache.start_query();
        lazy.prefetch(&terms).unwrap();
        cache.evict();
    }
    assert_eq!(reader.reads.load(Ordering::SeqCst), reads);

    // A failing reader: an error from prefetch, and the index still works
    // once the reader is back
    reader.fail.store(true, Ordering::SeqCst);
    let all: Vec<TermIndex> = (0..reference.len()).collect();
    cache.start_query();
    assert!(lazy.prefetch(&all).is_err());
    reader.fail.store(false, Ordering::SeqCst);
    cache.start_query();
    lazy.prefetch(&all).unwrap();
    assert_same(&*reference, &*lazy);
}

#[test]
fn test_lazy_compressed_index() {
    let mut data = test_index();
    let index = data.indexer.to_index(true);
    let out = TempDir::new().unwrap();
    compression().process(out.path(), &index).unwrap();
    check_lazy_index(read_dir(out.path()));
}

#[test]
fn test_lazy_split_index() {
    let mut data = test_index();
    let index = data.indexer.to_index(true);
    let out = TempDir::new().unwrap();
    SplitIndexTransform {
        sink: Box::new(compression()),
        quantiles: vec![0.5, 0.9],
    }
    .process(out.path(), &index)
    .unwrap();
    check_lazy_index(read_dir(out.path()));
}
