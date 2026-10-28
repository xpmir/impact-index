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
