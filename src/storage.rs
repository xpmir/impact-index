//! Where an index's files come from, and where they go.
//!
//! Index loaders read their files through [`IndexFiles`], so that the same
//! code loads an index from a directory on disk ([`DirFiles`], memory-mapped
//! or not) or from byte buffers handed over by the caller ([`MemoryFiles`]),
//! as in WASM where the embedding application does all file I/O. Writers
//! (e.g. [`crate::compress::CompressionTransform::compress_to`]) likewise
//! write through an [`IndexOutput`]: a directory ([`DirOutput`]) or memory
//! ([`MemoryOutput`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::utils::buffer::{ArcSlice, Buffer, MemoryBuffer, MmapBuffer, SharedBuffer, Slice};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// The files of one index directory
pub trait IndexFiles: Send + Sync {
    /// Whether the file `name` exists
    fn exists(&self, name: &str) -> bool;

    /// Opens the file `name` for sequential reading
    fn open(&self, name: &str) -> io::Result<Box<dyn Read + '_>>;

    /// Opens the data file `name` for random access; on disk, `in_memory`
    /// reads it into memory instead of mapping it
    fn buffer(&self, name: &str, in_memory: bool) -> io::Result<Box<dyn Buffer>>;

    /// The files of the sub-directory `name`
    fn sub(&self, name: &str) -> Box<dyn IndexFiles + '_>;

    /// The directory on disk, if the files come from one
    fn dir(&self) -> Option<&Path>;

    /// Human-readable location of `name`, for error messages
    fn describe(&self, name: &str) -> String;

    /// Reads the whole file `name`
    fn read_all(&self, name: &str) -> io::Result<Vec<u8>> {
        let mut data = Vec::new();
        self.open(name)?.read_to_end(&mut data)?;
        Ok(data)
    }
}

/// Files of a directory on disk
pub struct DirFiles {
    path: PathBuf,
}

impl DirFiles {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    /// Names the file in `error` (keeping its kind)
    fn with_path(path: &Path, error: io::Error) -> io::Error {
        io::Error::new(error.kind(), format!("{}: {error}", path.display()))
    }
}

impl IndexFiles for DirFiles {
    fn exists(&self, name: &str) -> bool {
        self.path.join(name).exists()
    }

    fn open(&self, name: &str) -> io::Result<Box<dyn Read + '_>> {
        let path = self.path.join(name);
        let file = std::fs::File::open(&path).map_err(|e| Self::with_path(&path, e))?;
        Ok(Box::new(BufReader::new(file)))
    }

    fn buffer(&self, name: &str, in_memory: bool) -> io::Result<Box<dyn Buffer>> {
        let path = self.path.join(name);
        let buffer: io::Result<Box<dyn Buffer>> = if in_memory {
            MemoryBuffer::open(&path).map(|b| Box::new(b) as Box<dyn Buffer>)
        } else {
            MmapBuffer::open(&path).map(|b| Box::new(b) as Box<dyn Buffer>)
        };
        buffer.map_err(|e| Self::with_path(&path, e))
    }

    fn sub(&self, name: &str) -> Box<dyn IndexFiles + '_> {
        Box::new(DirFiles::new(&self.path.join(name)))
    }

    fn dir(&self) -> Option<&Path> {
        Some(&self.path)
    }

    fn describe(&self, name: &str) -> String {
        self.path.join(name).display().to_string()
    }
}

/// Files held in memory, keyed by their path relative to the index
/// directory (`/`-separated, e.g. `inner/docids.dat` for a split index)
#[derive(Default, Clone)]
pub struct MemoryFiles {
    files: HashMap<String, Arc<Vec<u8>>>,
}

impl MemoryFiles {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the file `name`
    pub fn insert(&mut self, name: impl Into<String>, data: Vec<u8>) {
        self.files.insert(name.into(), Arc::new(data));
    }

    /// The file names, in no particular order
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(|s| s.as_str())
    }

    /// The content of the file `name`
    pub fn get_bytes(&self, name: &str) -> Option<&[u8]> {
        self.files.get(name).map(|data| data.as_slice())
    }

    fn get(&self, name: &str) -> io::Result<&Arc<Vec<u8>>> {
        self.files.get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Missing index file {name}"),
            )
        })
    }
}

impl FromIterator<(String, Vec<u8>)> for MemoryFiles {
    fn from_iter<I: IntoIterator<Item = (String, Vec<u8>)>>(iter: I) -> Self {
        let mut files = Self::new();
        for (name, data) in iter {
            files.insert(name, data);
        }
        files
    }
}

impl IndexFiles for MemoryFiles {
    fn exists(&self, name: &str) -> bool {
        self.files.contains_key(name)
    }

    fn open(&self, name: &str) -> io::Result<Box<dyn Read + '_>> {
        Ok(Box::new(&self.get(name)?[..]))
    }

    fn buffer(&self, name: &str, _in_memory: bool) -> io::Result<Box<dyn Buffer>> {
        Ok(Box::new(SharedBuffer::new(self.get(name)?.clone())))
    }

    fn sub(&self, name: &str) -> Box<dyn IndexFiles + '_> {
        Box::new(SubFiles {
            parent: self,
            prefix: format!("{name}/"),
        })
    }

    fn dir(&self) -> Option<&Path> {
        None
    }

    fn describe(&self, name: &str) -> String {
        name.to_string()
    }
}

/// A sub-directory of [`MemoryFiles`]
struct SubFiles<'a> {
    parent: &'a dyn IndexFiles,
    prefix: String,
}

impl SubFiles<'_> {
    fn path(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }
}

impl IndexFiles for SubFiles<'_> {
    fn exists(&self, name: &str) -> bool {
        self.parent.exists(&self.path(name))
    }

    fn open(&self, name: &str) -> io::Result<Box<dyn Read + '_>> {
        self.parent.open(&self.path(name))
    }

    fn buffer(&self, name: &str, in_memory: bool) -> io::Result<Box<dyn Buffer>> {
        self.parent.buffer(&self.path(name), in_memory)
    }

    fn sub(&self, name: &str) -> Box<dyn IndexFiles + '_> {
        Box::new(SubFiles {
            parent: self,
            prefix: format!("{name}/"),
        })
    }

    fn dir(&self) -> Option<&Path> {
        None
    }

    fn describe(&self, name: &str) -> String {
        self.parent.describe(&self.path(name))
    }
}

/// Where the files of one index directory are written. Generic rather than
/// object-safe, so that writing to disk compiles to plain buffered file
/// writes.
pub trait IndexOutput {
    type Writer: Write;

    /// Creates (or truncates) the file `name`
    fn create(&self, name: &str) -> io::Result<Self::Writer>;

    /// Completes the file `name` once everything has been written to it
    fn close(&self, name: &str, writer: Self::Writer) -> io::Result<()>;

    /// Human-readable location of `name`, for messages
    fn describe(&self, name: &str) -> String;
}

/// Writes into an existing directory on disk
pub struct DirOutput {
    path: PathBuf,
}

impl DirOutput {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl IndexOutput for DirOutput {
    type Writer = BufWriter<std::fs::File>;

    fn create(&self, name: &str) -> io::Result<Self::Writer> {
        let file = std::fs::File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(self.path.join(name))?;
        Ok(BufWriter::new(file))
    }

    fn close(&self, _name: &str, mut writer: Self::Writer) -> io::Result<()> {
        writer.flush()
    }

    fn describe(&self, name: &str) -> String {
        self.path.join(name).display().to_string()
    }
}

/// Collects the written files in memory
#[derive(Default)]
pub struct MemoryOutput {
    files: RefCell<MemoryFiles>,
}

impl MemoryOutput {
    pub fn new() -> Self {
        Self::default()
    }

    /// The files written so far
    pub fn into_files(self) -> MemoryFiles {
        self.files.into_inner()
    }
}

impl IndexOutput for MemoryOutput {
    type Writer = Vec<u8>;

    fn create(&self, _name: &str) -> io::Result<Self::Writer> {
        Ok(Vec::new())
    }

    fn close(&self, name: &str, writer: Self::Writer) -> io::Result<()> {
        self.files.borrow_mut().insert(name, writer);
        Ok(())
    }

    fn describe(&self, name: &str) -> String {
        name.to_string()
    }
}

/*
 * ---- Lazy reading through a ranged reader
 */

/// Ranged reads of index files, e.g. from a JS callback in WASM. File
/// names are relative to the index directory (`inner/docids.dat` for a
/// split index).
pub trait RangeReader: Send + Sync {
    /// Size of the file `name`, `None` if it does not exist
    fn size(&self, name: &str) -> io::Result<Option<u64>>;

    /// Exactly `length` bytes of `name` at `offset`
    fn read(&self, name: &str, offset: u64, length: usize) -> io::Result<Vec<u8>>;
}

/// A cached byte range of a file
struct CachedRange {
    end: u64,
    data: Arc<Vec<u8>>,
    /// Query during which the range was last used
    query: u64,
}

#[derive(Default)]
struct CacheState {
    /// File name → start → range
    files: HashMap<String, BTreeMap<u64, CachedRange>>,
    bytes: usize,
    query: u64,
    reads: u64,
}

/// Byte ranges read by [`ReaderFiles`], bounded in bytes, least recently
/// used first out. Ranges used by the current query are never evicted
/// (a query may thus exceed the capacity).
pub struct RangeCache {
    capacity: usize,
    state: Mutex<CacheState>,
}

impl RangeCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::default(),
        }
    }

    /// Starts a query: ranges prefetched from now on belong to it
    pub fn start_query(&self) {
        self.state.lock().unwrap().query += 1;
    }

    /// Evicts ranges of earlier queries, least recently used first, until
    /// the cache fits in its capacity
    pub fn evict(&self) {
        let mut state = self.state.lock().unwrap();
        if state.bytes <= self.capacity {
            return;
        }
        let current = state.query;
        let mut old: Vec<(u64, String, u64, usize)> = state
            .files
            .iter()
            .flat_map(|(name, ranges)| {
                ranges
                    .iter()
                    .filter(|(_, r)| r.query < current)
                    .map(move |(&start, r)| (r.query, name.clone(), start, r.data.len()))
            })
            .collect();
        old.sort_unstable();
        for (_, name, start, size) in old {
            if state.bytes <= self.capacity {
                break;
            }
            state.files.get_mut(&name).unwrap().remove(&start);
            state.bytes -= size;
        }
    }

    /// Bytes held
    pub fn bytes(&self) -> usize {
        self.state.lock().unwrap().bytes
    }

    /// Number of reads done so far (cache misses)
    pub fn reads(&self) -> u64 {
        self.state.lock().unwrap().reads
    }

    /// The cached range of `name` containing `start..end`, marked as used
    fn get(&self, name: &str, start: u64, end: u64) -> Option<(u64, Arc<Vec<u8>>)> {
        let mut state = self.state.lock().unwrap();
        let query = state.query;
        let (&range_start, range) = state.files.get_mut(name)?.range_mut(..=start).next_back()?;
        (end <= range.end).then(|| {
            range.query = query;
            (range_start, range.data.clone())
        })
    }

    fn insert(&self, name: &str, start: u64, data: Vec<u8>) -> Arc<Vec<u8>> {
        let data = Arc::new(data);
        let mut state = self.state.lock().unwrap();
        let query = state.query;
        state.bytes += data.len();
        let range = CachedRange {
            end: start + data.len() as u64,
            data: data.clone(),
            query,
        };
        if let Some(old) = state
            .files
            .entry(name.to_string())
            .or_default()
            .insert(start, range)
        {
            state.bytes -= old.data.len();
        }
        data
    }

    fn count_read(&self) {
        self.state.lock().unwrap().reads += 1;
    }
}

/// Index files read through a [`RangeReader`]: the posting files of a
/// compressed index (`docids.dat`, `impacts.dat`, `positions.dat`) lazily, by term, through a shared
/// [`RangeCache`]; the other files (metadata) whole, when loading. Call
/// [`crate::index::SparseIndex::prefetch`] with the query terms before each
/// search (between [`RangeCache::start_query`] and [`RangeCache::evict`]):
/// postings that were not prefetched are read during the search, where a
/// read error can only panic.
pub struct ReaderFiles {
    reader: Arc<dyn RangeReader>,
    cache: Arc<RangeCache>,
}

impl ReaderFiles {
    pub fn new(reader: Arc<dyn RangeReader>, cache: Arc<RangeCache>) -> Self {
        Self { reader, cache }
    }

    fn read_whole(&self, name: &str) -> io::Result<Vec<u8>> {
        let size = self.reader.size(name)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("Missing index file {name}"),
            )
        })?;
        let data = self.reader.read(name, 0, size as usize)?;
        check_length(name, 0, size as usize, &data)?;
        Ok(data)
    }
}

fn check_length(name: &str, offset: u64, length: usize, data: &[u8]) -> io::Result<()> {
    if data.len() != length {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "{name}: read {} bytes at {offset} instead of {length}",
                data.len()
            ),
        ));
    }
    Ok(())
}

/// Whether `name` is a posting file of a compressed index, read lazily
fn is_lazy(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    base == "docids.dat" || base == "impacts.dat" || base == "positions.dat"
}

impl IndexFiles for ReaderFiles {
    fn exists(&self, name: &str) -> bool {
        matches!(self.reader.size(name), Ok(Some(_)))
    }

    fn open(&self, name: &str) -> io::Result<Box<dyn Read + '_>> {
        Ok(Box::new(io::Cursor::new(self.read_whole(name)?)))
    }

    fn buffer(&self, name: &str, _in_memory: bool) -> io::Result<Box<dyn Buffer>> {
        if is_lazy(name) {
            if !self.exists(name) {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Missing index file {name}"),
                ));
            }
            Ok(Box::new(ReaderBuffer {
                name: name.to_string(),
                reader: self.reader.clone(),
                cache: self.cache.clone(),
            }))
        } else {
            Ok(Box::new(SharedBuffer::new(Arc::new(
                self.read_whole(name)?,
            ))))
        }
    }

    fn sub(&self, name: &str) -> Box<dyn IndexFiles + '_> {
        Box::new(SubFiles {
            parent: self,
            prefix: format!("{name}/"),
        })
    }

    fn dir(&self) -> Option<&Path> {
        None
    }

    fn describe(&self, name: &str) -> String {
        name.to_string()
    }
}

/// A posting file read through the cache of its [`ReaderFiles`]
struct ReaderBuffer {
    name: String,
    reader: Arc<dyn RangeReader>,
    cache: Arc<RangeCache>,
}

impl ReaderBuffer {
    fn load(&self, start: u64, end: u64) -> io::Result<(u64, Arc<Vec<u8>>)> {
        if let Some(found) = self.cache.get(&self.name, start, end) {
            return Ok(found);
        }
        let length = (end - start) as usize;
        self.cache.count_read();
        let data = self.reader.read(&self.name, start, length)?;
        check_length(&self.name, start, length, &data)?;
        Ok((start, self.cache.insert(&self.name, start, data)))
    }
}

impl Buffer for ReaderBuffer {
    fn slice(&'_ self, start: usize, end: usize) -> Box<dyn Slice + '_> {
        let (range_start, data) = self
            .load(start as u64, end as u64)
            .unwrap_or_else(|e| panic!("{} (postings not prefetched): {e}", self.name));
        let offset = (start as u64 - range_start) as usize;
        Box::new(ArcSlice {
            data,
            start: offset,
            end: offset + end - start,
        })
    }

    fn prefetch(&self, start: usize, end: usize) -> io::Result<()> {
        if end > start {
            self.load(start as u64, end as u64)?;
        }
        Ok(())
    }
}
