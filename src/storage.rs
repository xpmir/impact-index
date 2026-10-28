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

use crate::utils::buffer::{Buffer, MemoryBuffer, MmapBuffer, SharedBuffer};

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
