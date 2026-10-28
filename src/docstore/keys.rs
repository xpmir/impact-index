//! Keys of a document store, without zstd: per key name, the key value →
//! document number FST (`key_<name>.fst`) and, since format version 2, the
//! document number → key value table:
//!
//! - `key_<name>.dat`: the key values (UTF-8), by increasing document number
//! - `key_<name>.idx`: one u64 (little-endian) per document number, the end
//!   offset of its value in `key_<name>.dat`, with [`PRESENT`] set if the
//!   document has this key (its value starts at the previous end offset)
//!
//! [`KeyStoreBuilder`] builds key-only stores in memory.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

use crate::base::DocId;
use crate::storage::{DirOutput, IndexFiles, IndexOutput};
use crate::utils::buffer::Buffer;

use super::{
    key_fst_file, key_index_file, key_values_file, DocumentStoreMeta, DOCSTORE_VERSION, META_FILE,
};

/// Flag of `key_<name>.idx` entries: the document has a value for the key
pub const PRESENT: u64 = 1 << 63;

fn invalid_input(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Writes the FST and the document number → value table of one key name;
/// `entries` are (value, document number) by increasing document number
pub(crate) fn write_key<O: IndexOutput>(
    out: &O,
    key_name: &str,
    mut entries: Vec<(String, DocId)>,
    num_documents: DocId,
) -> io::Result<()> {
    let (idx_name, dat_name) = (key_index_file(key_name), key_values_file(key_name));
    let mut idx = out.create(&idx_name)?;
    let mut dat = out.create(&dat_name)?;
    let mut end = 0u64;
    let mut next = 0;
    for (value, docnum) in &entries {
        // Document numbers without this key
        for _ in next..*docnum {
            idx.write_all(&end.to_le_bytes())?;
        }
        dat.write_all(value.as_bytes())?;
        end += value.len() as u64;
        idx.write_all(&(end | PRESENT).to_le_bytes())?;
        next = docnum + 1;
    }
    for _ in next..num_documents {
        idx.write_all(&end.to_le_bytes())?;
    }
    out.close(&idx_name, idx)?;
    out.close(&dat_name, dat)?;

    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    if let Some(w) = entries.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(invalid_input(format!(
            "Duplicate key value '{}' for key '{}'",
            w[0].0, key_name
        )));
    }
    let fst_name = key_fst_file(key_name);
    let mut fst = fst::MapBuilder::new(out.create(&fst_name)?).map_err(io::Error::other)?;
    for (value, docnum) in &entries {
        fst.insert(value, *docnum).map_err(io::Error::other)?;
    }
    out.close(&fst_name, fst.into_inner().map_err(io::Error::other)?)
}

pub(crate) fn write_meta<O: IndexOutput>(out: &O, meta: &DocumentStoreMeta) -> io::Result<()> {
    let mut writer = out.create(META_FILE)?;
    ciborium::ser::into_writer(meta, &mut writer).map_err(io::Error::other)?;
    out.close(META_FILE, writer)
}

/// Document number → value table of one key name
pub(crate) struct KeyTable {
    idx: Box<dyn Buffer>,
    dat: Box<dyn Buffer>,
}

impl KeyTable {
    pub(crate) fn load(
        files: &dyn IndexFiles,
        key_name: &str,
        num_documents: u64,
        in_memory: bool,
    ) -> io::Result<Self> {
        let table = Self {
            idx: files.buffer(&key_index_file(key_name), in_memory)?,
            dat: files.buffer(&key_values_file(key_name), in_memory)?,
        };
        // Sizes must agree, so that lookups stay in bounds
        let consistent = num_documents.checked_mul(8) == Some(table.idx().len() as u64)
            && (num_documents == 0
                || table.end(num_documents as usize - 1) <= table.dat().len() as u64);
        if !consistent {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Inconsistent key files {}",
                    files.describe(&key_index_file(key_name))
                ),
            ));
        }
        Ok(table)
    }

    fn idx(&self) -> &[u8] {
        self.idx.as_bytes().expect("buffers expose their bytes")
    }

    fn dat(&self) -> &[u8] {
        self.dat.as_bytes().expect("buffers expose their bytes")
    }

    fn entry(&self, i: usize) -> u64 {
        u64::from_le_bytes(self.idx()[i * 8..i * 8 + 8].try_into().unwrap())
    }

    fn end(&self, i: usize) -> u64 {
        self.entry(i) & !PRESENT
    }

    /// The value of document number `docnum` (in range)
    pub(crate) fn get(&self, docnum: u64) -> Option<&str> {
        let i = docnum as usize;
        let entry = self.entry(i);
        if entry & PRESENT == 0 {
            return None;
        }
        let start = if i == 0 { 0 } else { self.end(i - 1) };
        let end = entry & !PRESENT;
        let dat = self.dat();
        if start > end || end > dat.len() as u64 {
            return None;
        }
        std::str::from_utf8(&dat[start as usize..end as usize]).ok()
    }
}

/// Builds a key-only document store in memory (no content, no zstd): for
/// mapping external document keys (e.g. `docno`) to index document ids and
/// back, including in WASM
#[derive(Default)]
pub struct KeyStoreBuilder {
    /// Values per key name, in order of first appearance
    keys: Vec<(String, Vec<(String, DocId)>)>,
    num_documents: DocId,
}

impl KeyStoreBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of document numbers used so far (largest + 1)
    pub fn num_documents(&self) -> DocId {
        self.num_documents
    }

    /// Adds a document with the next document number, which is returned
    pub fn add(&mut self, keys: &HashMap<String, String>) -> io::Result<DocId> {
        let docid = self.num_documents;
        self.add_with_docid(docid, keys)?;
        Ok(docid)
    }

    /// Adds a document with a given document number (e.g. its id in an
    /// index), larger than the previous ones
    pub fn add_with_docid(
        &mut self,
        docid: DocId,
        keys: &HashMap<String, String>,
    ) -> io::Result<()> {
        if docid < self.num_documents {
            return Err(invalid_input(format!(
                "Document numbers must be increasing ({docid} after {})",
                self.num_documents - 1
            )));
        }
        // Sorted, so that key names appear in a deterministic order
        let mut keys: Vec<(&String, &String)> = keys.iter().collect();
        keys.sort();
        for (name, value) in keys {
            match self.keys.iter_mut().find(|(n, _)| n == name) {
                Some((_, values)) => values.push((value.clone(), docid)),
                None => self.keys.push((name.clone(), vec![(value.clone(), docid)])),
            }
        }
        self.num_documents = docid + 1;
        Ok(())
    }

    /// Writes the store into the directory `dir` (created if needed)
    pub fn build(self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        self.build_to(&DirOutput::new(dir))
    }

    /// Writes the store files into `out`
    pub fn build_to<O: IndexOutput>(self, out: &O) -> io::Result<()> {
        let key_names = self.keys.iter().map(|(n, _)| n.clone()).collect();
        for (name, values) in self.keys {
            write_key(out, &name, values, self.num_documents)?;
        }
        write_meta(
            out,
            &DocumentStoreMeta {
                num_documents: self.num_documents,
                block_size: 0,
                num_blocks: 0,
                key_names,
                version: DOCSTORE_VERSION,
                content: false,
            },
        )
    }
}
