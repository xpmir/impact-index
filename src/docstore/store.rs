use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use crate::base::BoxResult;
use crate::storage::{DirFiles, IndexFiles};
use crate::utils::buffer::Buffer;

use super::keys::KeyTable;
use super::{
    key_fst_file, BlockMeta, Document, DocumentStoreMeta, ABSENT_BLOCK, BLOCKS_FILE, CONTENT_FILE,
    DOCSTORE_VERSION, META_FILE, OFFSETS_FILE,
};

/// How content.dat (and the key tables) are accessed.
#[derive(Debug, Clone, Copy)]
pub enum ContentAccess {
    /// Load entire file into RAM (fastest, high RAM).
    Memory,
    /// Memory-mapped (OS manages paging, low RAM).
    Mmap,
    /// Read blocks from disk on demand via seek+read (lowest RAM); the key
    /// tables are memory-mapped.
    Disk,
}

enum ContentBackend {
    Buffer(Box<dyn Buffer>),
    Disk(Mutex<File>),
}

/// Compressed documents (absent from key-only stores)
struct Content {
    backend: ContentBackend,
    blocks_data: Vec<u8>,
    offsets_data: Vec<u8>,
}

/// Read-only document store with block-based zstd compression.
pub struct DocumentStore {
    meta: DocumentStoreMeta,
    content: Option<Content>,
    key_fsts: HashMap<String, fst::Map<Vec<u8>>>,
    /// Document number → key value (format version 2)
    key_tables: HashMap<String, KeyTable>,
}

#[cfg(feature = "zstd")]
fn decompress(data: &[u8]) -> BoxResult<Vec<u8>> {
    Ok(zstd::decode_all(data)?)
}

#[cfg(not(feature = "zstd"))]
fn decompress(_data: &[u8]) -> BoxResult<Vec<u8>> {
    Err("Reading document contents requires the `zstd` feature".into())
}

impl DocumentStore {
    /// Load a document store from disk.
    pub fn load(dir: &Path, access: ContentAccess) -> BoxResult<Self> {
        Self::load_from(&DirFiles::new(dir), access)
    }

    /// Load a document store from its files (e.g. in memory, see
    /// [`crate::storage::MemoryFiles`]); `Disk` access needs files from a
    /// directory, and falls back to `Memory` otherwise.
    pub fn load_from(files: &dyn IndexFiles, access: ContentAccess) -> BoxResult<Self> {
        let meta: DocumentStoreMeta = ciborium::de::from_reader(files.open(META_FILE)?)
            .map_err(|e| format!("Failed to read metadata: {}", e))?;
        if meta.version > DOCSTORE_VERSION {
            return Err(format!(
                "{} has format version {}, but this version of impact-index reads up to \
                 version {}: upgrade impact-index",
                files.describe(META_FILE),
                meta.version,
                DOCSTORE_VERSION
            )
            .into());
        }
        let in_memory = matches!(access, ContentAccess::Memory);

        let content = if meta.content {
            let backend = match (access, files.dir()) {
                (ContentAccess::Disk, Some(dir)) => {
                    ContentBackend::Disk(Mutex::new(File::open(dir.join(CONTENT_FILE))?))
                }
                _ => ContentBackend::Buffer(files.buffer(CONTENT_FILE, in_memory)?),
            };
            Some(Content {
                backend,
                blocks_data: files.read_all(BLOCKS_FILE)?,
                offsets_data: files.read_all(OFFSETS_FILE)?,
            })
        } else {
            None
        };

        let mut key_fsts = HashMap::new();
        let mut key_tables = HashMap::new();
        for key_name in &meta.key_names {
            let fst_data = files.read_all(&key_fst_file(key_name))?;
            let map = fst::Map::new(fst_data)
                .map_err(|e| format!("Failed to load FST for key '{}': {}", key_name, e))?;
            key_fsts.insert(key_name.clone(), map);
            if meta.version >= 2 {
                let table = KeyTable::load(files, key_name, meta.num_documents, in_memory)?;
                key_tables.insert(key_name.clone(), table);
            }
        }

        Ok(Self {
            meta,
            content,
            key_fsts,
            key_tables,
        })
    }

    /// Number of document numbers (largest + 1)
    pub fn num_documents(&self) -> u64 {
        self.meta.num_documents
    }

    /// Available key names.
    pub fn key_names(&self) -> &[String] {
        &self.meta.key_names
    }

    /// Whether the store holds document contents (`false` for key-only
    /// stores)
    pub fn has_content(&self) -> bool {
        self.content.is_some()
    }

    #[inline]
    fn get_block_index(content: &Content, doc_number: u64) -> u32 {
        let offset = (doc_number as usize) * 4;
        u32::from_le_bytes(content.offsets_data[offset..offset + 4].try_into().unwrap())
    }

    #[inline]
    fn get_intra_offset(&self, content: &Content, doc_number: u64) -> u32 {
        let n = self.meta.num_documents as usize;
        let offset = n * 4 + (doc_number as usize) * 4;
        u32::from_le_bytes(content.offsets_data[offset..offset + 4].try_into().unwrap())
    }

    #[inline]
    fn get_block_meta(content: &Content, block_index: u32) -> BlockMeta {
        let offset = (block_index as usize) * BlockMeta::SIZE;
        BlockMeta::from_bytes(&content.blocks_data[offset..])
    }

    fn decompress_block(content: &Content, block_meta: &BlockMeta) -> BoxResult<Vec<u8>> {
        let offset = block_meta.offset as usize;
        let size = block_meta.compressed_size as usize;

        match &content.backend {
            ContentBackend::Buffer(buffer) => {
                let data = buffer.as_bytes().expect("buffers expose their bytes");
                decompress(&data[offset..offset + size])
            }
            ContentBackend::Disk(file_mutex) => {
                let mut file = file_mutex
                    .lock()
                    .map_err(|e| format!("Lock error: {}", e))?;
                let mut buf = vec![0u8; size];
                file.seek(SeekFrom::Start(block_meta.offset))?;
                file.read_exact(&mut buf)?;
                decompress(&buf[..])
            }
        }
    }

    #[inline]
    fn decode_document_at(
        decompressed: &[u8],
        intra_offset: u32,
    ) -> BoxResult<(HashMap<String, String>, Vec<u8>)> {
        let mut pos = intra_offset as usize;

        let keys_len = u64::from_le_bytes(decompressed[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let keys: HashMap<String, String> =
            bincode::deserialize(&decompressed[pos..pos + keys_len])?;
        pos += keys_len;

        let content_len =
            u64::from_le_bytes(decompressed[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let content = decompressed[pos..pos + content_len].to_vec();

        Ok((keys, content))
    }

    /// Whether a document has the number `doc_number`
    pub fn contains(&self, doc_number: u64) -> bool {
        if doc_number >= self.meta.num_documents {
            return false;
        }
        match &self.content {
            Some(content) => Self::get_block_index(content, doc_number) != ABSENT_BLOCK,
            // Key-only: documents are known by their keys
            None => self
                .key_tables
                .values()
                .any(|t| t.get(doc_number).is_some()),
        }
    }

    /// Keys of a document from the key tables
    fn keys_from_tables(&self, doc_number: u64) -> HashMap<String, String> {
        self.key_tables
            .iter()
            .filter_map(|(name, table)| table.get(doc_number).map(|v| (name.clone(), v.into())))
            .collect()
    }

    /// Retrieve documents by their numbers (0-based); key-only stores
    /// return documents with an empty content.
    pub fn get_by_number(&self, doc_numbers: &[u64]) -> BoxResult<Vec<Document>> {
        for &num in doc_numbers {
            if !self.contains(num) {
                return Err(format!(
                    "No document with number {} (num_documents={})",
                    num, self.meta.num_documents
                )
                .into());
            }
        }

        let Some(content) = &self.content else {
            return Ok(doc_numbers
                .iter()
                .map(|&n| Document {
                    internal_id: n,
                    keys: self.keys_from_tables(n),
                    content: Vec::new(),
                })
                .collect());
        };

        // Fast path for single document retrieval (avoids HashMap overhead)
        if doc_numbers.len() == 1 {
            let doc_num = doc_numbers[0];
            let block_index = Self::get_block_index(content, doc_num);
            let intra_offset = self.get_intra_offset(content, doc_num);
            let block_meta = Self::get_block_meta(content, block_index);
            let decompressed = Self::decompress_block(content, &block_meta)?;
            let (keys, content) = Self::decode_document_at(&decompressed, intra_offset)?;
            return Ok(vec![Document {
                internal_id: doc_num,
                keys,
                content,
            }]);
        }

        // Group by block index to minimize decompression
        let mut block_groups: HashMap<u32, Vec<(usize, u64, u32)>> = HashMap::new();
        for (result_idx, &doc_num) in doc_numbers.iter().enumerate() {
            let block_index = Self::get_block_index(content, doc_num);
            let intra_offset = self.get_intra_offset(content, doc_num);
            block_groups
                .entry(block_index)
                .or_default()
                .push((result_idx, doc_num, intra_offset));
        }

        let mut results: Vec<Option<Document>> = vec![None; doc_numbers.len()];

        for (block_index, docs_in_block) in &block_groups {
            let block_meta = Self::get_block_meta(content, *block_index);
            let decompressed = Self::decompress_block(content, &block_meta)?;

            for &(result_idx, doc_num, intra_offset) in docs_in_block {
                let (keys, content) = Self::decode_document_at(&decompressed, intra_offset)?;
                results[result_idx] = Some(Document {
                    internal_id: doc_num,
                    keys,
                    content,
                });
            }
        }

        Ok(results.into_iter().map(|d| d.unwrap()).collect())
    }

    /// Document numbers of the given key values (`None` if not found)
    pub fn get_numbers(&self, key_name: &str, key_values: &[&str]) -> BoxResult<Vec<Option<u64>>> {
        let fst = self
            .key_fsts
            .get(key_name)
            .ok_or_else(|| format!("Unknown key name '{}'", key_name))?;
        Ok(key_values.iter().map(|v| fst.get(v)).collect())
    }

    /// Values of the key `key_name` for the given document numbers (`None`
    /// if there is no such document, or it has no value for the key).
    ///
    /// Reads the key tables; for stores written before format version 2,
    /// which have none, decompresses the documents instead.
    pub fn get_keys(&self, key_name: &str, doc_numbers: &[u64]) -> BoxResult<Vec<Option<String>>> {
        if !self.key_fsts.contains_key(key_name) {
            return Err(format!("Unknown key name '{}'", key_name).into());
        }
        if let Some(table) = self.key_tables.get(key_name) {
            return Ok(doc_numbers
                .iter()
                .map(|&n| {
                    (n < self.meta.num_documents)
                        .then(|| table.get(n).map(String::from))
                        .flatten()
                })
                .collect());
        }
        let present: Vec<u64> = doc_numbers
            .iter()
            .copied()
            .filter(|&n| self.contains(n))
            .collect();
        let mut docs = self.get_by_number(&present)?.into_iter();
        Ok(doc_numbers
            .iter()
            .map(|&n| {
                if self.contains(n) {
                    docs.next().and_then(|mut d| d.keys.remove(key_name))
                } else {
                    None
                }
            })
            .collect())
    }

    /// Retrieve documents by key name and values.
    /// Returns one Option<Document> per input value (None if not found).
    pub fn get_by_key(
        &self,
        key_name: &str,
        key_values: &[&str],
    ) -> BoxResult<Vec<Option<Document>>> {
        let fst = self
            .key_fsts
            .get(key_name)
            .ok_or_else(|| format!("Unknown key name '{}'", key_name))?;

        let mut found_indices: Vec<(usize, u64)> = Vec::new();

        for (i, &val) in key_values.iter().enumerate() {
            if let Some(doc_num) = fst.get(val) {
                found_indices.push((i, doc_num));
            }
        }

        if found_indices.is_empty() {
            return Ok(vec![None; key_values.len()]);
        }

        let doc_nums: Vec<u64> = found_indices.iter().map(|&(_, num)| num).collect();
        let mut docs = self.get_by_number(&doc_nums)?;

        let mut results: Vec<Option<Document>> = vec![None; key_values.len()];
        // Iterate in reverse to pop from the end (avoids shifting)
        for (found_pos, &(input_idx, _)) in found_indices.iter().enumerate().rev() {
            let doc = std::mem::replace(
                &mut docs[found_pos],
                Document {
                    internal_id: 0,
                    keys: HashMap::new(),
                    content: Vec::new(),
                },
            );
            results[input_idx] = Some(doc);
        }

        Ok(results)
    }
}
