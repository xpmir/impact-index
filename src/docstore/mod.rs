//! Document stores: documents (keys and content) by document number, and
//! document numbers by key.
//!
//! Files of a store directory:
//!
//! - `docstore.cbor`: [`DocumentStoreMeta`]
//! - `key_<name>.fst`: key value → document number, per key name
//! - `key_<name>.idx` / `key_<name>.dat` (version 2): document number → key
//!   value, uncompressed (see [`keys`])
//! - `content.dat`, `blocks.dat`, `offsets.dat`: zstd-compressed blocks of
//!   documents (keys and content), absent from key-only stores
//!
//! Key-only stores, and key lookups in any version 2 store, need neither
//! zstd nor a file system (WASM).

pub mod builder;
pub mod keys;
pub mod store;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Document data for building a document store (keys + content, no ID).
#[derive(Debug, Clone)]
pub struct DocumentData {
    pub keys: HashMap<String, String>,
    pub content: Vec<u8>,
}

/// A document retrieved from a store, with its internal sequential ID.
#[derive(Debug, Clone)]
pub struct Document {
    /// The internal sequential ID (0-based).
    pub internal_id: u64,
    pub keys: HashMap<String, String>,
    pub content: Vec<u8>,
}

/// Fixed-size record for block metadata in blocks.dat.
/// Each record is 20 bytes: offset(8) + compressed_size(8) + num_docs(4).
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct BlockMeta {
    pub offset: u64,
    pub compressed_size: u64,
    pub num_docs: u32,
}

impl BlockMeta {
    pub const SIZE: usize = 20; // 8 + 8 + 4

    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= Self::SIZE);
        let offset = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let compressed_size = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let num_docs = u32::from_le_bytes(data[16..20].try_into().unwrap());
        BlockMeta {
            offset,
            compressed_size,
            num_docs,
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.compressed_size.to_le_bytes());
        buf[16..20].copy_from_slice(&self.num_docs.to_le_bytes());
        buf
    }
}

/// Metadata stored in docstore.cbor.
#[derive(Serialize, Deserialize, Debug)]
pub struct DocumentStoreMeta {
    /// Number of document numbers (largest number + 1; with
    /// [`builder::DocumentStoreBuilder::add_with_docid`], some numbers may
    /// have no document)
    pub num_documents: u64,
    pub block_size: usize,
    pub num_blocks: usize,
    pub key_names: Vec<String>,
    /// Format version: 1 (no field, before the key tables), or
    /// [`DOCSTORE_VERSION`]
    #[serde(default = "version_1")]
    pub version: u32,
    /// Whether documents (and their content) are stored in blocks; `false`
    /// for key-only stores
    #[serde(default = "default_true")]
    pub content: bool,
}

fn version_1() -> u32 {
    1
}

fn default_true() -> bool {
    true
}

/// Current document store format version: adds the document number → key
/// tables, key-only stores and missing document numbers. Version 1 stores
/// are still read.
pub const DOCSTORE_VERSION: u32 = 2;

/// Block index (in `offsets.dat`) of a document number without document
pub const ABSENT_BLOCK: u32 = u32::MAX;

// File name constants
pub const META_FILE: &str = "docstore.cbor";
pub const CONTENT_FILE: &str = "content.dat";
pub const BLOCKS_FILE: &str = "blocks.dat";
pub const OFFSETS_FILE: &str = "offsets.dat";
pub const CHECKPOINT_FILE: &str = "docstore_checkpoint.cbor";

pub fn key_fst_file(key_name: &str) -> String {
    format!("key_{}.fst", key_name)
}

pub fn key_index_file(key_name: &str) -> String {
    format!("key_{}.idx", key_name)
}

pub fn key_values_file(key_name: &str) -> String {
    format!("key_{}.dat", key_name)
}
