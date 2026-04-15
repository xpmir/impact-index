pub mod builder;
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
    pub num_documents: u64,
    pub block_size: usize,
    pub num_blocks: usize,
    pub key_names: Vec<String>,
}

// File name constants
pub const META_FILE: &str = "docstore.cbor";
pub const CONTENT_FILE: &str = "content.dat";
pub const BLOCKS_FILE: &str = "blocks.dat";
pub const OFFSETS_FILE: &str = "offsets.dat";
pub const CHECKPOINT_FILE: &str = "docstore_checkpoint.cbor";

pub fn key_fst_file(key_name: &str) -> String {
    format!("key_{}.fst", key_name)
}
