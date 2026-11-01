//! Document metadata storage (e.g., document lengths for BM25).
//!
//! Stores dense per-document metadata indexed by document ID (0-based contiguous).
//! Currently supports document lengths as `u32` values.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

/// Metadata about the document metadata file format.
#[derive(Serialize, Deserialize)]
pub struct DocMetadataInfo {
    /// Number of documents stored.
    pub num_docs: u64,
    /// Field names (currently just ["doc_length"]).
    pub fields: Vec<String>,
}

/// In-memory document metadata, providing per-document lengths.
pub struct DocMetadata {
    /// Document lengths indexed by docid (0-based).
    pub doc_lengths: Vec<u32>,
}

impl DocMetadata {
    /// Create empty metadata.
    pub fn new() -> Self {
        Self {
            doc_lengths: Vec::new(),
        }
    }

    /// Create from a pre-built lengths vector.
    pub fn from_lengths(doc_lengths: Vec<u32>) -> Self {
        Self { doc_lengths }
    }

    /// Number of documents.
    pub fn num_docs(&self) -> u64 {
        self.doc_lengths.len() as u64
    }

    /// Average document length.
    pub fn avg_dl(&self) -> f32 {
        if self.doc_lengths.is_empty() {
            return 0.0;
        }
        let total: u64 = self.doc_lengths.iter().map(|&l| l as u64).sum();
        total as f32 / self.doc_lengths.len() as f32
    }

    /// Minimum document length (returns 0 if empty).
    pub fn min_dl(&self) -> u32 {
        self.doc_lengths.iter().copied().min().unwrap_or(0)
    }

    /// Save document metadata to a directory.
    ///
    /// Writes `docmeta.dat` (raw u32 LE) and `docmeta.cbor` (metadata).
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        // Write raw doc lengths
        let dat_path = dir.join("docmeta.dat");
        let mut writer = BufWriter::new(
            File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&dat_path)?,
        );
        for &length in &self.doc_lengths {
            writer.write_u32::<LittleEndian>(length)?;
        }
        writer.flush()?;

        // Write metadata info
        let info = DocMetadataInfo {
            num_docs: self.doc_lengths.len() as u64,
            fields: vec!["doc_length".to_string()],
        };
        let cbor_path = dir.join("docmeta.cbor");
        let cbor_file = File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&cbor_path)?;
        ciborium::ser::into_writer(&info, cbor_file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }

    /// Load document metadata from a directory.
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        // Read metadata info
        let cbor_path = dir.join("docmeta.cbor");
        let cbor_file = File::open(&cbor_path)?;
        let info: DocMetadataInfo = ciborium::de::from_reader(cbor_file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Read raw doc lengths
        let dat_path = dir.join("docmeta.dat");
        let mut reader = BufReader::new(File::open(&dat_path)?);
        let mut doc_lengths = Vec::with_capacity(info.num_docs as usize);
        for _ in 0..info.num_docs {
            doc_lengths.push(reader.read_u32::<LittleEndian>()?);
        }

        Ok(Self { doc_lengths })
    }

    /// Check if document metadata files exist in a directory.
    pub fn exists(dir: &Path) -> bool {
        dir.join("docmeta.dat").exists() && dir.join("docmeta.cbor").exists()
    }

    /// Copy (hard-link with fallback) document metadata files from source to destination.
    pub fn copy_files(src_dir: &Path, dst_dir: &Path) -> std::io::Result<()> {
        for filename in &["docmeta.dat", "docmeta.cbor"] {
            let src = src_dir.join(filename);
            let dst = dst_dir.join(filename);
            if src.exists() {
                crate::utils::fs::link_or_copy(&src, &dst)?;
            }
        }
        Ok(())
    }
}
