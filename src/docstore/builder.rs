use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use log::info;
use serde::{Deserialize, Serialize};

use crate::base::BoxResult;

use super::{
    BlockMeta, DocumentData, DocumentStoreMeta, BLOCKS_FILE, CHECKPOINT_FILE, CONTENT_FILE,
    META_FILE, OFFSETS_FILE,
};

/// Configuration options for the [`DocumentStoreBuilder`].
#[derive(Clone, Debug)]
pub struct BuilderOptions {
    pub block_size: usize,
    pub zstd_level: i32,
    /// Build a checkpoint every N documents (0 disables checkpointing).
    ///
    /// Checkpoints allow resuming after a crash by persisting the current
    /// in-flight state (open block, offsets, key temp files) to disk.
    pub checkpoint_frequency: u64,
}

impl Default for BuilderOptions {
    fn default() -> Self {
        Self {
            block_size: 4096,
            zstd_level: 3,
            checkpoint_frequency: 0,
        }
    }
}

/// On-disk checkpoint state. Written atomically via a `.tmp` rename.
#[derive(Serialize, Deserialize)]
struct CheckpointState {
    block_size: usize,
    zstd_level: i32,
    num_blocks: u64,
    num_documents: u64,
    content_offset: u64,
    doc_block_indices: Vec<u32>,
    doc_intra_offsets: Vec<u32>,
    current_block_buf: Vec<u8>,
    current_block_doc_count: u32,
    /// Ordered list of key names and the temp-file length at checkpoint time
    /// (used to truncate any bytes written after the last checkpoint on resume).
    key_file_lengths: Vec<(String, u64)>,
}

fn tmp_key_path(dir: &Path, key_name: &str) -> PathBuf {
    dir.join(format!(".tmp_key_{}", key_name))
}

/// Builds a DocumentStore on disk with block-based zstd compression.
pub struct DocumentStoreBuilder {
    dir: PathBuf,
    options: BuilderOptions,
    current_block_buf: Vec<u8>,
    current_block_doc_count: u32,
    doc_block_indices: Vec<u32>,
    doc_intra_offsets: Vec<u32>,
    /// Temp files for each key: lines of "key_value\tdoc_num\n"
    key_files: HashMap<String, BufWriter<File>>,
    key_names: Vec<String>,
    data_file: BufWriter<File>,
    blocks_file: BufWriter<File>,
    num_blocks: u64,
    num_documents: u64,
    content_offset: u64,
    last_checkpoint_doc_count: u64,
}

impl DocumentStoreBuilder {
    /// Create a new builder writing to the given directory.
    pub fn new(dir: &Path, block_size: usize, zstd_level: i32) -> BoxResult<Self> {
        let options = BuilderOptions {
            block_size,
            zstd_level,
            checkpoint_frequency: 0,
        };
        Self::new_with_options(dir, &options)
    }

    /// Create a new builder with full options (including checkpointing).
    ///
    /// If `options.checkpoint_frequency > 0` and a checkpoint file exists in
    /// `dir`, the builder resumes from that state. Otherwise it starts fresh
    /// (truncating any existing output files).
    pub fn new_with_options(dir: &Path, options: &BuilderOptions) -> BoxResult<Self> {
        fs::create_dir_all(dir)?;

        let checkpoint_path = dir.join(CHECKPOINT_FILE);
        let resuming = options.checkpoint_frequency > 0 && checkpoint_path.exists();

        if resuming {
            Self::resume(dir, options, &checkpoint_path)
        } else {
            Self::fresh(dir, options)
        }
    }

    fn fresh(dir: &Path, options: &BuilderOptions) -> BoxResult<Self> {
        let data_file = BufWriter::new(
            File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dir.join(CONTENT_FILE))?,
        );
        let blocks_file = BufWriter::new(
            File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dir.join(BLOCKS_FILE))?,
        );

        // Clean up any stale checkpoint artifacts left over from an aborted
        // run whose checkpointing was disabled on retry.
        for name in [CHECKPOINT_FILE, &format!("{}.tmp", CHECKPOINT_FILE)] {
            let p = dir.join(name);
            if p.exists() {
                fs::remove_file(&p)?;
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            options: options.clone(),
            current_block_buf: Vec::new(),
            current_block_doc_count: 0,
            doc_block_indices: Vec::new(),
            doc_intra_offsets: Vec::new(),
            key_files: HashMap::new(),
            key_names: Vec::new(),
            data_file,
            blocks_file,
            num_blocks: 0,
            num_documents: 0,
            content_offset: 0,
            last_checkpoint_doc_count: 0,
        })
    }

    fn resume(dir: &Path, options: &BuilderOptions, checkpoint_path: &Path) -> BoxResult<Self> {
        let ckpt_file = File::open(checkpoint_path)?;
        let decoder = zstd::stream::Decoder::new(BufReader::new(ckpt_file))?;
        let state: CheckpointState = ciborium::de::from_reader(decoder)
            .map_err(|e| format!("Failed to read checkpoint: {}", e))?;

        if state.block_size != options.block_size {
            return Err(format!(
                "Checkpoint block_size {} does not match configured {}",
                state.block_size, options.block_size
            )
            .into());
        }

        // Truncate output files back to the checkpointed state, dropping any
        // bytes that may have been written after the last successful checkpoint.
        let content_path = dir.join(CONTENT_FILE);
        let blocks_path = dir.join(BLOCKS_FILE);

        let content_file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&content_path)?;
        content_file.set_len(state.content_offset)?;
        let mut data_file = BufWriter::new(content_file);
        data_file.seek(SeekFrom::Start(state.content_offset))?;

        let blocks_len = state.num_blocks * BlockMeta::SIZE as u64;
        let blocks_raw = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&blocks_path)?;
        blocks_raw.set_len(blocks_len)?;
        let mut blocks_file = BufWriter::new(blocks_raw);
        blocks_file.seek(SeekFrom::Start(blocks_len))?;

        // Reopen key temp files, truncating to the length captured at checkpoint.
        let mut key_files: HashMap<String, BufWriter<File>> = HashMap::new();
        let mut key_names: Vec<String> = Vec::with_capacity(state.key_file_lengths.len());
        for (name, len) in &state.key_file_lengths {
            let path = tmp_key_path(dir, name);
            let f = File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            f.set_len(*len)?;
            let mut w = BufWriter::new(f);
            w.seek(SeekFrom::Start(*len))?;
            key_files.insert(name.clone(), w);
            key_names.push(name.clone());
        }

        info!(
            "Resumed DocumentStoreBuilder from checkpoint: {} documents, {} blocks",
            state.num_documents, state.num_blocks
        );

        Ok(Self {
            dir: dir.to_path_buf(),
            options: options.clone(),
            current_block_buf: state.current_block_buf,
            current_block_doc_count: state.current_block_doc_count,
            doc_block_indices: state.doc_block_indices,
            doc_intra_offsets: state.doc_intra_offsets,
            key_files,
            key_names,
            data_file,
            blocks_file,
            num_blocks: state.num_blocks,
            num_documents: state.num_documents,
            content_offset: state.content_offset,
            last_checkpoint_doc_count: state.num_documents,
        })
    }

    /// Number of documents successfully added so far.
    pub fn num_documents(&self) -> u64 {
        self.num_documents
    }

    /// Add a document to the store.
    pub fn add(&mut self, doc: &DocumentData) -> BoxResult<()> {
        let block_index = self.num_blocks as u32;
        let intra_offset = self.current_block_buf.len() as u32;

        self.doc_block_indices.push(block_index);
        self.doc_intra_offsets.push(intra_offset);

        // Serialize into current_block_buf:
        // keys (bincode) + content_len (u64 LE) + content bytes
        let keys_bytes = bincode::serialize(&doc.keys)?;
        let keys_len = keys_bytes.len() as u64;
        self.current_block_buf
            .extend_from_slice(&keys_len.to_le_bytes());
        self.current_block_buf.extend_from_slice(&keys_bytes);

        let content_len = doc.content.len() as u64;
        self.current_block_buf
            .extend_from_slice(&content_len.to_le_bytes());
        self.current_block_buf.extend_from_slice(&doc.content);

        // Write key entries to temp files
        let doc_num = self.num_documents;
        for (key_name, key_value) in &doc.keys {
            if !self.key_files.contains_key(key_name) {
                let temp_path = tmp_key_path(&self.dir, key_name);
                let file = BufWriter::new(
                    File::options()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&temp_path)?,
                );
                self.key_files.insert(key_name.clone(), file);
                self.key_names.push(key_name.clone());
            }
            let writer = self.key_files.get_mut(key_name).unwrap();
            writeln!(writer, "{}\t{}", key_value, doc_num)?;
        }

        self.num_documents += 1;
        self.current_block_doc_count += 1;

        if self.current_block_buf.len() >= self.options.block_size {
            self.flush_block()?;
        }

        if self.options.checkpoint_frequency > 0
            && self.num_documents - self.last_checkpoint_doc_count
                >= self.options.checkpoint_frequency
        {
            self.checkpoint()?;
        }

        Ok(())
    }

    fn flush_block(&mut self) -> BoxResult<()> {
        if self.current_block_buf.is_empty() {
            return Ok(());
        }

        let compressed = zstd::encode_all(&self.current_block_buf[..], self.options.zstd_level)?;

        self.data_file.write_all(&compressed)?;

        let block_meta = BlockMeta {
            offset: self.content_offset,
            compressed_size: compressed.len() as u64,
            num_docs: self.current_block_doc_count,
        };
        self.blocks_file.write_all(&block_meta.to_bytes())?;

        self.content_offset += compressed.len() as u64;
        self.num_blocks += 1;
        self.current_block_buf.clear();
        self.current_block_doc_count = 0;

        Ok(())
    }

    /// Persist the current in-flight state atomically so that a later
    /// [`new_with_options`](Self::new_with_options) call can resume from here.
    pub fn checkpoint(&mut self) -> BoxResult<()> {
        info!(
            "Checkpointing DocumentStoreBuilder at {} documents",
            self.num_documents
        );

        // Flush buffered writes so what's on disk matches our tracked offsets.
        self.data_file.flush()?;
        self.blocks_file.flush()?;

        // Capture key temp-file lengths (after flushing their BufWriters).
        let mut key_file_lengths: Vec<(String, u64)> = Vec::with_capacity(self.key_names.len());
        for name in &self.key_names {
            let writer = self.key_files.get_mut(name).expect("key file missing");
            writer.flush()?;
            let len = writer.get_ref().metadata()?.len();
            key_file_lengths.push((name.clone(), len));
        }

        let state = CheckpointState {
            block_size: self.options.block_size,
            zstd_level: self.options.zstd_level,
            num_blocks: self.num_blocks,
            num_documents: self.num_documents,
            content_offset: self.content_offset,
            doc_block_indices: self.doc_block_indices.clone(),
            doc_intra_offsets: self.doc_intra_offsets.clone(),
            current_block_buf: self.current_block_buf.clone(),
            current_block_doc_count: self.current_block_doc_count,
            key_file_lengths,
        };

        let final_path = self.dir.join(CHECKPOINT_FILE);
        let tmp_path = self.dir.join(format!("{}.tmp", CHECKPOINT_FILE));

        {
            let tmp_file = File::options()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            let encoder = zstd::stream::Encoder::new(BufWriter::new(tmp_file), 3)?;
            let mut encoder = encoder.auto_finish();
            ciborium::ser::into_writer(&state, &mut encoder)
                .map_err(|e| format!("Failed to serialize checkpoint: {}", e))?;
        }

        fs::rename(&tmp_path, &final_path)?;
        self.last_checkpoint_doc_count = self.num_documents;
        Ok(())
    }

    /// Finalize and write all index files.
    pub fn build(mut self) -> BoxResult<()> {
        // Flush remaining block
        self.flush_block()?;

        // Flush data and blocks files
        self.data_file.flush()?;
        self.blocks_file.flush()?;

        // Write offsets.dat: block_indices then intra_offsets as contiguous u32 LE arrays
        {
            let mut offsets_file = BufWriter::new(
                File::options()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(self.dir.join(OFFSETS_FILE))?,
            );
            for &idx in &self.doc_block_indices {
                offsets_file.write_all(&idx.to_le_bytes())?;
            }
            for &off in &self.doc_intra_offsets {
                offsets_file.write_all(&off.to_le_bytes())?;
            }
            offsets_file.flush()?;
        }

        // Build FSTs for each key
        // Close temp files first
        let key_names = self.key_names.clone();
        for name in &key_names {
            self.key_files.remove(name); // drops and closes
        }

        for key_name in &key_names {
            let temp_path = tmp_key_path(&self.dir, key_name);

            // Read all entries
            let reader = BufReader::new(File::open(&temp_path)?);
            let mut entries: Vec<(String, u64)> = Vec::new();
            for line in reader.lines() {
                let line = line?;
                let mut parts = line.splitn(2, '\t');
                let key_value = parts
                    .next()
                    .ok_or("invalid temp key file format")?
                    .to_string();
                let doc_num: u64 = parts
                    .next()
                    .ok_or("invalid temp key file format")?
                    .parse()?;
                entries.push((key_value, doc_num));
            }

            // Sort by key value
            entries.sort_by(|a, b| a.0.cmp(&b.0));

            // Check for duplicates and build FST
            let fst_path = self.dir.join(super::key_fst_file(key_name));
            let fst_file = BufWriter::new(
                File::options()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&fst_path)?,
            );
            let mut fst_builder = fst::MapBuilder::new(fst_file)?;

            for i in 0..entries.len() {
                if i > 0 && entries[i].0 == entries[i - 1].0 {
                    return Err(format!(
                        "Duplicate key value '{}' for key '{}'",
                        entries[i].0, key_name
                    )
                    .into());
                }
                fst_builder.insert(&entries[i].0, entries[i].1)?;
            }
            fst_builder.finish()?;

            // Clean up temp file
            fs::remove_file(&temp_path)?;
        }

        // Write metadata
        let meta = DocumentStoreMeta {
            num_documents: self.num_documents,
            block_size: self.options.block_size,
            num_blocks: self.num_blocks as usize,
            key_names,
        };
        let meta_file = File::options()
            .write(true)
            .create(true)
            .truncate(true)
            .open(self.dir.join(META_FILE))?;
        ciborium::ser::into_writer(&meta, meta_file)
            .map_err(|e| format!("Failed to write metadata: {}", e))?;

        // Remove checkpoint artifacts now that the build is durable.
        for name in [CHECKPOINT_FILE, &format!("{}.tmp", CHECKPOINT_FILE)] {
            let p = self.dir.join(name);
            if p.exists() {
                fs::remove_file(&p)?;
            }
        }

        Ok(())
    }
}
