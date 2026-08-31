//! # impact-index
//!
//! A Rust library with Python bindings for efficient sparse retrieval from
//! neural information retrieval systems. Unlike traditional IR libraries, it
//! specifically targets neural IR models with floating-point impact scores
//! (no quantization assumed) and does not implement frequency-based algorithms.
//!
//! ## Main Components
//!
//! - [`builder`]: Index construction with checkpointing support
//! - [`index`]: Core traits for sparse indices and block-based iteration
//! - [`search`]: Search algorithms ([WAND](search::wand) and [MaxScore](search::maxscore))
//! - [`compress`]: Compression schemes for document IDs (Elias-Fano) and impact values
//! - [`transforms`]: Index transformations (e.g., splitting posting lists by quantile)
//! - [`docstore`]: Compressed document storage with key-based retrieval
//! - [`bmp`]: Integration with Block-Max Pruning (BMP) for fast approximate search
//!
//! ## Quick Start (Rust)
//!
//! ```no_run
//! use std::path::Path;
//! use std::collections::HashMap;
//! use impact_index::base::{load_index, ImpactValue, TermIndex};
//! use impact_index::search::wand::search_wand;
//!
//! // Load an index from disk
//! let index = load_index(Path::new("my_index"), true);
//!
//! // Build a query: term_id -> weight
//! let mut query: HashMap<TermIndex, ImpactValue> = HashMap::new();
//! query.insert(42, 1.5);
//! query.insert(100, 0.8);
//!
//! // Search for top-10 documents
//! let results = search_wand(&*index, &query, 10);
//! for doc in &results {
//!     println!("doc {} score {}", doc.docid, doc.score);
//! }
//! ```

pub mod base;
pub mod bmp;
pub mod bow;
pub mod builder;
pub mod compress;
pub mod docmeta;
pub mod docstore;
pub mod index;
pub mod manifest;
pub mod scoring;
pub mod search;
pub mod transforms;
pub mod vocab;

pub mod py;
mod utils;
