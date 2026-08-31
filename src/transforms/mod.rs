//! Index transformations that produce new index representations on disk.
//!
//! Transforms read from a [`SparseIndexView`] and write a new index to a
//! specified directory. Available transforms:
//! - [`split::SplitIndexTransform`]: Splits posting lists by impact quantiles
//!   for use with term impact decomposition algorithms
//! - [`reorder::ReorderTransform`]: Renumbers document ids by recursive
//!   graph bisection so similar documents get adjacent ids (P2)

use std::path::Path;

pub mod reorder;
pub mod split;
use super::index::SparseIndexView;

/// Trait for index transformations that read an existing index and produce
/// a new one on disk.
pub trait IndexTransform: Send + Sync {
    /// Applies the transform, writing the result to `path`.
    fn process(&self, path: &Path, index: &dyn SparseIndexView) -> Result<(), std::io::Error>;
}
