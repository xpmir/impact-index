//! Core type definitions and traits used throughout the library.

use std::collections::HashMap;
use std::fmt;
use std::io::{Read as IoRead, Write};
use std::{fs::File, path::Path};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::builder::load_forward_index_dynamic;
use crate::index::{BlockTermImpactIterator, SparseIndex};
use crate::search::ScoredDocument;

/// Index identifying a term in the vocabulary (0-based).
pub type TermIndex = usize;

/// Floating-point impact score assigned to a term in a document.
/// Must be strictly positive (> 0) when indexing.
pub type ImpactValue = f32;

/// Unique document identifier (monotonically increasing during indexing).
pub type DocId = u64;

/// Convenience alias for boxed error results.
pub type BoxResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Trait for objects that have a countable length (e.g., number of terms).
pub trait Len {
    /// Returns the number of elements.
    fn len(&self) -> usize;
}

pub const INDEX_CBOR: &str = "index.cbor";
pub const BUILDER_INDEX_CBOR: &str = "information.cbor";

// --- PostingValue trait and implementations ---

/// Trait for numeric types that can be used as posting impact values.
///
/// The index builder and forward index are generic over this type,
/// while search algorithms always work with `ImpactValue` (f32).
/// Conversion happens at the `SparseIndex` trait boundary.
pub trait PostingValue:
    Copy + PartialOrd + Send + Sync + Serialize + for<'de> Deserialize<'de> + fmt::Display + 'static
{
    /// Convert to f32 for use in search algorithms.
    fn to_f32(self) -> f32;

    /// Check if the value is strictly positive.
    fn is_positive(self) -> bool;

    /// Negative infinity (or minimum representable value for integer types).
    fn neg_infinity() -> Self;

    /// Size of this value in bytes when serialized.
    const BYTE_SIZE: usize;

    /// Write this value in big-endian format.
    fn write_be(self, writer: &mut dyn Write) -> std::io::Result<()>;

    /// Read a value in big-endian format.
    fn read_be(reader: &mut &[u8]) -> Self;
}

impl PostingValue for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn is_positive(self) -> bool {
        self > 0.0
    }
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    const BYTE_SIZE: usize = 4;
    fn write_be(self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_f32::<BigEndian>(self)
    }
    fn read_be(reader: &mut &[u8]) -> Self {
        reader.read_f32::<BigEndian>().expect("read error")
    }
}

impl PostingValue for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn is_positive(self) -> bool {
        self > 0.0
    }
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    const BYTE_SIZE: usize = 8;
    fn write_be(self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_f64::<BigEndian>(self)
    }
    fn read_be(reader: &mut &[u8]) -> Self {
        reader.read_f64::<BigEndian>().expect("read error")
    }
}

impl PostingValue for f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn is_positive(self) -> bool {
        self.to_f32() > 0.0
    }
    fn neg_infinity() -> Self {
        f16::NEG_INFINITY
    }
    const BYTE_SIZE: usize = 2;
    fn write_be(self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_u16::<BigEndian>(self.to_bits())
    }
    fn read_be(reader: &mut &[u8]) -> Self {
        f16::from_bits(reader.read_u16::<BigEndian>().expect("read error"))
    }
}

impl PostingValue for bf16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn is_positive(self) -> bool {
        self.to_f32() > 0.0
    }
    fn neg_infinity() -> Self {
        bf16::NEG_INFINITY
    }
    const BYTE_SIZE: usize = 2;
    fn write_be(self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_u16::<BigEndian>(self.to_bits())
    }
    fn read_be(reader: &mut &[u8]) -> Self {
        bf16::from_bits(reader.read_u16::<BigEndian>().expect("read error"))
    }
}

impl PostingValue for i32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn is_positive(self) -> bool {
        self > 0
    }
    fn neg_infinity() -> Self {
        i32::MIN
    }
    const BYTE_SIZE: usize = 4;
    fn write_be(self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_i32::<BigEndian>(self)
    }
    fn read_be(reader: &mut &[u8]) -> Self {
        reader.read_i32::<BigEndian>().expect("read error")
    }
}

impl PostingValue for i64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn is_positive(self) -> bool {
        self > 0
    }
    fn neg_infinity() -> Self {
        i64::MIN
    }
    const BYTE_SIZE: usize = 8;
    fn write_be(self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_i64::<BigEndian>(self)
    }
    fn read_be(reader: &mut &[u8]) -> Self {
        reader.read_i64::<BigEndian>().expect("read error")
    }
}

/// Tag identifying the value type stored on disk, for type-erased loading.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum ValueType {
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
}

/// Returns the `ValueType` tag for a given `PostingValue` implementor.
pub fn value_type_of<V: PostingValue>() -> ValueType {
    use std::any::TypeId;
    let id = TypeId::of::<V>();
    if id == TypeId::of::<f32>() {
        ValueType::F32
    } else if id == TypeId::of::<f64>() {
        ValueType::F64
    } else if id == TypeId::of::<f16>() {
        ValueType::F16
    } else if id == TypeId::of::<bf16>() {
        ValueType::BF16
    } else if id == TypeId::of::<i32>() {
        ValueType::I32
    } else if id == TypeId::of::<i64>() {
        ValueType::I64
    } else {
        panic!("Unknown PostingValue type")
    }
}

// --- Core data structures ---

/// A generic term impact = document ID + value of type V
#[derive(Serialize, Deserialize, Clone, Copy)]
#[serde(bound(deserialize = "V: PostingValue"))]
pub struct GenericTermImpact<V: PostingValue> {
    pub docid: DocId,
    pub value: V,
}

impl<V: PostingValue> fmt::Display for GenericTermImpact<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({},{})", self.docid, self.value)
    }
}

/// Term impact = document ID + impact value (f32)
pub type TermImpact = GenericTermImpact<ImpactValue>;

impl TermImpact {
    /// Create from a generic term impact by converting the value to f32.
    pub fn from_generic<V: PostingValue>(g: GenericTermImpact<V>) -> Self {
        TermImpact {
            docid: g.docid,
            value: g.value.to_f32(),
        }
    }
}

/// An iterator on term impacts
pub type TermImpactIterator<'a> = Box<dyn BlockTermImpactIterator + 'a>;

/// A search function
pub type SearchFn = fn(
    index: &dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument>;

/// Trait for deserializing and instantiating a [`SparseIndex`] from disk.
///
/// Implementations are stored alongside the index data (serialized via CBOR)
/// so the correct loader is automatically selected when loading.
#[typetag::serde(tag = "type")]
pub trait IndexLoader {
    /// Consumes the loader and returns a ready-to-query index.
    fn into_index(self: Box<Self>, path: &Path, in_memory: bool) -> Box<dyn SparseIndex>;
}

/// Loads a sparse index from the given directory.
///
/// Supports both legacy forward-index format (`information.cbor`) and
/// the newer loader-based format (`index.cbor`).
///
/// # Arguments
///
/// * `path` - Directory containing the index files
/// * `in_memory` - If `true`, loads data into memory; otherwise uses memory-mapped I/O
pub fn load_index(path: &Path, in_memory: bool) -> Box<dyn SparseIndex> {
    let info_path = path.join(BUILDER_INDEX_CBOR);
    if info_path.exists() {
        // Takes care of old/new format with the raw builder index
        load_forward_index_dynamic(path, in_memory)
    } else {
        // Uses the new way to load indices
        let info_path = path.join(INDEX_CBOR);
        let info_file = File::options()
            .read(true)
            .open(info_path)
            .expect("Error while opening the index information file");

        let loader: Box<dyn IndexLoader> = ciborium::de::from_reader(info_file)
            .expect("Error loading compressed term index information");

        loader.into_index(path, in_memory)
    }
}

/// Saves an index loader to disk so the index can be loaded later.
///
/// Serializes the loader as CBOR into `index.cbor` within the given directory.
pub fn save_index(loader: Box<dyn IndexLoader>, path: &Path) -> Result<(), std::io::Error> {
    use std::io::BufWriter;

    let info_path = path.join(INDEX_CBOR);
    let info_path_s = info_path.display().to_string();

    let info_file = File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(info_path)
        .expect(&format!("Error while creating file {}", info_path_s));

    eprintln!("Writing index metadata to {}...", info_path_s);
    let writer = BufWriter::new(info_file);
    ciborium::ser::into_writer(&loader, writer)
        .expect("Error saving compressed term index information");
    eprintln!("Done writing index metadata.");

    Ok(())
}
