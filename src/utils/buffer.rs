use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

pub trait Slice: Send + Sync {
    fn data(&'_ self) -> &'_ [u8];
    fn read(&mut self, index: usize, buf: &mut [u8]) -> std::io::Result<usize>;
}

pub struct SliceReader<'a> {
    slice: Box<dyn Slice + 'a>,
    index: usize,
}

impl<'a> SliceReader<'a> {
    pub fn new(slice: Box<dyn Slice + 'a>) -> Self {
        SliceReader {
            slice: slice,
            index: 0,
        }
    }
}

impl<'a> Read for SliceReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let r = self.slice.read(self.index, buf);
        if let std::io::Result::Ok(read) = r {
            self.index += read;
        }
        r
    }
}

struct MemorySlice<'a> {
    _data: &'a [u8],
}

impl Slice for MemorySlice<'_> {
    fn data(&'_ self) -> &'_ [u8] {
        self._data
    }

    fn read(&mut self, index: usize, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut data = &self._data[index..];
        data.read(buf)
    }
}

pub trait Buffer: Send + Sync {
    fn slice(&'_ self, start: usize, end: usize) -> Box<dyn Slice + '_>;

    /// Direct access to the underlying byte buffer (if available).
    /// Returns None for memory-mapped buffers that don't hold contiguous data.
    fn as_bytes(&self) -> Option<&[u8]> {
        None
    }
}

/// Stores the data in memory
pub struct MemoryBuffer {
    data: Vec<u8>,
}

impl MemoryBuffer {
    pub fn new(path: &PathBuf) -> Self {
        let mut file = File::options()
            .read(true)
            .open(path)
            .expect("Error while reading posting file");

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .expect("Error while reading file");

        Self { data: data }
    }
}

impl Buffer for MemoryBuffer {
    fn slice(&'_ self, start: usize, end: usize) -> Box<dyn Slice + '_> {
        Box::new(MemorySlice {
            _data: &self.data[start..end],
        })
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        Some(&self.data)
    }
}

/// Uses a memory map
pub struct MmapBuffer {
    mmap: Mmap,
}

impl MmapBuffer {
    pub fn new(path: &PathBuf) -> Self {
        let file = File::options()
            .read(true)
            .open(path)
            .expect("Error while reading posting file");
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .expect("Cannot create a memory map")
        };
        Self { mmap: mmap }
    }
}

struct MmapSlice {
    vector: Vec<u8>,
}

impl Slice for MmapSlice {
    fn data(&'_ self) -> &'_ [u8] {
        &self.vector
    }

    fn read(&mut self, index: usize, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut data = &self.vector[index..];
        data.read(buf)
    }
}

impl Buffer for MmapBuffer {
    fn slice(&'_ self, start: usize, end: usize) -> Box<dyn Slice> {
        let vector = Vec::from_iter(self.mmap[start..end].iter().map(|t| *t));
        let _data: &[u8] = &vector;

        Box::new(MmapSlice { vector: vector })
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        Some(&self.mmap)
    }
}
