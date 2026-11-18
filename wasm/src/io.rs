//! I/O through JavaScript callbacks: index files leave (or enter) the WASM
//! memory piece by piece instead of as whole `Uint8Array`s.

use std::io::{self, Write};
use std::rc::Rc;

use impact_index::storage::IndexOutput;
use js_sys::{Function, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Default size of the pieces handed to `write`
const DEFAULT_CHUNK_SIZE: usize = 1 << 20;

/// Message of a JS exception
pub(crate) fn js_message(e: &JsValue) -> String {
    if let Some(e) = e.dyn_ref::<js_sys::Error>() {
        String::from(e.message())
    } else if let Some(s) = e.as_string() {
        s
    } else {
        format!("{e:?}")
    }
}

pub(crate) fn js_io_error(name: &str, what: &str, e: &JsValue) -> io::Error {
    io::Error::other(format!("{name}: {what} failed: {}", js_message(e)))
}

/// Method `name` of `object`, if it is a function
pub(crate) fn method(object: &JsValue, name: &str) -> Result<Option<Function>, JsError> {
    let value = Reflect::get(object, &JsValue::from_str(name))
        .map_err(|_| JsError::new("expected an object"))?;
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }
    value
        .dyn_into::<Function>()
        .map(Some)
        .map_err(|_| JsError::new(&format!("{name} must be a function")))
}

struct Callbacks {
    this: JsValue,
    create: Option<Function>,
    write: Function,
    close: Option<Function>,
    chunk_size: usize,
}

/// An [`IndexOutput`] calling `{create(name), write(name, bytes),
/// close(name)}` (`create` and `close` optional), all synchronously.
/// `bytes` is a fresh `Uint8Array` (not a view on the WASM memory), so
/// it can be queued; the value returned by the callbacks is ignored, and an
/// exception aborts the build.
pub(crate) struct JsOutput(Rc<Callbacks>);

impl JsOutput {
    pub(crate) fn new(output: &JsValue) -> Result<Self, JsError> {
        let write = method(output, "write")?
            .ok_or_else(|| JsError::new("the output needs a write(name, bytes) method"))?;
        let chunk_size = match Reflect::get(output, &JsValue::from_str("chunkSize"))
            .ok()
            .and_then(|v| v.as_f64())
        {
            None => DEFAULT_CHUNK_SIZE,
            Some(n) if n >= 1. => n as usize,
            Some(_) => return Err(JsError::new("chunkSize must be positive")),
        };
        Ok(Self(Rc::new(Callbacks {
            this: output.clone(),
            create: method(output, "create")?,
            write,
            close: method(output, "close")?,
            chunk_size,
        })))
    }
}

pub(crate) struct JsWriter {
    callbacks: Rc<Callbacks>,
    name: JsValue,
    buffer: Vec<u8>,
}

impl JsWriter {
    fn send(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let bytes = Uint8Array::from(&self.buffer[..]);
        self.buffer.clear();
        let c = &self.callbacks;
        c.write
            .call2(&c.this, &self.name, &bytes)
            .map(|_| ())
            .map_err(|e| js_io_error(&self.name.as_string().unwrap_or_default(), "write", &e))
    }
}

impl Write for JsWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let chunk_size = self.callbacks.chunk_size;
        let mut rest = buf;
        while !rest.is_empty() {
            let n = (chunk_size - self.buffer.len()).min(rest.len());
            self.buffer.extend_from_slice(&rest[..n]);
            rest = &rest[n..];
            if self.buffer.len() >= chunk_size {
                self.send()?;
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.send()
    }
}

impl IndexOutput for JsOutput {
    type Writer = JsWriter;

    fn create(&self, name: &str) -> io::Result<JsWriter> {
        let js_name = JsValue::from_str(name);
        if let Some(create) = &self.0.create {
            create
                .call1(&self.0.this, &js_name)
                .map_err(|e| js_io_error(name, "create", &e))?;
        }
        Ok(JsWriter {
            callbacks: self.0.clone(),
            name: js_name,
            buffer: Vec::new(),
        })
    }

    fn close(&self, name: &str, mut writer: JsWriter) -> io::Result<()> {
        writer.send()?;
        if let Some(close) = &self.0.close {
            close
                .call1(&self.0.this, &writer.name)
                .map_err(|e| js_io_error(name, "close", &e))?;
        }
        Ok(())
    }

    fn describe(&self, name: &str) -> String {
        name.to_string()
    }
}

/// Appends to the caller's scratch stream through `write(bytes)` (a copy)
pub(crate) struct JsSpillWriter(pub(crate) Function);

impl Write for JsSpillWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .call1(&JsValue::NULL, &Uint8Array::from(buf))
            .map_err(|e| js_io_error("spill", "write", &e))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Reads the scratch stream back through `read(offset, length)`
pub(crate) struct JsSpillReader(pub(crate) Function);

// wasm32-unknown-unknown without atomics is single-threaded: the JS
// function is never shared between threads
unsafe impl Sync for JsSpillReader {}

impl impact_index::memory_indexer::SpillReader for JsSpillReader {
    fn read(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let bytes = self
            .0
            .call2(
                &JsValue::NULL,
                &JsValue::from_f64(offset as f64),
                &JsValue::from_f64(length as f64),
            )
            .map_err(|e| js_io_error("spill", "read", &e))?
            .dyn_into::<Uint8Array>()
            .map_err(|_| io::Error::other("spill: read must return a Uint8Array"))?;
        Ok(bytes.to_vec())
    }
}

/// Ranged reads of index files through `{size(name), read(name, offset,
/// length)}`, both synchronous
pub(crate) struct JsRangeReader {
    this: JsValue,
    size: Function,
    read: Function,
}

// wasm32-unknown-unknown without atomics is single-threaded: the JS
// functions are never shared between threads
unsafe impl Send for JsRangeReader {}
unsafe impl Sync for JsRangeReader {}

impl JsRangeReader {
    pub(crate) fn new(reader: &JsValue) -> Result<Self, JsError> {
        let size = method(reader, "size")?
            .ok_or_else(|| JsError::new("the reader needs a size(name) method"))?;
        let read = method(reader, "read")?
            .ok_or_else(|| JsError::new("the reader needs a read(name, offset, length) method"))?;
        Ok(Self {
            this: reader.clone(),
            size,
            read,
        })
    }
}

impl impact_index::storage::RangeReader for JsRangeReader {
    fn size(&self, name: &str) -> io::Result<Option<u64>> {
        let size = self
            .size
            .call1(&self.this, &JsValue::from_str(name))
            .map_err(|e| js_io_error(name, "size", &e))?;
        if size.is_undefined() || size.is_null() {
            return Ok(None);
        }
        size.as_f64()
            .filter(|s| *s >= 0.)
            .map(|s| Some(s as u64))
            .ok_or_else(|| io::Error::other(format!("{name}: size must be a number")))
    }

    fn read(&self, name: &str, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let bytes = self
            .read
            .call3(
                &self.this,
                &JsValue::from_str(name),
                &JsValue::from_f64(offset as f64),
                &JsValue::from_f64(length as f64),
            )
            .map_err(|e| js_io_error(name, "read", &e))?
            .dyn_into::<Uint8Array>()
            .map_err(|_| io::Error::other(format!("{name}: read must return a Uint8Array")))?;
        Ok(bytes.to_vec())
    }
}
