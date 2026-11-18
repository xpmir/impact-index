//! `rayon` when the `parallel` feature is on, a sequential stand-in with the
//! same method names otherwise (WASM builds), so call sites stay unchanged.

#[cfg(feature = "parallel")]
pub(crate) use rayon::{join, prelude::*};

#[cfg(not(feature = "parallel"))]
mod sequential {
    /// Sequential `par_iter`: the plain borrowing iterator
    #[allow(dead_code)] // only used by the text analysis (feature `analysis`)
    pub(crate) trait IntoParallelRefIterator<'a> {
        type Iter: Iterator;
        fn par_iter(&'a self) -> Self::Iter;
    }

    impl<'a, T: 'a> IntoParallelRefIterator<'a> for [T] {
        type Iter = std::slice::Iter<'a, T>;
        fn par_iter(&'a self) -> Self::Iter {
            self.iter()
        }
    }

    /// Sequential `into_par_iter`: the plain owning iterator
    pub(crate) trait IntoParallelIterator: IntoIterator + Sized {
        fn into_par_iter(self) -> Self::IntoIter {
            self.into_iter()
        }
    }

    impl<I: IntoIterator> IntoParallelIterator for I {}

    /// Sequential `join`: runs `a` then `b`
    pub(crate) fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA,
        B: FnOnce() -> RB,
    {
        (a(), b())
    }
}

#[cfg(not(feature = "parallel"))]
pub(crate) use sequential::*;
