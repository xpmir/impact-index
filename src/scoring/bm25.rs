//! BM25 scoring implementation.
//!
//! Implements the Okapi BM25 formula:
//! - TF component: `(k1 + 1) * tf / (k1 * (1 - b + b * dl / avgdl) + tf)`
//! - IDF component: `ln(1 + (N - df + 0.5) / (df + 0.5))`
//!
//! Upper bounds use `min_dl` for tightest bound (BM25's TF component is
//! monotonically decreasing in document length).

use std::sync::Arc;

use half::f16;

use crate::base::DocId;

use super::{ScoringFunction, ScoringModel};

/// BM25 scoring model.
///
/// Create with desired parameters, then pass to [`ScoredIndex::new`](super::ScoredIndex::new).
pub struct BM25Scoring {
    /// Term frequency saturation parameter (default: 1.2).
    pub k1: f32,
    /// Length normalization parameter (default: 0.75).
    pub b: f32,

    // Computed on initialize():
    min_dl_norm: f32,
    /// `k1 * (1 - b)`: the length-independent half of the BM25 norm.
    /// Cached so `max_score_with_dl` can recompute `norm(dl)` for an
    /// arbitrary `dl` (a block or term's minimum document length) without
    /// re-deriving it from `k1`/`b`/`avgdl` on every query.
    k1_one_minus_b: f32,
    /// `k1 * b / avgdl`: the length-dependent coefficient of the BM25 norm.
    k1_b_over_avgdl: f32,
    num_docs: u64,
    /// Pre-computed BM25 norms per document as f16: k1 * (1 - b + b * dl / avgdl).
    /// Half the memory of f32/u32 (17.5MB vs 35MB for 8.8M docs), with
    /// hardware f16→f32 conversion (~1 cycle on ARM NEON / x86 F16C).
    doc_norms: Option<Arc<Vec<f16>>>,
}

impl BM25Scoring {
    /// Create a new BM25 scoring model with default parameters (k1=1.2, b=0.75).
    pub fn new() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_dl_norm: 0.0,
            k1_one_minus_b: 0.0,
            k1_b_over_avgdl: 0.0,
            num_docs: 0,
            doc_norms: None,
        }
    }

    /// Create with custom k1 and b parameters.
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            min_dl_norm: 0.0,
            k1_one_minus_b: 0.0,
            k1_b_over_avgdl: 0.0,
            num_docs: 0,
            doc_norms: None,
        }
    }
}

impl ScoringModel for BM25Scoring {
    fn initialize(&mut self, doc_lengths: Arc<Vec<u32>>, num_docs: u64) {
        if !doc_lengths.is_empty() {
            let total: u64 = doc_lengths.iter().map(|&l| l as u64).sum();
            let avg_dl = total as f32 / doc_lengths.len() as f32;
            let min_dl = doc_lengths.iter().copied().min().unwrap_or(0);

            let k1_one_minus_b = self.k1 * (1.0 - self.b);
            let k1_b_over_avgdl = self.k1 * self.b / avg_dl;
            self.k1_one_minus_b = k1_one_minus_b;
            self.k1_b_over_avgdl = k1_b_over_avgdl;
            self.min_dl_norm = k1_one_minus_b + k1_b_over_avgdl * min_dl as f32;

            // Pre-compute BM25 norms as f16 — half the cache footprint of f32/u32
            let norms: Vec<f16> = doc_lengths
                .iter()
                .map(|&dl| f16::from_f32(k1_one_minus_b + k1_b_over_avgdl * dl as f32))
                .collect();
            self.doc_norms = Some(Arc::new(norms));
        }
        self.num_docs = num_docs;
    }

    fn term_scorer(&self, df: u64, _max_value: f32) -> Box<dyn ScoringFunction> {
        Box::new(self.build_term_scorer(df))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl BM25Scoring {
    /// Builds the concrete per-term scorer (shared by `term_scorer` and
    /// `term_scorer_typed`).
    fn build_term_scorer(&self, df: u64) -> BM25TermScorer {
        // IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        let n = self.num_docs as f64;
        let df_f64 = df as f64;
        let idf = ((n - df_f64 + 0.5) / (df_f64 + 0.5) + 1.0).ln() as f32;

        BM25TermScorer {
            idf,
            min_dl_norm: self.min_dl_norm,
            k1_one_minus_b: self.k1_one_minus_b,
            k1_b_over_avgdl: self.k1_b_over_avgdl,
            doc_norms: self
                .doc_norms
                .as_ref()
                .expect("BM25Scoring not initialized")
                .clone(),
        }
    }

    /// Concrete (unboxed) per-term scorer, for the monomorphized search
    /// fast path (P3) — avoids the `Box<dyn ScoringFunction>` vtable.
    pub(crate) fn term_scorer_typed(&self, df: u64) -> BM25TermScorer {
        self.build_term_scorer(df)
    }
}

/// Extra multiplicative safety margin applied (beyond the f16-rounding
/// guard below) to the norm used by [`BM25TermScorer::max_score_with_dl`].
/// Guards against any residual floating-point order-of-operations mismatch
/// between the bound computation and the actual per-posting score
/// computation, on top of the explicit f16-rounding guard. See
/// `max_score_with_dl` for the full safety argument.
const BOUND_SAFETY_MARGIN: f32 = 1e-3;

/// Per-term BM25 scorer (includes IDF).
pub(crate) struct BM25TermScorer {
    idf: f32,
    /// Pre-computed normalization for the collection-wide min_dl (fallback
    /// for `max_score` and for `max_score_with_dl(_, 0)`).
    min_dl_norm: f32,
    /// `k1 * (1 - b)` (see [`BM25Scoring::k1_one_minus_b`]).
    k1_one_minus_b: f32,
    /// `k1 * b / avgdl` (see [`BM25Scoring::k1_b_over_avgdl`]).
    k1_b_over_avgdl: f32,
    /// Pre-computed BM25 norms per document (f16, shared across all term scorers)
    doc_norms: Arc<Vec<f16>>,
}

impl ScoringFunction for BM25TermScorer {
    #[inline]
    fn score(&self, tf: f32, docid: DocId) -> f32 {
        let norm = unsafe { self.doc_norms.get_unchecked(docid as usize) }.to_f32();
        self.idf * tf / (norm + tf)
    }

    #[inline]
    fn max_score(&self, max_tf: f32) -> f32 {
        self.idf * max_tf / (self.min_dl_norm + max_tf)
    }

    /// Tightened bound (P1a): same formula as [`Self::max_score`], but
    /// using `norm(min_dl)` for a caller-supplied `min_dl` (a block or
    /// term's own minimum document length) instead of the collection-wide
    /// minimum -- BM25's TF-normalization is monotone decreasing in `dl`,
    /// so a larger (tighter) `min_dl` always yields a smaller, still-safe
    /// upper bound. `min_dl == 0` is the iterator-side "not available"
    /// sentinel (see [`crate::index::BlockTermImpactIterator::min_dl`]) and
    /// falls back to [`Self::max_score`].
    ///
    /// # Safety margin (why this isn't just `idf * max_tf / (norm + max_tf)`)
    ///
    /// Actual per-doc scores (`score`, `score_chunk`) are computed against
    /// `doc_norms`, which stores each document's *exact* norm rounded to
    /// `f16`. `f16` round-to-nearest can round a norm **down**, which would
    /// make a real score computed against that (smaller) rounded norm
    /// **larger** than a bound computed against the exact `f32` norm --
    /// silently breaking the "bound dominates every real score" invariant
    /// pruning depends on. Since every document of length exactly `min_dl`
    /// gets the identical, deterministic `f16` rounding of `norm(min_dl)`
    /// (precomputed once in `initialize`), we can compute that exact same
    /// rounding here and pick whichever of the exact/rounded norm is
    /// *smaller* (a smaller norm can only ever raise the bound, never lower
    /// it below a real score). On top of that, an extra small multiplicative
    /// margin ([`BOUND_SAFETY_MARGIN`]) guards against any remaining
    /// floating-point order-of-operations mismatch. The
    /// `debug_assert!`s in `compress::CompressedScoringCursor::fill_chunk`
    /// and `scoring::ScoringBlockIterator::current` check this invariant
    /// against real data on every debug-mode query.
    #[inline]
    fn max_score_with_dl(&self, max_tf: f32, min_dl: u32) -> f32 {
        if min_dl == 0 {
            return self.max_score(max_tf);
        }

        let raw_norm = self.k1_one_minus_b + self.k1_b_over_avgdl * min_dl as f32;
        let f16_norm = f16::from_f32(raw_norm).to_f32();
        let safe_norm = raw_norm.min(f16_norm) * (1.0 - BOUND_SAFETY_MARGIN);

        self.idf * max_tf / (safe_norm + max_tf)
    }

    /// Batched BM25 scoring (P1b). Two passes over the chunk rather than
    /// one fused loop:
    ///
    /// 1. Gather norms into `out` (reused as scratch — no extra buffer).
    ///    These are independent random loads into `doc_norms`; issuing them
    ///    back-to-back lets the CPU overlap the cache misses that were
    ///    previously serialized one-per-posting behind a vtable call.
    /// 2. A branch-free loop computing `idf * tf / (norm + tf)` — the exact
    ///    same scalar expression (and f16 norm read) as `score()`, so
    ///    results are bit-identical. This shape (independent f16->f32
    ///    converts, multiply, divide over a slice) auto-vectorizes on both
    ///    NEON and AVX2.
    #[inline]
    fn score_chunk(
        &self,
        block_min_doc_id: DocId,
        docid_offsets: &[u32],
        tfs: &[f32],
        out: &mut [f32],
    ) {
        let n = tfs.len();
        debug_assert_eq!(docid_offsets.len(), n);
        debug_assert_eq!(out.len(), n);

        let doc_norms = &self.doc_norms;
        for i in 0..n {
            let docid = block_min_doc_id + docid_offsets[i] as DocId;
            out[i] = unsafe { doc_norms.get_unchecked(docid as usize) }.to_f32();
        }

        let idf = self.idf;
        for i in 0..n {
            out[i] = idf * tfs[i] / (out[i] + tfs[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_idf() {
        let mut scoring = BM25Scoring::new();
        let doc_lengths = Arc::new(vec![10u32; 100]);
        scoring.initialize(doc_lengths, 100);

        // df=10, N=100: idf = ln(1 + (100 - 10 + 0.5) / (10 + 0.5))
        let scorer = scoring.term_scorer(10, 5.0);
        let expected_idf = ((100.0 - 10.0 + 0.5) / (10.0 + 0.5) + 1.0f64).ln() as f32;
        // Score with tf=1, dl=avgdl => tf / (k1 + tf) = 1 / (1.2 + 1) = 1/2.2
        let score = scorer.score(1.0, 0);
        let expected = expected_idf * 1.0 / (1.2 * 1.0 + 1.0);
        assert!(
            (score - expected).abs() < 1e-3,
            "score={}, expected={}",
            score,
            expected
        );
    }

    #[test]
    fn test_bm25_max_score_geq_score() {
        let mut scoring = BM25Scoring::new();
        let doc_lengths = Arc::new(vec![5, 10, 15, 20, 100]);
        scoring.initialize(doc_lengths, 5);

        let scorer = scoring.term_scorer(3, 10.0);
        let max = scorer.max_score(10.0);

        // max_score should be >= score for any valid docid and tf <= max_tf
        // Note: f16 quantization means scored values may slightly exceed max_score
        for docid in 0..5u64 {
            for tf in [1.0, 2.0, 5.0, 10.0] {
                let s = scorer.score(tf, docid);
                assert!(
                    max >= s - 1e-2,
                    "max_score ({}) < score ({}) for docid={}, tf={}",
                    max,
                    s,
                    docid,
                    tf
                );
            }
        }
    }
}
