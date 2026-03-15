//! BM25 scoring implementation.
//!
//! Implements the Okapi BM25 formula:
//! - TF component: `(k1 + 1) * tf / (k1 * (1 - b + b * dl / avgdl) + tf)`
//! - IDF component: `ln(1 + (N - df + 0.5) / (df + 0.5))`
//!
//! Upper bounds use `min_dl` for tightest bound (BM25's TF component is
//! monotonically decreasing in document length).

use std::sync::Arc;

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
    avg_dl: f32,
    min_dl: u32,
    num_docs: u64,
    doc_lengths: Option<Arc<Vec<u32>>>,
}

impl BM25Scoring {
    /// Create a new BM25 scoring model with default parameters (k1=1.2, b=0.75).
    pub fn new() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            avg_dl: 0.0,
            min_dl: 0,
            num_docs: 0,
            doc_lengths: None,
        }
    }

    /// Create with custom k1 and b parameters.
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            avg_dl: 0.0,
            min_dl: 0,
            num_docs: 0,
            doc_lengths: None,
        }
    }
}

impl ScoringModel for BM25Scoring {
    fn initialize(&mut self, doc_lengths: Arc<Vec<u32>>, num_docs: u64) {
        if !doc_lengths.is_empty() {
            let total: u64 = doc_lengths.iter().map(|&l| l as u64).sum();
            self.avg_dl = total as f32 / doc_lengths.len() as f32;
            self.min_dl = doc_lengths.iter().copied().min().unwrap_or(0);
        }
        self.num_docs = num_docs;
        self.doc_lengths = Some(doc_lengths);
    }

    fn term_scorer(&self, df: u64, _max_value: f32) -> Box<dyn ScoringFunction> {
        // IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        let n = self.num_docs as f64;
        let df_f64 = df as f64;
        let idf = ((n - df_f64 + 0.5) / (df_f64 + 0.5) + 1.0).ln() as f32;

        Box::new(BM25TermScorer {
            k1: self.k1,
            b: self.b,
            idf,
            avg_dl: self.avg_dl,
            min_dl: self.min_dl,
            doc_lengths: self
                .doc_lengths
                .as_ref()
                .expect("BM25Scoring not initialized")
                .clone(),
        })
    }
}

/// Per-term BM25 scorer (includes IDF).
struct BM25TermScorer {
    k1: f32,
    b: f32,
    idf: f32,
    avg_dl: f32,
    min_dl: u32,
    doc_lengths: Arc<Vec<u32>>,
}

impl ScoringFunction for BM25TermScorer {
    fn score(&self, tf: f32, docid: DocId) -> f32 {
        let dl = self.doc_lengths[docid as usize] as f32;
        let norm = self.k1 * (1.0 - self.b + self.b * dl / self.avg_dl);
        self.idf * tf / (norm + tf)
    }

    fn max_score(&self, max_tf: f32) -> f32 {
        // Use min_dl for the tightest upper bound
        let dl = self.min_dl as f32;
        let norm = self.k1 * (1.0 - self.b + self.b * dl / self.avg_dl);
        self.idf * max_tf / (norm + max_tf)
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
            (score - expected).abs() < 1e-5,
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
        for docid in 0..5u64 {
            for tf in [1.0, 2.0, 5.0, 10.0] {
                let s = scorer.score(tf, docid);
                assert!(
                    max >= s - 1e-6,
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
