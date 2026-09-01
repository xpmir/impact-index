//! Terrier-matchop-style structured query language, evaluated as "virtual
//! posting lists" over the existing WAND/MaxScore search loops.
//!
//! Every [`QueryNode`] operator ([`QueryNode::Combine`], [`QueryNode::Syn`],
//! [`QueryNode::Band`], [`QueryNode::Phrase`], [`QueryNode::Window`])
//! evaluates ([`evaluate`]) to a composite [`crate::index::BlockTermImpactIterator`]
//! (`crate::search::ops`), so [`crate::search::wand::search_wand_cursors`]
//! and [`crate::search::maxscore::search_maxscore_cursors`] run over it
//! exactly as they would over a plain term cursor -- see
//! [`search_wand_query`]/[`search_maxscore_query`], the two public search
//! entry points.
//!
//! See `positions-plan.md` (Phase 3) for the design rationale, and
//! `src/search/ops.rs` for the pruning-safety argument behind each
//! composite's bounds.

use std::collections::HashMap;

use crate::base::TermIndex;
use crate::index::{BlockTermImpactIterator, SparseIndex};
use crate::scoring::{wrap_scored_cursor, ScoredIndex};
use crate::search::maxscore::MaxScoreOptions;
use crate::search::ops::{AndCursor, PhraseCursor, SumCursor, WindowCursor};
use crate::search::ScoredDocument;

// ---------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------

/// A structured query, Terrier-matchop style.
///
/// Every variant evaluates to one virtual posting list ([`evaluate`]):
/// scalar terms score directly, while [`Combine`](QueryNode::Combine)/
/// [`Syn`](QueryNode::Syn)/[`Band`](QueryNode::Band)/
/// [`Phrase`](QueryNode::Phrase)/[`Window`](QueryNode::Window) are
/// composites over their children's virtual posting lists (see
/// `crate::search::ops`). Construct by hand, or parse Terrier matchop text
/// with [`parse_matchop`].
#[derive(Debug, Clone, PartialEq)]
pub enum QueryNode {
    /// A single term with a query weight (multiplies its score).
    Term { term: TermIndex, weight: f32 },
    /// Weighted sum of children's scores (matchop `#combine`). The root
    /// combinator: [`evaluate`] flattens a root `Combine` directly into the
    /// top-level `(weight, cursor)` list the search loops consume.
    Combine { children: Vec<(f32, QueryNode)> },
    /// Synonym/OR (matchop `#syn`): children's term frequencies are summed
    /// and the merged posting list is scored as ONE virtual term.
    /// v1 restriction: children must be plain terms.
    Syn { terms: Vec<TermIndex> },
    /// Boolean AND (matchop `#band`): matches docs containing every child;
    /// score = sum of children's scores (Lucene MUST semantics).
    Band { children: Vec<QueryNode> },
    /// Exact phrase (matchop `#1`): adjacent positions. Requires an index
    /// built with positions ([`QueryError::PositionsNotAvailable`]
    /// otherwise).
    Phrase { terms: Vec<TermIndex> },
    /// Unordered window of width `width` tokens (matchop `#uwN`). Requires
    /// positions, same as [`Phrase`](QueryNode::Phrase).
    Window { terms: Vec<TermIndex>, width: u32 },
}

/// Errors from parsing or evaluating a [`QueryNode`].
#[derive(Debug, Clone, PartialEq)]
pub enum QueryError {
    /// A [`QueryNode::Phrase`]/[`QueryNode::Window`] was evaluated against
    /// an index that was not built with positions.
    PositionsNotAvailable,
    /// The query (or a subtree pruned down by `evaluate`/the parser to
    /// nothing) matches no documents by construction -- e.g. every term
    /// was unresolved. Not a hard error at the search entry points
    /// ([`search_wand_query`]/[`search_maxscore_query`]): they turn this
    /// into an empty result set.
    EmptyQuery,
    /// Malformed matchop text.
    Parse(String),
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::PositionsNotAvailable => write!(
                f,
                "index was built without positions; rebuild with positions=true"
            ),
            QueryError::EmptyQuery => write!(f, "query is empty (matches no documents)"),
            QueryError::Parse(msg) => write!(f, "query parse error: {}", msg),
        }
    }
}

impl std::error::Error for QueryError {}

impl QueryNode {
    /// Structural validation: every operator has enough children/terms to
    /// be meaningful. Does not check term-index bounds against a concrete
    /// index, or positional availability -- both are [`evaluate`]'s job,
    /// since they need an index to answer.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            QueryNode::Term { .. } => Ok(()),
            QueryNode::Combine { children } => {
                if children.is_empty() {
                    return Err(QueryError::EmptyQuery);
                }
                children.iter().try_for_each(|(_, c)| c.validate())
            }
            QueryNode::Syn { terms } => {
                // >= 2 terms is the normal case; exactly 1 is allowed (it
                // just degenerates to a plain Term), 0 is not.
                if terms.is_empty() {
                    return Err(QueryError::EmptyQuery);
                }
                Ok(())
            }
            QueryNode::Band { children } => {
                if children.is_empty() {
                    return Err(QueryError::EmptyQuery);
                }
                children.iter().try_for_each(QueryNode::validate)
            }
            QueryNode::Phrase { terms } => {
                if terms.is_empty() {
                    return Err(QueryError::EmptyQuery);
                }
                Ok(())
            }
            QueryNode::Window { terms, width } => {
                if terms.is_empty() {
                    return Err(QueryError::EmptyQuery);
                }
                if *width < 2 {
                    return Err(QueryError::Parse(format!(
                        "window width must be >= 2, got {}",
                        width
                    )));
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------

/// Turns a query tree into the top-level list of `(weight, cursor)` pairs
/// the search loops consume ([`crate::search::wand::search_wand_cursors`]/
/// [`crate::search::maxscore::search_maxscore_cursors`]).
///
/// The root [`QueryNode::Combine`] is flattened directly into the returned
/// list (its per-child weight, further multiplied by that child's own
/// [`QueryNode::Term::weight`] when the child is a plain term); any other
/// root becomes a single entry with weight `1.0`. `EmptyQuery` (see
/// [`QueryError`]) can come from either [`QueryNode::validate`] or from
/// evaluating a composite down to zero live children -- both
/// [`search_wand_query`]/[`search_maxscore_query`] treat it as "no
/// results", not a hard error.
pub fn evaluate<'a>(
    index: &'a dyn SparseIndex,
    node: &QueryNode,
) -> Result<Vec<(f32, Box<dyn BlockTermImpactIterator + 'a>)>, QueryError> {
    node.validate()?;

    match node {
        QueryNode::Combine { children } => {
            let mut entries = Vec::with_capacity(children.len());
            for (weight, child) in children {
                let (child_weight, cursor) = eval_weighted(index, child)?;
                entries.push((weight * child_weight, cursor));
            }
            if entries.is_empty() {
                return Err(QueryError::EmptyQuery);
            }
            Ok(entries)
        }
        _ => Ok(vec![(1.0, eval_node(index, node)?)]),
    }
}

/// Evaluates `node`, additionally reporting its own weight if it is a
/// [`QueryNode::Term`] (so a `Combine`/`Band` parent can fold `weight *
/// term.weight` without an extra wrapper cursor). Any other node
/// contributes weight `1.0` here -- its own internal weighting (if any) is
/// already baked into its cursor's values by [`eval_node`].
fn eval_weighted<'a>(
    index: &'a dyn SparseIndex,
    node: &QueryNode,
) -> Result<(f32, Box<dyn BlockTermImpactIterator + 'a>), QueryError> {
    match node {
        QueryNode::Term { term, weight } => Ok((*weight, index.block_iterator(*term))),
        _ => Ok((1.0, eval_node(index, node)?)),
    }
}

/// Wraps `cursor` in a single-child, weighted [`SumCursor`] to scale its
/// value by `weight` -- used to apply [`QueryNode::Term::weight`] in
/// contexts (e.g. [`QueryNode::Band`] children) that don't otherwise carry
/// a per-child weight. A no-op (returns `cursor` unchanged) when `weight ==
/// 1.0`, the overwhelmingly common case.
fn apply_weight<'a>(
    weight: f32,
    cursor: Box<dyn BlockTermImpactIterator + 'a>,
) -> Box<dyn BlockTermImpactIterator + 'a> {
    if weight == 1.0 {
        cursor
    } else {
        Box::new(SumCursor::new(vec![(weight, cursor)]))
    }
}

/// Evaluates any [`QueryNode`] to a single cursor.
///
/// Dispatches once on whether `index` is a [`ScoredIndex`]
/// (`index.as_any().downcast_ref`): a plain [`QueryNode::Term`] just calls
/// `index.block_iterator` either way (already scored when `index` is a
/// `ScoredIndex`, since virtual dispatch resolves to
/// `ScoredIndex::block_iterator`). [`QueryNode::Syn`]/
/// [`QueryNode::Phrase`]/[`QueryNode::Window`] are different: their
/// virtual term frequency has to be computed in *raw* tf space (from
/// `scored.inner_index()`'s cursors) before being scored, so those three
/// branches special-case the `ScoredIndex` case to build the composite
/// over raw children and then [`wrap_scored_cursor`] it with
/// [`crate::scoring::ScoringModel::compound_scorer`]. Over a raw
/// (unscored) index, the same tree evaluates directly in raw-value space
/// (no wrapping) -- useful for quantized/learned-impact experimentation.
fn eval_node<'a>(
    index: &'a dyn SparseIndex,
    node: &QueryNode,
) -> Result<Box<dyn BlockTermImpactIterator + 'a>, QueryError> {
    let scored = index.as_any().downcast_ref::<ScoredIndex>();

    match node {
        QueryNode::Term { term, .. } => Ok(index.block_iterator(*term)),

        QueryNode::Combine { children } => {
            let mut parts = Vec::with_capacity(children.len());
            for (weight, child) in children {
                let (child_weight, cursor) = eval_weighted(index, child)?;
                parts.push((weight * child_weight, cursor));
            }
            if parts.is_empty() {
                return Err(QueryError::EmptyQuery);
            }
            Ok(Box::new(SumCursor::new(parts)))
        }

        QueryNode::Band { children } => {
            let mut parts = Vec::with_capacity(children.len());
            for child in children {
                let (weight, cursor) = eval_weighted(index, child)?;
                parts.push(apply_weight(weight, cursor));
            }
            if parts.is_empty() {
                return Err(QueryError::EmptyQuery);
            }
            Ok(Box::new(AndCursor::new(parts)))
        }

        QueryNode::Syn { terms } => {
            if terms.is_empty() {
                return Err(QueryError::EmptyQuery);
            }
            if let Some(scored) = scored {
                let inner = scored.inner_index();
                let (dfs, children) = raw_term_children(inner, terms);
                let merged: Box<dyn BlockTermImpactIterator + 'a> =
                    Box::new(SumCursor::new(children));
                let max_value = merged.max_value();
                // For synonyms Lucene blends idfs rather than summing them;
                // summing children dfs directly would overcount docs that
                // contain several synonyms. Passing per-child dfs and
                // letting `compound_scorer` sum idfs follows the phrase
                // convention instead -- documented v1 choice, see
                // `positions-plan.md` Phase 3.
                let scorer = scored.model().compound_scorer(&dfs, max_value);
                Ok(wrap_scored_cursor(merged, scorer))
            } else {
                let children = terms
                    .iter()
                    .map(|&t| (1.0f32, index.block_iterator(t)))
                    .collect();
                Ok(Box::new(SumCursor::new(children)))
            }
        }

        QueryNode::Phrase { terms } => {
            eval_positional(index, scored, terms, PositionalKind::Phrase)
        }
        QueryNode::Window { terms, width } => {
            eval_positional(index, scored, terms, PositionalKind::Window(*width))
        }
    }
}

/// Which positional operator [`eval_positional`] should build.
enum PositionalKind {
    Phrase,
    Window(u32),
}

/// Shared `Phrase`/`Window` evaluation: raw positional children (from the
/// scored index's inner index, or `index` itself when unscored), wrapped
/// with a compound scorer only in the scored case.
fn eval_positional<'a>(
    index: &'a dyn SparseIndex,
    scored: Option<&'a ScoredIndex>,
    terms: &[TermIndex],
    kind: PositionalKind,
) -> Result<Box<dyn BlockTermImpactIterator + 'a>, QueryError> {
    if !SparseIndex::has_positions(index) {
        return Err(QueryError::PositionsNotAvailable);
    }
    if terms.is_empty() {
        return Err(QueryError::EmptyQuery);
    }

    let source: &'a dyn SparseIndex = match scored {
        Some(s) => s.inner_index(),
        None => index,
    };
    let (dfs, children) = raw_term_children(source, terms);
    let children: Vec<Box<dyn BlockTermImpactIterator + 'a>> =
        children.into_iter().map(|(_, c)| c).collect();

    let positional: Box<dyn BlockTermImpactIterator + 'a> = match kind {
        PositionalKind::Phrase => Box::new(PhraseCursor::new(children)),
        PositionalKind::Window(width) => Box::new(WindowCursor::new(children, width)),
    };

    match scored {
        Some(scored) => {
            let max_value = positional.max_value();
            let scorer = scored.model().compound_scorer(&dfs, max_value);
            Ok(wrap_scored_cursor(positional, scorer))
        }
        None => Ok(positional),
    }
}

/// Builds raw (unscored) per-term cursors from `source` for every term in
/// `terms`, plus each one's document frequency (`length()`) -- the shared
/// shape [`QueryNode::Syn`]/[`QueryNode::Phrase`]/[`QueryNode::Window`] all
/// need before merging/aligning and (in the scored case) computing a
/// compound scorer.
fn raw_term_children<'a>(
    source: &'a dyn SparseIndex,
    terms: &[TermIndex],
) -> (Vec<u64>, Vec<(f32, Box<dyn BlockTermImpactIterator + 'a>)>) {
    let mut dfs = Vec::with_capacity(terms.len());
    let mut children = Vec::with_capacity(terms.len());
    for &t in terms {
        let c = source.block_iterator(t);
        dfs.push(c.length() as u64);
        children.push((1.0f32, c));
    }
    (dfs, children)
}

// ---------------------------------------------------------------------
// Search entry points
// ---------------------------------------------------------------------

/// Evaluates `query` against `index` and searches with WAND, returning the
/// top-k documents by score.
///
/// An empty query ([`QueryError::EmptyQuery`] -- e.g. every term was
/// unresolved, or a required clause can never match) returns `Ok(vec![])`,
/// not an error: real errors are reserved for
/// [`QueryError::PositionsNotAvailable`] (a `Phrase`/`Window` node against
/// an index without positions) and [`QueryError::Parse`] (propagated from
/// a hand-built invalid tree; [`parse_matchop`] itself never produces one).
/// Document ids in the result are original ids, even if `index` was
/// reordered ([`crate::transforms::reorder::ReorderTransform`]).
pub fn search_wand_query(
    index: &dyn SparseIndex,
    query: &QueryNode,
    top_k: usize,
) -> Result<Vec<ScoredDocument>, QueryError> {
    let cursors = match evaluate(index, query) {
        Ok(cursors) => cursors,
        Err(QueryError::EmptyQuery) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut results = crate::search::wand::search_wand_cursors(cursors, top_k);
    crate::search::remap_to_original_ids(index, &mut results);
    Ok(results)
}

/// Evaluates `query` against `index` and searches with MaxScore, returning
/// the top-k documents by score. Same `EmptyQuery`/error/doc-id-remapping
/// contract as [`search_wand_query`].
pub fn search_maxscore_query(
    index: &dyn SparseIndex,
    query: &QueryNode,
    top_k: usize,
    options: MaxScoreOptions,
) -> Result<Vec<ScoredDocument>, QueryError> {
    let cursors = match evaluate(index, query) {
        Ok(cursors) => cursors,
        Err(QueryError::EmptyQuery) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut results = crate::search::maxscore::search_maxscore_cursors(cursors, top_k, options);
    crate::search::remap_to_original_ids(index, &mut results);
    Ok(results)
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

/// Parses a Terrier-matchop-style query string into a [`QueryNode`] tree.
///
/// Grammar: a whitespace-separated sequence of nodes at top level becomes a
/// [`QueryNode::Combine`] with weight `1.0` each (collapsed to the single
/// node itself when the sequence has exactly one, unweighted, element);
/// `#combine(...)` with optional `:IDX=WEIGHT` suffixes
/// (`#combine:0=2:1=1(a b)` -- Terrier syntax: index=weight pairs,
/// `IDX` a 0-based position in the following paren-list, separated by
/// `:`); `#syn(t1 t2 ...)`; `#band(n1 n2 ...)` (children can be any node,
/// not just terms); `#1(t1 t2 ...)` (phrase); `#uwN(t1 t2 ...)` (window,
/// `N` = digits, the window width). Parens nest; anything not starting
/// with `#` is a term token resolved via `resolve`.
///
/// Unresolved-term handling mirrors `BOWIndexBuilder::analyze_query`'s
/// skip-unknown behavior, generalized per-operator:
/// - Under `Combine` (including the top level) or `Syn`: the single
///   unresolved term is dropped, siblings are kept.
/// - Under `Band`/`Phrase`/`Window`: positional/conjunctive semantics mean
///   one missing term makes the WHOLE node unable to match anything, so the
///   whole node is dropped instead -- and, recursively, anything that
///   thereby becomes empty (e.g. a `Combine` all of whose children were
///   dropped this way). An entirely empty query parses successfully (as an
///   empty [`QueryNode::Combine`]); [`evaluate`] turns that into
///   [`QueryError::EmptyQuery`], and the search entry points turn THAT into
///   an empty result set.
///
/// The parser itself is a small hand-rolled recursive descent (no new
/// dependencies): malformed input (unbalanced parens, an unknown operator,
/// a malformed `:IDX=WEIGHT` suffix, an operator nested where only plain
/// terms are allowed) is a [`QueryError::Parse`].
pub fn parse_matchop(
    text: &str,
    resolve: &dyn Fn(&str) -> Option<TermIndex>,
) -> Result<QueryNode, QueryError> {
    let tokens = tokenize(text);
    let mut parser = Parser {
        tokens,
        pos: 0,
        resolve,
    };
    let node = parser.parse_top_level()?;
    if parser.pos != parser.tokens.len() {
        return Err(QueryError::Parse(format!(
            "unexpected trailing token '{}'",
            parser.tokens[parser.pos]
        )));
    }
    Ok(node)
}

/// Splits `text` into `(`, `)`, and whitespace-separated word tokens (a
/// word run stops at whitespace or a paren, so `#combine(a` tokenizes as
/// `["#combine", "(", "a"]` with no space required before `(`).
fn tokenize(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some(&(i, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c == '(' || c == ')' {
            tokens.push(&text[i..i + c.len_utf8()]);
            chars.next();
            continue;
        }
        let start = i;
        let mut end = i + c.len_utf8();
        chars.next();
        while let Some(&(j, c2)) = chars.peek() {
            if c2.is_whitespace() || c2 == '(' || c2 == ')' {
                break;
            }
            end = j + c2.len_utf8();
            chars.next();
        }
        tokens.push(&text[start..end]);
    }
    tokens
}

struct Parser<'a, 'r> {
    tokens: Vec<&'a str>,
    pos: usize,
    resolve: &'r dyn Fn(&str) -> Option<TermIndex>,
}

impl<'a, 'r> Parser<'a, 'r> {
    fn peek(&self) -> Option<&'a str> {
        self.tokens.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<&'a str> {
        let t = self.peek();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, tok: &str) -> Result<(), QueryError> {
        match self.next() {
            Some(t) if t == tok => Ok(()),
            Some(t) => Err(QueryError::Parse(format!(
                "expected '{}', found '{}'",
                tok, t
            ))),
            None => Err(QueryError::Parse(format!(
                "expected '{}', found end of input",
                tok
            ))),
        }
    }

    /// Top-level sequence: every node up to end-of-input, each with weight
    /// `1.0`, collapsed to the bare node when exactly one survives.
    fn parse_top_level(&mut self) -> Result<QueryNode, QueryError> {
        let mut children = Vec::new();
        while self.peek().is_some() && self.peek() != Some(")") {
            if let Some(node) = self.parse_node()? {
                children.push((1.0f32, node));
            }
        }
        if children.len() == 1 && children[0].0 == 1.0 {
            Ok(children.into_iter().next().unwrap().1)
        } else {
            Ok(QueryNode::Combine { children })
        }
    }

    /// Parses one node: an operator (`#...`) or a bare term. `Ok(None)`
    /// means "dropped" (an unresolved term, or an operator that pruned
    /// itself away -- see [`parse_matchop`]'s doc comment).
    fn parse_node(&mut self) -> Result<Option<QueryNode>, QueryError> {
        let tok = self
            .peek()
            .ok_or_else(|| QueryError::Parse("unexpected end of input".to_string()))?;
        if let Some(stripped) = tok.strip_prefix('#') {
            self.parse_op(stripped)
        } else {
            self.next();
            Ok((self.resolve)(tok).map(|term| QueryNode::Term { term, weight: 1.0 }))
        }
    }

    /// Parses one operator node, `name` being the `#`-prefixed token with
    /// the leading `#` already stripped (so `#combine:0=2` -> `name ==
    /// "combine:0=2"`).
    fn parse_op(&mut self, name: &str) -> Result<Option<QueryNode>, QueryError> {
        let full_tok = self.next().expect("caller peeked this token");
        let mut parts = name.split(':');
        let op = parts.next().unwrap_or("");

        match op {
            "combine" => {
                let mut weights: HashMap<usize, f32> = HashMap::new();
                for part in parts {
                    let (idx_s, w_s) = part.split_once('=').ok_or_else(|| {
                        QueryError::Parse(format!(
                            "malformed combine weight spec '{}' in '{}'",
                            part, full_tok
                        ))
                    })?;
                    let idx: usize = idx_s.parse().map_err(|_| {
                        QueryError::Parse(format!(
                            "bad combine index '{}' in '{}'",
                            idx_s, full_tok
                        ))
                    })?;
                    let w: f32 = w_s.parse().map_err(|_| {
                        QueryError::Parse(format!("bad combine weight '{}' in '{}'", w_s, full_tok))
                    })?;
                    weights.insert(idx, w);
                }

                self.expect("(")?;
                let mut children = Vec::new();
                let mut ix = 0usize;
                while self.peek().is_some() && self.peek() != Some(")") {
                    if let Some(node) = self.parse_node()? {
                        let w = weights.get(&ix).copied().unwrap_or(1.0);
                        children.push((w, node));
                    }
                    ix += 1;
                }
                self.expect(")")?;

                if children.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(QueryNode::Combine { children }))
                }
            }

            "syn" => {
                self.expect("(")?;
                let terms = self.parse_term_list_drop_unresolved("#syn")?;
                self.expect(")")?;
                if terms.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(QueryNode::Syn { terms }))
                }
            }

            "band" => {
                self.expect("(")?;
                let mut children = Vec::new();
                let mut all_present = true;
                while self.peek().is_some() && self.peek() != Some(")") {
                    match self.parse_node()? {
                        Some(node) => children.push(node),
                        None => all_present = false,
                    }
                }
                self.expect(")")?;

                if !all_present || children.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(QueryNode::Band { children }))
                }
            }

            "1" => {
                self.expect("(")?;
                let terms = self.parse_term_list_all_or_none("#1")?;
                self.expect(")")?;
                match terms {
                    Some(t) if !t.is_empty() => Ok(Some(QueryNode::Phrase { terms: t })),
                    _ => Ok(None),
                }
            }

            _ if op.len() > 2
                && &op[..2] == "uw"
                && op[2..].bytes().all(|b| b.is_ascii_digit()) =>
            {
                let width: u32 = op[2..].parse().map_err(|_| {
                    QueryError::Parse(format!("bad window width in '{}'", full_tok))
                })?;
                if width < 2 {
                    return Err(QueryError::Parse(format!(
                        "window width must be >= 2, got {} (in '{}')",
                        width, full_tok
                    )));
                }
                self.expect("(")?;
                let terms = self.parse_term_list_all_or_none(op)?;
                self.expect(")")?;
                match terms {
                    Some(t) if !t.is_empty() => Ok(Some(QueryNode::Window { terms: t, width })),
                    _ => Ok(None),
                }
            }

            _ => Err(QueryError::Parse(format!(
                "unknown operator '{}'",
                full_tok
            ))),
        }
    }

    /// Term list for `#syn`: individually-unresolved terms are dropped,
    /// the rest kept. Rejects a nested operator token (v1: no nested ops
    /// where only terms are expected).
    fn parse_term_list_drop_unresolved(
        &mut self,
        op_name: &str,
    ) -> Result<Vec<TermIndex>, QueryError> {
        let mut terms = Vec::new();
        while self.peek().is_some() && self.peek() != Some(")") {
            let tok = self.next().unwrap();
            if tok.starts_with('#') {
                return Err(QueryError::Parse(format!(
                    "operator '{}' not allowed inside {}",
                    tok, op_name
                )));
            }
            if let Some(t) = (self.resolve)(tok) {
                terms.push(t);
            }
        }
        Ok(terms)
    }

    /// Term list for `#1`/`#uwN`: ANY unresolved term makes the whole node
    /// unmatchable (`Ok(None)`), since positional matching needs every
    /// term present. Still consumes every token up to the matching `)`.
    fn parse_term_list_all_or_none(
        &mut self,
        op_name: &str,
    ) -> Result<Option<Vec<TermIndex>>, QueryError> {
        let mut terms = Vec::new();
        let mut all_resolved = true;
        while self.peek().is_some() && self.peek() != Some(")") {
            let tok = self.next().unwrap();
            if tok.starts_with('#') {
                return Err(QueryError::Parse(format!(
                    "operator '{}' not allowed inside {}",
                    tok, op_name
                )));
            }
            match (self.resolve)(tok) {
                Some(t) => terms.push(t),
                None => all_resolved = false,
            }
        }
        Ok(if all_resolved { Some(terms) } else { None })
    }
}
