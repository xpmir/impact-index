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
use crate::search::ops::{AndCursor, AndValue, PhraseCursor, SumCursor, WindowCursor};
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
    /// Weighted sum of children's scores (matchop `#combine`). [`evaluate`]
    /// flattens every `Combine` (nested ones included) into the top-level
    /// `(weight, cursor)` list the search loops consume.
    Combine { children: Vec<(f32, QueryNode)> },
    /// Synonym/OR (matchop `#syn`): children's term frequencies are summed
    /// and the merged posting list is scored as ONE virtual term (df = sum
    /// of the children's dfs). Children must be plain terms.
    Syn { terms: Vec<TermIndex> },
    /// Boolean AND (matchop `#band`): matches docs containing every child.
    /// Scored as ONE virtual term with tf = 1 and df = sum of the
    /// children's dfs (Terrier); over a raw index, the value is the sum of
    /// the children's values instead. Children can't be `Combine`.
    Band { children: Vec<QueryNode> },
    /// Exact phrase (matchop `#1`): adjacent positions, scored as a virtual
    /// term with tf = match count and df = N / 100. Requires an index
    /// built with positions ([`QueryError::PositionsNotAvailable`]
    /// otherwise).
    Phrase { terms: Vec<TermIndex> },
    /// Unordered window of width `width` tokens (matchop `#uwN`), scored
    /// like [`Phrase`](QueryNode::Phrase). Requires positions.
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
/// Scoring follows Terrier 5's matchop semantics (see the "Structured
/// queries" section of the Python guide for the full description):
///
/// - Every `Combine` is flattened: each non-`Combine` descendant becomes
///   one top-level clause whose weight is the product of the `#combine`
///   weights on its path (times [`QueryNode::Term::weight`] for a term);
///   identical clauses are merged, their weights summed.
///   Over a [`ScoredIndex`] the resulting weights then go through
///   [`crate::scoring::ScoringModel::query_weights`] (e.g. BM25's `k3`).
/// - A term clause is scored like a plain term.
/// - `Syn`/`Band`/`Phrase`/`Window` clauses are *virtual terms*: a raw
///   posting list whose "tf" is the operator's own frequency (summed tfs /
///   1 / occurrence count), scored by the model's
///   [`crate::scoring::ScoringModel::term_scorer`] with a virtual document
///   frequency -- the sum of the children's dfs for `Syn` and `Band`, and
///   `N / 100` for `Phrase`/`Window` (Terrier's heuristic, from Ivory).
///
/// Over a raw (unscored) index the same tree evaluates in raw-value space
/// (no scorer, weights unchanged); there `Band` sums its children's values
/// instead of reporting 1, which keeps it useful on learned-impact indices.
///
/// A clause that cannot match anything (e.g. an empty sub-tree) is
/// dropped; `EmptyQuery` (see [`QueryError`]) is returned only when no
/// clause is left. [`search_wand_query`]/[`search_maxscore_query`] treat it
/// as "no results", not a hard error.
pub fn evaluate<'a>(
    index: &'a dyn SparseIndex,
    node: &QueryNode,
) -> Result<Vec<(f32, Box<dyn BlockTermImpactIterator + 'a>)>, QueryError> {
    node.validate()?;
    let scored = index.as_any().downcast_ref::<ScoredIndex>();

    let mut leaves = Vec::new();
    flatten(node, 1.0, &mut leaves);
    // Identical clauses merge into one, weights summed (Terrier's
    // MatchingQueryTerms): `#combine(a a b)` is `#combine:0=2(a b)`, which
    // matters once `query_weights` is non-linear.
    let mut clauses: Vec<(f32, &QueryNode)> = Vec::with_capacity(leaves.len());
    for (weight, leaf) in leaves {
        match clauses.iter_mut().find(|(_, c)| same_clause(c, leaf)) {
            Some((w, _)) => *w += weight,
            None => clauses.push((weight, leaf)),
        }
    }

    let mut weights = Vec::with_capacity(clauses.len());
    let mut cursors = Vec::with_capacity(clauses.len());
    for (weight, clause) in clauses {
        match eval_clause(index, scored, clause) {
            Ok(cursor) => {
                weights.push(weight);
                cursors.push(cursor);
            }
            Err(QueryError::EmptyQuery) => {}
            Err(e) => return Err(e),
        }
    }
    if cursors.is_empty() {
        return Err(QueryError::EmptyQuery);
    }
    if let Some(scored) = scored {
        scored.model().query_weights(&mut weights);
    }
    Ok(weights.into_iter().zip(cursors).collect())
}

/// Collects every non-`Combine` node under `node` with its accumulated
/// weight (product of the `#combine` weights on the path, times the term's
/// own weight for a [`QueryNode::Term`]).
fn flatten<'n>(node: &'n QueryNode, weight: f32, out: &mut Vec<(f32, &'n QueryNode)>) {
    match node {
        QueryNode::Combine { children } => {
            for (w, child) in children {
                flatten(child, weight * w, out);
            }
        }
        QueryNode::Term { weight: w, .. } => out.push((weight * w, node)),
        _ => out.push((weight, node)),
    }
}

/// Whether two flattened clauses denote the same virtual term (a term's
/// own weight is already folded into the clause weight, so it is ignored).
fn same_clause(a: &QueryNode, b: &QueryNode) -> bool {
    match (a, b) {
        (QueryNode::Term { term: x, .. }, QueryNode::Term { term: y, .. }) => x == y,
        _ => a == b,
    }
}

/// Evaluates one top-level clause (anything but a `Combine`) to a cursor:
/// scored when `index` is a [`ScoredIndex`], raw otherwise.
fn eval_clause<'a>(
    index: &'a dyn SparseIndex,
    scored: Option<&'a ScoredIndex>,
    node: &QueryNode,
) -> Result<Box<dyn BlockTermImpactIterator + 'a>, QueryError> {
    match (node, scored) {
        (QueryNode::Term { term, .. }, _) => Ok(index.block_iterator(*term)),
        (_, Some(scored)) => {
            let num_docs = scored.num_docs();
            let (df, raw) = eval_raw(scored.inner_index(), Some(num_docs), node)?;
            // A summed df (`#syn`/`#band`) can exceed N, where idf formulas
            // break down (Terrier's own BM25 takes the log of a negative
            // number there); capping at N keeps every idf variant finite and
            // non-negative without affecting any df <= N.
            let scorer = scored
                .model()
                .term_scorer(df.min(num_docs), raw.max_value());
            Ok(wrap_scored_cursor(raw, scorer))
        }
        (_, None) => Ok(eval_raw(index, None, node)?.1),
    }
}

/// Evaluates `node` to a raw (unscored) virtual posting list over `source`,
/// together with its virtual document frequency (Terrier's statistics, see
/// [`evaluate`]). `num_docs` is `Some(N)` when the result is going to be
/// scored -- which also selects `Band`'s tf = 1 semantics -- and `None`
/// for raw-value evaluation.
fn eval_raw<'a>(
    source: &'a dyn SparseIndex,
    num_docs: Option<u64>,
    node: &QueryNode,
) -> Result<(u64, Box<dyn BlockTermImpactIterator + 'a>), QueryError> {
    match node {
        QueryNode::Term { term, .. } => {
            let cursor = source.block_iterator(*term);
            Ok((cursor.length() as u64, cursor))
        }

        QueryNode::Combine { .. } => Err(QueryError::Parse(
            "#combine can only appear at the top level or inside another #combine".to_string(),
        )),

        QueryNode::Syn { terms } => {
            if terms.is_empty() {
                return Err(QueryError::EmptyQuery);
            }
            let (dfs, children) = raw_term_children(source, terms);
            let children = children.into_iter().map(|c| (1.0f32, c)).collect();
            Ok((dfs.iter().sum(), Box::new(SumCursor::new(children))))
        }

        QueryNode::Band { children } => {
            if children.is_empty() {
                return Err(QueryError::EmptyQuery);
            }
            let mut df = 0u64;
            let mut cursors = Vec::with_capacity(children.len());
            for child in children {
                let (child_df, cursor) = eval_raw(source, num_docs, child)?;
                df += child_df;
                cursors.push(cursor);
            }
            let mode = if num_docs.is_some() {
                AndValue::One
            } else {
                AndValue::Sum
            };
            Ok((df, Box::new(AndCursor::new(cursors, mode))))
        }

        QueryNode::Phrase { terms } | QueryNode::Window { terms, .. } => {
            if !SparseIndex::has_positions(source) {
                return Err(QueryError::PositionsNotAvailable);
            }
            if terms.is_empty() {
                return Err(QueryError::EmptyQuery);
            }
            let (_, children) = raw_term_children(source, terms);
            let cursor: Box<dyn BlockTermImpactIterator + 'a> = match node {
                QueryNode::Window { width, .. } => Box::new(WindowCursor::new(children, *width)),
                _ => Box::new(PhraseCursor::new(children)),
            };
            // Terrier's PhraseOp/UnorderedWindowOp: a fixed df of N/100
            // (integer division), whatever the terms.
            Ok((num_docs.unwrap_or(0) / 100, cursor))
        }
    }
}

/// Raw (unscored) per-term cursors from `source` for every term in
/// `terms`, plus each one's document frequency (`length()`).
fn raw_term_children<'a>(
    source: &'a dyn SparseIndex,
    terms: &[TermIndex],
) -> (Vec<u64>, Vec<Box<dyn BlockTermImpactIterator + 'a>>) {
    let children: Vec<_> = terms.iter().map(|&t| source.block_iterator(t)).collect();
    let dfs = children.iter().map(|c| c.length() as u64).collect();
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
/// Here every `None` from `resolve` counts as an unknown term; use
/// [`parse_matchop_with`] to have stop words skipped instead (what the
/// Python API does).
///
/// The parser itself is a small hand-rolled recursive descent (no new
/// dependencies): malformed input (unbalanced parens, an unknown operator,
/// a malformed `:IDX=WEIGHT` suffix, an operator nested where only plain
/// terms are allowed) is a [`QueryError::Parse`].
pub fn parse_matchop(
    text: &str,
    resolve: &dyn Fn(&str) -> Option<TermIndex>,
) -> Result<QueryNode, QueryError> {
    parse_matchop_with(text, &|tok| match resolve(tok) {
        Some(t) => Resolved::Term(t),
        None => Resolved::Unknown,
    })
}

/// How a query token resolved against the vocabulary (see
/// [`parse_matchop_with`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolved {
    /// The token maps to this term.
    Term(TermIndex),
    /// The token is removed by the analyzer (a stop word, or nothing left
    /// after tokenization): it is skipped everywhere, as Terrier's term
    /// pipeline does -- `#1(bank of america)` is `#1(bank america)`.
    Stopword,
    /// The token is a real word absent from the index: it can never match.
    Unknown,
}

/// Like [`parse_matchop`], but with a resolver that tells stop words
/// ([`Resolved::Stopword`], silently skipped even inside `#1`/`#uwN`/
/// `#band`) apart from unknown terms ([`Resolved::Unknown`], which drop
/// the whole enclosing `#1`/`#uwN`/`#band`).
pub fn parse_matchop_with(
    text: &str,
    resolve: &dyn Fn(&str) -> Resolved,
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
    resolve: &'r dyn Fn(&str) -> Resolved,
}

/// Outcome of parsing one node.
enum Parsed {
    Node(QueryNode),
    /// Nothing to add, but nothing wrong either (a stop word).
    Skip,
    /// Can never match (unknown term, or an operator pruned to nothing).
    Fail,
}

impl Parsed {
    fn from_option(node: Option<QueryNode>) -> Self {
        node.map_or(Parsed::Fail, Parsed::Node)
    }
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
            if let Parsed::Node(node) = self.parse_node()? {
                children.push((1.0f32, node));
            }
        }
        if children.len() == 1 && children[0].0 == 1.0 {
            Ok(children.into_iter().next().unwrap().1)
        } else {
            Ok(QueryNode::Combine { children })
        }
    }

    /// Parses one node: an operator (`#...`) or a bare term (see
    /// [`Parsed`] and [`parse_matchop`]'s doc comment for dropping rules).
    fn parse_node(&mut self) -> Result<Parsed, QueryError> {
        let tok = self
            .peek()
            .ok_or_else(|| QueryError::Parse("unexpected end of input".to_string()))?;
        if let Some(stripped) = tok.strip_prefix('#') {
            Ok(Parsed::from_option(self.parse_op(stripped)?))
        } else {
            self.next();
            Ok(match (self.resolve)(tok) {
                Resolved::Term(term) => Parsed::Node(QueryNode::Term { term, weight: 1.0 }),
                Resolved::Stopword => Parsed::Skip,
                Resolved::Unknown => Parsed::Fail,
            })
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
                    if let Parsed::Node(node) = self.parse_node()? {
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
                        Parsed::Node(node) => children.push(node),
                        Parsed::Skip => {}
                        Parsed::Fail => all_present = false,
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
            if let Resolved::Term(t) = (self.resolve)(tok) {
                terms.push(t);
            }
        }
        Ok(terms)
    }

    /// Term list for `#1`/`#uwN`: stop words are skipped, but ANY unknown
    /// term makes the whole node unmatchable (`Ok(None)`), since positional
    /// matching needs every term present. Still consumes every token up to the matching `)`.
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
                Resolved::Term(t) => terms.push(t),
                Resolved::Stopword => {}
                Resolved::Unknown => all_resolved = false,
            }
        }
        Ok(if all_resolved { Some(terms) } else { None })
    }
}
