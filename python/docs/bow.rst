.. _bm25:

Bag-of-words indices (BM25)
===========================

For traditional IR with BM25 scoring, use
:class:`~impact_index.BOWIndexBuilder` instead of
:class:`~impact_index.IndexBuilder`. It automatically tracks document
lengths and optionally integrates text analysis (tokenization + stemming).

.. note::

    :class:`~impact_index.BOWIndexBuilder` is a layer on top of the same
    storage engine as :class:`~impact_index.IndexBuilder`: a BOW index
    *is* a sparse impact index (postings, WAND/MaxScore search,
    compression, BMP conversion all apply unchanged), where the "impact
    value" happens to be a raw or analyzer-computed term frequency. What
    ``BOWIndexBuilder`` adds on top is bookkeeping BM25 needs but a raw
    impact index doesn't: per-document length tracking and, optionally,
    the text analysis pipeline (tokenizer, stemmer, stop words,
    vocabulary) described below.

Pre-tokenized input
-------------------

If you already have term indices and term-frequency values:

.. code-block:: python

    import numpy as np
    import impact_index

    builder = impact_index.BOWIndexBuilder("/path/to/index", dtype="int32")

    # Add documents: docid, term_indices, tf_values
    terms = np.array([0, 5, 42], dtype=np.uintp)
    tf = np.array([3, 1, 2], dtype=np.int32)
    builder.add(0, terms, tf)

    builder.add(1, np.array([2, 5, 8], dtype=np.uintp),
                np.array([1, 4, 1], dtype=np.int32))

    # Build returns searchable Index (doc metadata stored automatically)
    index = builder.build(in_memory=True)

    # Create a BM25-scored index (doc lengths loaded automatically)
    scored = index.with_scoring(impact_index.BM25Scoring(k1=1.2, b=0.75))

    # Search with MaxScore (fastest algorithm)
    query = {0: 1.0, 5: 1.0}
    results = scored.search_maxscore(query, top_k=10)
    for doc in results:
        print(f"Document {doc.docid}: {doc.score}")

Raw text input with stemming
----------------------------

For direct text indexing with automatic tokenization, stemming, and
vocabulary management:

.. code-block:: python

    import impact_index

    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        dtype="int32",
        stemmer="porter",  # Lucene-compatible Porter stemmer
        stop_words=True,   # Lucene default English stop words
    )

    builder.add_text(0, "the quick brown fox jumps over the lazy dog")
    builder.add_text(1, "a quick brown cat jumps high")
    builder.add_text(2, "the lazy dog sleeps all day")

    # Build index (doc metadata and analyzer config saved automatically)
    index = builder.build(in_memory=True)

    # BM25 scoring (doc lengths loaded automatically from index)
    scored = index.with_scoring(impact_index.BM25Scoring())

    # Query analysis (analyzer loaded automatically from index)
    query = index.analyzer().analyze_query("quick fox")
    results = scored.search_maxscore(query, top_k=10)

Matching a reference pipeline
-----------------------------

A "pipeline" is the *combination* of tokenizer, stemmer, and stop-word
list/timing a reference IR system uses — and the three don't vary
independently: Lucene/Pyserini, Terrier, and PISA each disagree on more
than one axis at once (see the table below). ``pipeline=`` is the *only*
``BOWIndexBuilder`` argument that names another system; picking one sets
all three axes to match that system's own defaults in one go:

.. code-block:: python

    import impact_index

    # Matches Lucene/Anserini/Pyserini's own defaults
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index", pipeline="pyserini", stop_words=True,
    )

    # Matches real Terrier 5's own defaults (and closely approximates
    # PISA's tokenizer -- see the table below)
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index", pipeline="terrier", stop_words=True,
    )

    # Matches PISA's own index instead (as built by the pyterrier_pisa
    # wrapper commonly compared against) -- see the table below for why
    # this differs from "terrier"
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index", pipeline="terrier-pisa",
    )

``stemmer=``, ``stop_words=``, and ``language=`` still compose freely on
top of a ``pipeline=`` choice — an explicit value overrides just that one
axis, keeping the pipeline's tokenizer and stop-word timing:

.. code-block:: python

    # Terrier's tokenizer + stop-word timing, but PISA's own stemmer
    # (Snowball/Porter2) instead of real Terrier 5's classic Porter
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index", pipeline="terrier", stemmer="snowball",
    )

Omitting ``pipeline`` entirely falls back to today's defaults (equivalent
to ``pipeline="pyserini"`` for English text), so existing code that sets
``stemmer=``/``stop_words=`` directly keeps working unchanged.

**Why all three axes at once matters:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - System
     - Tokenizer
     - Stemmer
     - Stop words checked
   * - Lucene/Anserini/Pyserini
     - strips trailing ``'s`` only
     - classic Porter (``PorterStemFilter``)
     - *before* stemming, against the raw token (Lucene's ~33-word list)
   * - Terrier 5 (real)
     - (Java tokenizer, not independently verified here)
     - classic Porter
     - *before* stemming, against the raw token (Terrier's own ~730-word
       list) -- default ``termpipelines=Stopwords,PorterStemmer``, verified
       by dumping an isolated Terrier 5 index's lexicon directly
   * - PISA (native CLI)
     - truncates at the *first* apostrophe anywhere
     - Snowball/Porter2
     - *after* stemming, against the raw (never-stemmed) list -- PISA's own
       ``tools/app.cpp``, not exercised by the ``pyterrier_pisa`` wrapper
       this project benchmarks against (see below)

``pipeline="terrier"`` matches the middle row (real Terrier 5, verified):
pre-stem filtering, both at index time and query time, plus PISA's own
tokenizer (the closest available approximation to Terrier 5's, whose Java
tokenizer hasn't been independently verified).

``pipeline="terrier-pisa"`` instead matches the bottom row's index -- but
not by using post-stem filtering (that's PISA's *native CLI tool*, not
what its commonly-used Python wrapper actually does). The
``pyterrier_pisa`` wrapper this project's PISA benchmark numbers are built
with doesn't remove stop words from its *index* at all, at any setting --
"the" alone ends up indexed in 87% of MS MARCO passages -- but its own
*query* processing does exclude them (querying it with only stop words
returns no results). ``"terrier-pisa"`` reproduces exactly that asymmetry:
nothing filtered when building the index, Terrier's list still filtered
when analyzing a query. Filtering neither side, or both, measurably hurts
agreement with a real PISA index built this way. See `BENCHMARKS.md
<https://github.com/xpmir/impact-index/blob/master/BENCHMARKS.md>`_
for the measured numbers (full-query-set result overlap: 0.98 against PISA
with ``"terrier-pisa"``, 0.96 against Terrier 5 with ``"terrier"`` -- each
pipeline is a close match for the system it targets, not for the other
one).

The tokenizer difference is easy to miss but changes a large fraction of
the vocabulary: PISA's ``EnglishTokenStream`` (``tools/tokenizer.cpp`` in
the PISA source) doesn't have a separate "possessive filter" step the way
Lucene does — its *only* rule for an apostrophe is "keep the substring
before the first one", applied to *any* token matching
``[a-zA-Z0-9]+('[a-zA-Z]+)``. So ``"don't"`` -> ``"don"`` and
``"it's"`` -> ``"it"``, not just ``"king's"`` -> ``"king"``. Lucene's
``EnglishPossessiveFilter`` only ever strips a *trailing* ``'s``, leaving
``"don't"`` untouched. impact-index's ``pipeline="terrier"`` replicates
PISA's rule exactly (``Tokenizer::PisaEnglish`` in
``src/vocab/analyzer.rs``); ``pipeline="pyserini"`` (or no ``pipeline`` at
all) replicates Lucene's (``Tokenizer::LuceneEnglish``). Roughly 18% of MS
MARCO passages contain at least one apostrophe token, so getting this
wrong silently mismatches a real IR system on a large slice of the
collection's vocabulary — not just on the possessives the name suggests.

Choosing a stemmer
------------------

Two stemmers are available via ``stemmer=`` (or picked automatically by
``pipeline=``, see above):

- ``"porter"`` — a direct port of Lucene's ``PorterStemFilter``. Pyserini's
  own default, and real Terrier 5's.
- ``"snowball"`` — the classic Porter2/Snowball algorithm. PISA's own
  default (and what ``pipeline="terrier"`` picks by default — pass
  ``stemmer="porter"`` explicitly to match real Terrier 5's stemmer
  instead).
- ``None`` (default without a stemmer) — no stemming.

The two disagree on some common words (e.g. "community", "day", "use"
stem differently), so pick based on which system you're comparing against
or reproducing rather than assuming they're interchangeable. See the
top-level README's Performance section for a benchmark of both
configurations against their respective reference systems.

Tokenizer variant
------------------

Besides stemming, the analyzer also picks a tokenizer variant at the Rust
level: ``Tokenizer::Standard``, ``Tokenizer::LuceneEnglish``, or
``Tokenizer::PisaEnglish`` (``src/vocab/analyzer.rs`` — see the pipeline
table above for what each one does with an apostrophe). This is a
tokenization concern, not a stemming one, so it composes with any stemmer
choice (or none).

.. note::

    There is no separate ``BOWIndexBuilder`` argument for the tokenizer
    variant directly — it's selected via ``pipeline=`` (recommended) or
    implied by ``language="english"`` (the default) for backward
    compatibility when no ``pipeline`` is given, which always means
    ``LuceneEnglish``. Every other language gets plain ``Standard``
    splitting. Use the lower-level Rust ``TextAnalyzer`` API directly
    (``set_tokenizer``) for anything not covered by these two paths.

Stop words
----------

Stop words (common words like "the", "is", "a") can be filtered during
indexing and querying to reduce index size and improve search speed.
Two built-in *families* are available, selectable independently of the
stemmer/language settings: ``"lucene"`` (default; short, per-language
lists matching Lucene's language analyzers, 17 languages) and
``"terrier"`` (Terrier's own, much longer list — the default PISA and
Terrier 5 themselves use; English only).

.. code-block:: python

    import impact_index

    # Use default stop words for the language (matches Lucene/Pyserini)
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="snowball",
        language="english",
        stop_words=True,
    )

    # Or select the Terrier family (matches PISA/Terrier 5 defaults)
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="snowball",
        language="english",
        stop_words="terrier",
    )

    # Or provide an explicit list
    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="snowball",
        stop_words=["the", "a", "is", "in"],
    )

    # Get the stop word list for any supported language/family
    words = impact_index.get_stop_words("english")              # 33 words (Lucene default)
    words = impact_index.get_stop_words("french")                # 154 words (Lucene)
    words = impact_index.get_stop_words("german")                 # 231 words (Lucene)
    words = impact_index.get_stop_words("english", "terrier")     # 733 words (Terrier)

Supported Lucene-family languages: arabic, danish, dutch, english,
finnish, french, german, greek, hungarian, italian, norwegian,
portuguese, romanian, russian, spanish, swedish, turkish. The Terrier
family covers English only; requesting it for another language raises
an error rather than silently substituting a Lucene list.

``stop_words=True`` is a permanent alias for ``stop_words="lucene"``, and
the family (or custom list) an index was built with is saved and
restored automatically on reload.

.. note::

    For fair comparison with Pyserini/Lucene, always enable stop words.
    Without them, high-frequency terms like "the" create very long
    posting lists that slow down search significantly.

.. note::

    Both built-in families filter the raw token *before* stemming,
    matching their reference system exactly: the ``"lucene"`` family as
    Lucene's ``EnglishAnalyzer`` does, and the ``"terrier"`` family as
    real Terrier 5's default ``termpipelines=Stopwords,PorterStemmer``
    does (verified by dumping an isolated Terrier 5 index's lexicon
    directly). So an inflected form like "however" is caught directly
    against the raw list, before it would otherwise stem to "howev". A
    custom ``stop_words=[...]`` list checks both pre- and post-stem
    stages, since there's no single reference pipeline to match. With an
    explicit ``pipeline=`` (see above), that pipeline's own timing always
    wins, even if ``stop_words=`` is also overridden with a custom list or
    the other family's built-in one.

Loading a saved index
---------------------

The index automatically detects and loads auxiliary components
(doc metadata, analyzer config, vocabulary) from the directory:

.. code-block:: python

    import impact_index

    index = impact_index.Index.load("/path/to/index", in_memory=True)

    # Doc metadata and analyzer are loaded automatically
    scored = index.with_scoring(impact_index.BM25Scoring())
    analyzer = index.analyzer()
    query = analyzer.analyze_query("quick fox")
    results = scored.search_maxscore(query, top_k=10)

How BM25 scores are computed
----------------------------

Scores are computed at query time: the index stores raw term frequencies,
and :class:`~impact_index.BM25Scoring` turns them into scores. For a query
``q`` and a document ``d`` of length ``dl`` (number of indexed tokens):

.. math::

    \mathrm{score}(q, d) = \sum_{t \in q} w_t \cdot \mathrm{idf}(t) \cdot
        \frac{(k_1 + 1)\, \mathrm{tf}_{t,d}}
             {k_1 \left(1 - b + b \frac{dl}{\mathrm{avgdl}}\right) + \mathrm{tf}_{t,d}}

- ``k1`` and ``b``: the usual BM25 parameters (defaults 1.2 and 0.75).
  ``avgdl`` is the average document length, and ``N`` below is the number
  of documents.
- ``idf`` depends on ``variant``. ``"bm25"`` (default, used by Terrier and
  PISA) is ``ln((N - df + 0.5) / (df + 0.5))``, floored at a tiny positive
  value so very common terms never get a negative weight. ``"lucene"``
  (Pyserini/Anserini) is ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
- ``w_t`` is the query weight of term ``t``: its count in the query for
  :meth:`~impact_index.TextAnalyzer.analyze_query`, or the value you put in
  the query dict. With ``k3=None`` (default), weights are used as they are
  (Lucene, PISA). With ``k3`` set (Terrier uses ``k3=8``), weights are
  first divided by the query's largest weight, then saturated as
  ``(k3 + 1) w / (k3 + w)``, as Terrier 5's BM25 does.

Rankings match the reference systems. Absolute score values may differ by
a constant factor: Terrier, for example, uses base-2 logarithms.

.. code-block:: python

    # Terrier 5's BM25
    scored = index.with_scoring(impact_index.BM25Scoring(k1=1.2, b=0.75, k3=8))
    # Pyserini/Anserini's BM25
    scored = index.with_scoring(impact_index.BM25Scoring(k1=0.9, b=0.4, variant="lucene"))

Token positions
---------------

By default, a BOW index stores only term frequencies — enough for BM25,
but not enough to know whether two terms were adjacent. Building with
``positions=True`` additionally records each term's token positions
within each document, which the phrase (``#1``) and window (``#uwN``)
structured query operators below need to evaluate.

.. code-block:: python

    builder = impact_index.BOWIndexBuilder(
        "/path/to/index",
        stemmer="porter",
        stop_words=True,
        positions=True,
    )
    builder.add_text(0, "the quick brown fox jumps over the lazy dog")
    builder.add_text(1, "a quick brown cat jumps high")
    index = builder.build(in_memory=True)

``position_gaps`` controls what happens to positions when stop words are
removed at indexing time:

- ``position_gaps=True`` (Lucene; the default without a pipeline and for
  ``pipeline="pyserini"``): a removed stop word keeps its position, so
  "bank of america" stores ``bank@0 america@2``.
- ``position_gaps=False`` (Terrier 5; the default for
  ``pipeline="terrier"`` and ``"terrier-pisa"``): positions only count the
  tokens that were kept, so it stores ``bank@0 america@1``.

Matchop queries drop stop words inside operators too, so
``#1(bank of america)`` becomes ``#1(bank america)``. It matches "bank of
america" only without gaps.

.. note::

    From Python, positional indexing only goes through text: with
    ``positions=True``, ``add_text``/``add_texts`` work as usual, but the
    pre-tokenized ``add(docid, terms, values)`` method raises an error
    instead (its message points to a Rust-only ``add_with_positions``
    method that isn't exposed to Python). Positions are stored per block
    and decoded lazily, so a query without ``#1``/``#uwN`` pays no extra
    cost at search time — the cost is the extra on-disk positions data
    written at build time.

Structured queries
------------------

Beyond flat ``{term_id: weight}`` dicts,
:meth:`~impact_index.Index.search_wand_query` /
:meth:`~impact_index.Index.search_maxscore_query` (and the
:class:`~impact_index.ScoredIndex` equivalents) accept Terrier-matchop-style
structured queries. They are evaluated as "virtual" posting lists, pruned
by the same WAND/MaxScore algorithms as flat queries:

- ``#combine(...)``: weighted sum of children (the default combinator
  when a query has several terms). ``#combine:0=2:1=1(quick fox)``
  weights the first child 2x and the second 1x.
- ``#syn(t1 t2 ...)``: synonyms. Matches documents containing any of the
  terms, scored as one term.
- ``#band(n1 n2 ...)``: boolean AND. Matches documents containing every
  child.
- ``#1(t1 t2 ...)``: exact phrase (adjacent positions). **Requires an
  index built with** ``positions=True``.
- ``#uwN(t1 t2 ...)``: unordered window. All terms occur within a span of
  ``N`` tokens, in any order. **Requires positions**, like ``#1``.

.. code-block:: python

    scored = index.with_scoring(impact_index.BM25Scoring(k3=8))

    results = scored.search_wand_query(
        "#combine(#1(new york) #syn(city town) #band(guide budget))",
        top_k=10,
    )

A matchop string is resolved with the index's own analyzer (the same
tokenizer, stemmer and stop words used at indexing time), so it requires
an index built with ``BOWIndexBuilder``. Stop words are removed
everywhere, including inside ``#1``/``#uwN``/``#band``. A word that is not
in the vocabulary is dropped from ``#combine``/``#syn``, but makes a whole
``#1``/``#uwN``/``#band`` unmatchable, so that operator is dropped.

You can also build the query tree directly from term ids, as nested
dicts, with no analyzer involved: ``{"term": ix}`` (or ``{"term": [ix,
weight]}``), ``{"combine": [[w1, node1], ...]}``, ``{"syn": [ix, ...]}``,
``{"band": [node, ...]}``, ``{"phrase": [ix, ...]}``, or ``{"window":
{"terms": [ix, ...], "width": N}}``.

How structured queries are scored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scoring follows Terrier 5's matchop semantics. With a Terrier-aligned
index (``pipeline="terrier", stemmer="porter"``) and
``BM25Scoring(k3=8)``, rankings are identical to Terrier's.

1. **Flattening.** Nested ``#combine`` operators are flattened into one
   list of clauses. A clause's weight is the product of the ``#combine``
   weights on its path: ``#combine:0=0.8:1=0.2(#combine(a b) #1(a b))``
   has clauses ``a`` (0.8), ``b`` (0.8) and ``#1(a b)`` (0.2). Identical
   clauses are merged and their weights summed, so ``#combine(a a b)`` is
   ``#combine:0=2(a b)``.
2. **Query weights.** The scoring model adjusts the clause weights
   (``k3`` for BM25, see `How BM25 scores are computed`_). The score of a
   document is the weighted sum of its clause scores.
3. **Clause scores.** A term is scored as usual. Every other operator is
   scored as one *virtual term*, with the model's own formula (for
   example BM25 above), using this virtual term frequency (tf) and
   document frequency (df):

   ================ ============================================= ========================
   Operator         Virtual tf in a document                      Virtual df
   ================ ============================================= ========================
   ``#syn``         sum of the children's tfs                     sum of the children's dfs
   ``#band``        1 (the document contains every child)         sum of the children's dfs
   ``#1``           number of phrase occurrences                  ``N / 100``
   ``#uwN``         number of occurrences of the rarest term      ``N / 100``
                    that fall in a window of ``N`` tokens
                    containing all the other terms
   ================ ============================================= ========================

   ``N / 100`` (integer division) is a heuristic that Terrier inherited
   from Ivory. Counting a phrase's real df would need a pass over every
   document containing all its terms. As a result, every phrase and every
   window gets the same idf, whatever its terms.

Over a raw (unscored) index, the same operators evaluate to raw values
instead: tfs for ``#syn``, counts for ``#1``/``#uwN``, and the sum of the
children's values for ``#band``. This keeps ``#band`` useful on
learned-impact indices.


API reference
-------------

.. autoapiclass:: impact_index.BOWIndexBuilder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.TextAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.BM25Scoring
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.ScoredIndex
   :members:
   :undoc-members:
   :show-inheritance:

.. autoapiclass:: impact_index.DocMetadata
   :members:
   :undoc-members:
   :show-inheritance:
