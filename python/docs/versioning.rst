.. _versioning:

Index versioning and migration
==============================

Every index directory contains a ``manifest.json`` recording its format
version. When a library upgrade changes the on-disk format, loading an
older index raises an error telling you to migrate:

.. code-block:: python

    # Error: "index format v1, this version requires v2 — run
    #         Index.update(path) ... to migrate"
    impact_index.Index.update("/path/to/index")            # migrate in place
    impact_index.Index.update("/path/to/index", "/dest")   # or to a copy

    index = impact_index.Index.load("/path/to/index", in_memory=True)

Migrations are streaming and fast (metadata-only where possible — e.g.
adding per-block statistics does not rewrite the postings files). Indices
that predate versioning (no ``manifest.json``) load normally and are
stamped with a manifest on first load.
