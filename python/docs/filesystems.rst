Network filesystems
===================

Some operations copy index files: compressing or reordering an index copies
its vocabulary, analyzer and document metadata files into the new index, and
migrating an index into another directory copies the whole index. When the
source and destination are on the same filesystem, files are hard-linked
instead, and nothing is copied.

On Linux, the native file copy (``copy_file_range``) lets the filesystem copy
the data itself. On NFS v4.2 this becomes a server-side copy, and some NFS
servers never finish it: the process then hangs forever in the kernel (in
``nfs42_proc_copy``). To avoid this, impact-index has a process-wide *copy
mode*:

``auto`` (default)
    Copies through a user-space buffer when the source or the destination is
    on a network filesystem (NFS, SMB/CIFS, Lustre, Ceph, GPFS, AFS, 9P), and
    uses the native copy otherwise. Network filesystems are only detected on
    Linux, the only platform where the native copy asks the server to copy.

``buffered``
    Always copies through a user-space buffer: slower on local disks, never
    hangs.

``native``
    Always uses the native copy (``copy_file_range`` and reflinks on Linux,
    ``clonefile`` on macOS), even on network filesystems.

The mode is set from Python, for the whole process:

.. code-block:: python

    import impact_index

    impact_index.set_copy_mode("buffered")
    print(impact_index.copy_mode())  # "buffered"

or, without changing any code (e.g. for cluster jobs), with the
``IMPACT_INDEX_COPY_MODE`` environment variable, read the first time a file is
copied:

.. code-block:: bash

    export IMPACT_INDEX_COPY_MODE=buffered

An explicit ``set_copy_mode`` takes precedence over the environment variable.
From Rust, use ``impact_index::set_copy_mode(CopyMode::Buffered)``.
