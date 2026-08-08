#!/usr/bin/env bash
# apply_patches.sh — Apply required ChromaDB patches after pip install
#
# ChromaDB 0.5.5 has two bugs that surface when reading a ChromaDB store
# that was created by an older version (seq_id INTEGER vs BYTES, and
# PersistentData serialised as a plain dict).  These patches fix both.
#
# Usage:
#   bash patches/apply_patches.sh            # uses ./venv by default
#   VENV=path/to/venv bash patches/apply_patches.sh

set -euo pipefail

VENV="${VENV:-./venv}"
CHROMA_BASE="$VENV/lib/python3.11/site-packages/chromadb"

SQLITE_FILE="$CHROMA_BASE/segment/impl/metadata/sqlite.py"
HNSW_FILE="$CHROMA_BASE/segment/impl/vector/local_persistent_hnsw.py"

echo "Applying ChromaDB patches..."

patch -p1 --forward "$SQLITE_FILE" \
    < "$(dirname "$0")/chromadb_sqlite_decode_seq_id.patch" \
    || echo "  [SKIP] sqlite patch already applied or conflicts — check manually."

patch -p1 --forward "$HNSW_FILE" \
    < "$(dirname "$0")/chromadb_hnsw_persistent_data.patch" \
    || echo "  [SKIP] hnsw patch already applied or conflicts — check manually."

echo "Done. Patches applied to $CHROMA_BASE"
