"""Project paths — single source of truth.

All filesystem paths used anywhere in the codebase are defined here.
They are anchored to this file's location, so everything works regardless
of which directory you launch Python from.
"""
from pathlib import Path

# This file lives at: <project_root>/src/common/paths.py
# parents[0] = src/common, parents[1] = src, parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories — where raw and processed documents live.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
# Where the Chroma vector database persists its index on disk.
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Config and runtime artifacts.
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
# Hand-authored evaluation test set + where eval reports are written.
EVAL_DIR = PROJECT_ROOT / "eval"

# Ensure runtime directories exist on import.
# (Source-controlled directories already exist; this is for robustness.)
for _dir in (RAW_DIR, PARSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, VECTOR_STORE_DIR, LOGS_DIR, EVAL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
