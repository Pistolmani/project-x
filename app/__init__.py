"""AI Investigation Engine: the code shared by the API, the UI and the offline ingestion."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep model and dataset caches inside the project (git-ignored) so everything lives in one folder.
# Must run before transformers / datasets / fastembed are imported, hence here.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(PROJECT_ROOT / ".cache" / "fastembed"))
