import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file for change detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_volume_file_hashes(volume_path: str) -> Dict[str, str]:
    """Return {filename: md5_hash} for every .pptx in the volume."""
    hashes = {}
    for fname in sorted(os.listdir(volume_path)):
        if fname.lower().endswith(".pptx"):
            fpath = os.path.join(volume_path, fname)
            hashes[fname] = compute_file_hash(fpath)
    return hashes


# ── Cache persistence ─────────────────────────────────────────────

# Local cache lives next to app.py (deployed with the app)
_LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppt_cache.json")


def _volume_cache_path(volume_path: str) -> str:
    """Cache path on the UC Volume (one level above PPTs dir)."""
    parent = os.path.dirname(volume_path.rstrip("/"))
    return os.path.join(parent, "ppt_cache.json")


def load_cache(volume_path: str) -> Optional[Dict[str, Any]]:
    """Try local cache first (deployed with app), then Volume cache."""
    # 1) Local cache (most reliable in Databricks App)
    if os.path.exists(_LOCAL_CACHE):
        try:
            with open(_LOCAL_CACHE, "r") as f:
                data = json.load(f)
            logger.info("Cache loaded from local file: %s", _LOCAL_CACHE)
            return data
        except Exception as e:
            logger.warning("Failed to load local cache: %s", e)

    # 2) Volume cache (fallback)
    vol_path = _volume_cache_path(volume_path)
    if os.path.exists(vol_path):
        try:
            with open(vol_path, "r") as f:
                data = json.load(f)
            logger.info("Cache loaded from volume: %s", vol_path)
            return data
        except Exception as e:
            logger.warning("Failed to load volume cache: %s", e)

    logger.warning("No cache found at %s or %s", _LOCAL_CACHE, vol_path)
    return None


def save_cache(volume_path: str, cache_data: Dict[str, Any]):
    """Save cache locally and attempt to save to Volume."""
    # Always save locally
    try:
        with open(_LOCAL_CACHE, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Cache saved locally: %s", _LOCAL_CACHE)
    except Exception as e:
        logger.warning("Failed to save local cache: %s", e)

    # Try Volume as well
    try:
        vol_path = _volume_cache_path(volume_path)
        with open(vol_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Cache saved to volume: %s", vol_path)
    except Exception as e:
        logger.warning("Volume cache write failed (non-critical): %s", e)


# ── Change detection ──────────────────────────────────────────────

def get_changed_files(
    volume_path: str, cached_hashes: Dict[str, str]
) -> Tuple[List[str], Dict[str, str]]:
    """Return (changed_filenames, current_hashes).

    A file is 'changed' if it is new or its hash differs from the cache.
    """
    current_hashes = get_volume_file_hashes(volume_path)
    changed = [
        fname
        for fname, h in current_hashes.items()
        if fname not in cached_hashes or cached_hashes[fname] != h
    ]
    return changed, current_hashes


def get_removed_files(
    current_hashes: Dict[str, str], cached_hashes: Dict[str, str]
) -> List[str]:
    """Return filenames present in the cache but no longer on the volume."""
    return [f for f in cached_hashes if f not in current_hashes]
