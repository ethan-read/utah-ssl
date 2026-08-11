"""Write a JSON snapshot of the current SSL autoresearch data inventory."""

from __future__ import annotations

import json

from data import discover_b2t25_sources, inventory_summary, inventory_to_jsonable, load_b2t25_cache_inventory
from prepare import BRAINTOTEXT25_ROOT, SBP_CACHE_DIR, TX_CACHE_DIR, ensure_artifact_dirs, source_root_metadata


def main() -> int:
    artifacts = ensure_artifact_dirs()
    entries = load_b2t25_cache_inventory(TX_CACHE_DIR, SBP_CACHE_DIR)
    payload = {
        "summary": inventory_summary(entries),
        "source_roots": source_root_metadata(),
        "sources": [source.__dict__ for source in discover_b2t25_sources(BRAINTOTEXT25_ROOT)],
        "entries": inventory_to_jsonable(entries),
    }
    out_path = artifacts.inventory_dir / "brain2text25_inventory.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
