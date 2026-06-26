"""Create and initialize persistent Modal volumes for Utah SSL runs.

Usage:

    modal run scripts/modal/create_utah_ssl_volumes.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "utah-ssl-create-volumes"
CACHE_VOLUME_NAME = "utah-ssl-cache"
OUTPUT_VOLUME_NAME = "utah-ssl-outputs"

CACHE_MOUNT = Path("/vol/cache")
OUTPUT_MOUNT = Path("/vol/outputs")


cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    cpu=1,
    volumes={
        str(CACHE_MOUNT): cache_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def initialize_volumes() -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat()

    cache_sentinel = CACHE_MOUNT / ".volume_initialized.txt"
    output_sentinel = OUTPUT_MOUNT / ".volume_initialized.txt"

    cache_sentinel.write_text(
        f"Persistent Modal cache volume initialized at {timestamp}\n"
    )
    output_sentinel.write_text(
        f"Persistent Modal output volume initialized at {timestamp}\n"
    )

    cache_volume.commit()
    output_volume.commit()

    return {
        "cache_volume": CACHE_VOLUME_NAME,
        "output_volume": OUTPUT_VOLUME_NAME,
        "cache_mount": str(CACHE_MOUNT),
        "output_mount": str(OUTPUT_MOUNT),
        "cache_sentinel": str(cache_sentinel),
        "output_sentinel": str(output_sentinel),
    }


@app.local_entrypoint()
def main() -> None:
    result = initialize_volumes.remote()
    print("Created/verified persistent Modal volumes:")
    for key, value in result.items():
        print(f"{key}: {value}")
