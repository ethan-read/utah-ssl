"""Extract a tar archive inside a persistent Modal volume.

Typical workflow:

1. Upload one archive:

   modal volume put utah-ssl-cache /path/to/cache_v1_smoothed_sigma2p0.tar /uploads/cache_v1_smoothed_sigma2p0.tar

2. Extract it inside Modal:

   modal run workflows/modal/extract_volume_archive.py --archive /uploads/cache_v1_smoothed_sigma2p0.tar --dest /
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import modal


APP_NAME = "utah-ssl-extract-volume-archive"
CACHE_VOLUME_NAME = "utah-ssl-cache"
CACHE_MOUNT = Path("/vol/cache")


cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    cpu=4,
    timeout=60 * 60 * 12,
    volumes={str(CACHE_MOUNT): cache_volume},
)
def extract_archive(
    *,
    archive_path: str,
    dest_dir: str = "/",
) -> dict[str, str]:
    archive = CACHE_MOUNT / archive_path.lstrip("/")
    destination = CACHE_MOUNT / dest_dir.lstrip("/")
    destination.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        raise FileNotFoundError(f"Archive not found in volume: {archive}")

    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(destination)

    cache_volume.commit()
    return {
        "archive": str(archive),
        "destination": str(destination),
    }


@app.local_entrypoint()
def main(
    archive: str,
    dest: str = "/",
) -> None:
    result = extract_archive.remote(archive_path=archive, dest_dir=dest)
    print("Extraction complete:")
    for key, value in result.items():
        print(f"{key}: {value}")
