"""Restore the newest durable GitHub release archive for a product."""

import argparse
import io
import os
from pathlib import Path
import zipfile
import requests


def restore(repository, kind, root):
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
        }
    )
    page = 1
    while True:
        response = session.get(
            f"https://api.github.com/repos/{repository}/releases",
            params={"per_page": 100, "page": page},
            timeout=60,
        )
        response.raise_for_status()
        releases = response.json()
        if not releases:
            return
        for release in releases:
            if not release["draft"] and release["tag_name"].startswith(
                "refresh-" + kind + "-"
            ):
                assets = [a for a in release["assets"] if a["name"] == "snapshot.zip"]
                if len(assets) != 1:
                    continue
                response = session.get(
                    assets[0]["url"],
                    headers={"Accept": "application/octet-stream"},
                    timeout=120,
                )
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    for name in archive.namelist():
                        if not (root / name).resolve().is_relative_to(root.resolve()):
                            raise ValueError("Archive path escapes destination")
                    archive.extractall(root)
                import hashlib, json

                latest = root / kind / "latest.json"
                entry = json.loads(latest.read_text())
                snapshot = (root / entry["snapshot"]).resolve()
                if not snapshot.is_relative_to(root.resolve()):
                    raise ValueError("Invalid snapshot path")
                for name, digest in entry["files"].items():
                    path = (snapshot / name).resolve()
                    if (
                        not path.is_relative_to(snapshot)
                        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                    ):
                        raise ValueError("Archive checksum mismatch")
                return
        page += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", required=True)
    p.add_argument("--kind", required=True)
    p.add_argument("--root", type=Path, default=Path("data/refresh"))
    a = p.parse_args()
    restore(a.repository, a.kind, a.root)
