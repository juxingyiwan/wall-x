"""Regular-file video storage and relocatable source lookup."""

import json
import shutil
from pathlib import Path


def store_video(source, destination, *, move=False):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if move:
        # Rename deliberately fails across filesystems; it never silently copies.
        source.rename(destination)
    else:
        shutil.copyfile(source, destination)


class RelocatedVideoIndex:
    def __init__(self, manifest=None):
        self.root = None
        self.entries = {}
        if manifest is not None:
            manifest = Path(manifest).resolve(strict=True)
            self.root = manifest.parent.parent
            data = json.loads(manifest.read_text())
            self.entries = {r["source_relative"]: r for r in data["videos"]}
            if len(self.entries) != len(data["videos"]):
                raise ValueError("Duplicate video source identifiers")

    def resolve(self, source, source_root):
        source = Path(source)
        if source.is_file():
            return source
        key = str(source.relative_to(source_root))
        if key not in self.entries:
            raise FileNotFoundError(
                f"{source}; relocated data requires --video-index /dataset/meta/video_sources.json"
            )
        entry = self.entries[key]
        target = (self.root / entry["path"]).resolve(strict=True)
        if not target.is_relative_to(self.root):
            raise ValueError("Video index points outside the prepared dataset")
        if not target.is_file() or target.stat().st_size != entry["size"]:
            raise ValueError(f"Video index size mismatch: {target}")
        return target
