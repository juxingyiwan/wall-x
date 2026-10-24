import json

import pytest

from wall_x.robotwin.storage import RelocatedVideoIndex, store_video


def test_move_preserves_inode_and_leaves_no_source_or_link(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    inode = source.stat().st_ino
    target = tmp_path / "prepared/video.mp4"
    store_video(source, target, move=True)
    assert target.stat().st_ino == inode
    assert not source.exists() and not target.is_symlink()


def test_destination_collision_does_not_modify_source(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    target = tmp_path / "target.mp4"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        store_video(source, target, move=True)
    assert source.read_bytes() == b"original" and target.read_bytes() == b"keep"


def test_relocated_video_resolves_without_old_path(tmp_path):
    root = tmp_path / "prepared"
    (root / "meta").mkdir(parents=True)
    (root / "video.mp4").write_bytes(b"video")
    manifest = root / "meta/video_sources.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {"source_relative": "task/old.mp4", "path": "video.mp4", "size": 5}
                ]
            }
        )
    )
    index = RelocatedVideoIndex(manifest)
    assert (
        index.resolve(tmp_path / "raw/task/old.mp4", tmp_path / "raw")
        == root / "video.mp4"
    )
    assert not (tmp_path / "raw").exists()


def test_manifest_cannot_escape_prepared_dataset(tmp_path):
    root = tmp_path / "prepared"
    (root / "meta").mkdir(parents=True)
    (tmp_path / "outside.mp4").write_bytes(b"video")
    manifest = root / "meta/video_sources.json"
    manifest.write_text(
        json.dumps(
            {"videos": [{"source_relative": "x", "path": "../outside.mp4", "size": 5}]}
        )
    )
    with pytest.raises(ValueError, match="outside"):
        RelocatedVideoIndex(manifest).resolve(tmp_path / "raw/x", tmp_path / "raw")


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_writer_emits_reusable_index_with_real_videos(tmp_path, mode):
    import av
    import numpy as np

    from wall_x.robotwin.convert_lerobot import DatasetWriter

    raw = tmp_path / "raw"
    raw.mkdir()
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, [3, 11]] = 1
    states[:, [7, 15]] = 0.5
    root = tmp_path / "prepared"
    writer = DatasetWriter(root, video_storage=mode)
    for split in ["train", "validation"]:
        videos = {}
        ids = {}
        for camera in ["face_view", "left_wrist_view", "right_wrist_view"]:
            path = raw / f"{split}-{camera}.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=50)
                stream.width = 640
                stream.height = 480
                stream.pix_fmt = "yuv420p"
                for index in range(2):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((480, 640, 3), index * 40, dtype=np.uint8),
                        format="rgb24",
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            key = f"observation.images.{camera}"
            videos[key] = path
            ids[key] = path.name
        writer.append(
            states,
            states,
            ["test task"] * 2,
            videos,
            split,
            {"task": "test"},
            video_sources=ids,
        )
    writer.finalize()
    manifest = root / "meta/video_sources.json"
    index = RelocatedVideoIndex(manifest)
    data = json.loads(manifest.read_text())
    assert len(data["videos"]) == 6
    for record in data["videos"]:
        target = root / record["path"]
        assert target.is_file() and not target.is_symlink()
        assert (raw / record["source_relative"]).exists() == (mode == "copy")
        if mode == "move":
            assert index.resolve(raw / record["source_relative"], raw) == target
