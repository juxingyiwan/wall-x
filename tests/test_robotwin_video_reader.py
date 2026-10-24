"""Bounding decoder threads must preserve frames, timestamps and EOF."""

from fractions import Fraction

import av
import numpy as np
import torch
import torchvision

from wall_x.robotwin.video import install_bounded_video_reader


def test_bounded_reader_preserves_decoded_pixels_and_timestamps(monkeypatch, tmp_path):
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=50)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "rgb24"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for i in range(6):
            pixels = np.full((64, 96, 3), i * 35, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 50)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    old_backend = torchvision.get_video_backend()
    monkeypatch.setattr(torchvision.io, "VideoReader", torchvision.io.VideoReader)
    try:
        torchvision.set_video_backend("pyav")
        original = torchvision.io.VideoReader(str(path), "video")
        try:
            expected = list(original)
        finally:
            original.container.close()
        install_bounded_video_reader()
        installed = torchvision.io.VideoReader
        install_bounded_video_reader()
        assert torchvision.io.VideoReader is installed
        bounded = installed(str(path), "video")
        try:
            assert bounded.container.streams.video[0].codec_context.thread_count == 1
            actual = list(bounded)
        finally:
            bounded.container.close()
        assert len(actual) == len(expected) == 6
        for result, reference in zip(actual, expected):
            assert result["pts"] == reference["pts"]
            torch.testing.assert_close(
                result["data"], reference["data"], rtol=0, atol=0
            )
    finally:
        torchvision.set_video_backend(old_backend)
