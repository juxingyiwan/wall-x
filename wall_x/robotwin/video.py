"""Bound PyAV decode/conversion threads for RoboTwin DataLoader workers."""


def install_bounded_video_reader():
    import av
    import torch
    import torchvision

    original = torchvision.io.VideoReader
    if getattr(original, "_robotwin_bounded_threads", False):
        return
    explicit_rgb_threads = int(av.__version__.split(".")[0]) >= 17

    class ThreadBoundVideoReader(original):
        _robotwin_bounded_threads = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.backend == "pyav":
                for stream in self.container.streams.video:
                    stream.codec_context.thread_count = 1

        def __next__(self):
            if self.backend != "pyav" or "video" not in self.pyav_stream:
                return super().__next__()
            try:
                frame = next(self._c)
                pts = float(frame.pts * frame.time_base)
                rgb = (
                    frame.to_rgb(threads=1) if explicit_rgb_threads else frame.to_rgb()
                )
                data = torch.as_tensor(rgb.to_ndarray()).permute(2, 0, 1)
            except av.error.EOFError:
                raise StopIteration
            if data.numel() == 0:
                raise StopIteration
            return {"data": data, "pts": pts}

    torchvision.io.VideoReader = ThreadBoundVideoReader
