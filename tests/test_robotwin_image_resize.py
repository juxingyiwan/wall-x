"""RoboTwin training and direct inference use identical image preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from wall_x._vendor.harrix.utils.train_config import (
    normalize_train_config_for_inference,
)
from wall_x.data.backends.lerobot.config import LerobotConfig
from wall_x.data.backends.lerobot.loader import PreprocessedDataset, get_data_configs
from wall_x.robotwin.adapter import RobotwinInferAdapter
from wall_x.robotwin.build import _build_flat_config
from wall_x.robotwin.config import load_config
from wall_x.robotwin.dataset import RobotwinPreprocessedDataset
from wall_x.robotwin.images import resize_robotwin_square


@pytest.mark.parametrize(
    "shape,bbox",
    [
        ((640, 480), (0, 28, 224, 196)),
        ((480, 640), (28, 0, 196, 224)),
        ((256, 256), (0, 0, 224, 224)),
    ],
)
def test_square_padding_keeps_complete_image_and_centers_it(shape, bbox):
    image = Image.new("RGB", shape, (255, 255, 255))
    result = resize_robotwin_square(image, 224)
    assert result.size == (224, 224)
    assert result.getbbox() == bbox
    assert image.size == shape


@pytest.mark.parametrize("size", [0, -1, True, 224.5])
def test_invalid_square_size_rejected(size):
    with pytest.raises(ValueError):
        resize_robotwin_square(Image.new("RGB", (640, 480)), size)


@pytest.mark.parametrize("pad", [False, True])
def test_robotwin_config_option_survives_training_and_inference_loaders(tmp_path, pad):
    lc = {"repo_id": "local/test", "adapter": "robotwin"}
    if pad:
        lc["pad_to_square"] = True
    raw = {
        "model_type": "qwen2_5",
        "task": {"dof_config": {"position": 3}},
        "model": {"config_path": "unused", "processor_path": "unused"},
        "data": {
            "dataset_type": "lerobot",
            "lerobot_config": lc,
            "key_mappings": {
                "camera": {},
                "state": "observation.state",
                "action": "action",
            },
            "resolution": {"face_view": 224},
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(raw))
    flat, actual_lc = _build_flat_config(load_config(str(path)))
    wrapped = RobotwinPreprocessedDataset(
        SimpleNamespace(),
        flat,
        {},
        get_data_configs(flat["data"]),
        actual_lc,
        test_only=True,
    )
    assert wrapped.lerobot_config.get("pad_to_square", False) is pad
    normalized = normalize_train_config_for_inference(raw, str(path))
    assert normalized["data"]["lerobot_config"].get("pad_to_square", False) is pad


@pytest.mark.parametrize(
    "pad,size,source_shape,expected",
    [
        (True, 224, (640, 480), (224, 224)),
        (True, 224, (480, 640), (224, 224)),
        (False, 448, (640, 480), (448, 336)),
        (False, 224, (640, 480), (224, 168)),
        (False, 256, (256, 256), (252, 252)),
    ],
)
def test_training_and_direct_inference_preprocess_agree(
    pad, size, source_shape, expected
):
    width, height = source_shape
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    pixels[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4, 1] = 255
    views = ["face_view", "left_wrist_view", "right_wrist_view"]
    keys = [f"observation.images.{view}" for view in views]
    resolution = dict.fromkeys(views, size)
    dataset = RobotwinPreprocessedDataset.__new__(RobotwinPreprocessedDataset)
    dataset.hf_dataset = SimpleNamespace(meta=SimpleNamespace(camera_keys=keys))
    dataset._cam_key_mapping = dict(zip(keys, views))
    dataset.data_config = LerobotConfig(resolution=resolution)
    dataset.lerobot_config = {"adapter": "robotwin", "pad_to_square": pad}
    frame = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    frames = dict.fromkeys(keys, frame)
    trained, *_ = dataset._vision_preprocess(frames)
    adapter = RobotwinInferAdapter.__new__(RobotwinInferAdapter)
    adapter._device = "cpu"
    adapter._data_config = dataset.data_config
    adapter._train_config = {
        "data": {"lerobot_config": {"adapter": "robotwin", "pad_to_square": pad}}
    }
    inferred = [adapter._resize_image(pixels, view) for view in views]
    assert len(trained) == len(inferred) == 3
    for train_image, infer_image in zip(trained, inferred):
        assert train_image.size == expected
        np.testing.assert_array_equal(np.asarray(train_image), infer_image.numpy())
    if not pad:
        legacy, *_ = PreprocessedDataset._vision_preprocess(dataset, frames)
        for actual, old in zip(trained, legacy):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(old))
