from fractions import Fraction

import cv2
import numpy as np
import pytest

from wall_x.robotwin.video_repair import (
    decode_video,
    encode_lossless,
    fill_missing_frames,
    frame_sha256,
    interpolate,
    repair_video,
)


def translation_sequence(count=10):
    rng = np.random.default_rng(2026)
    base = rng.integers(0, 256, (128, 192, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    return {
        i: cv2.warpAffine(
            base,
            np.float32([[1, 0, 2 * i], [0, 1, 0]]),
            (192, 128),
            borderMode=cv2.BORDER_REFLECT,
        )
        for i in range(count)
    }


def mse(a, b):
    return np.mean((a[12:-12, 24:-24].astype(float) - b[12:-12, 24:-24]) ** 2)


@pytest.mark.parametrize("bad", [[3, 4], [3, 4, 5, 6], [8, 9], [0, 1]])
def test_moving_gaps_follow_time_direction_without_flat_copy_runs(bad):
    truth = translation_sequence()
    original = {i: frame for i, frame in truth.items() if i not in bad}
    repaired, records = fill_missing_frames(original, 10)
    assert set(repaired) == set(truth)
    assert len({frame_sha256(repaired[i]) for i in bad}) == len(bad)
    for i in original:
        assert np.array_equal(repaired[i], original[i])
    for i in bad:
        nearest = min(original, key=lambda j: abs(i - j))
        assert mse(repaired[i], truth[i]) < mse(original[nearest], truth[i]) * 0.5
    assert all(set(r["source_frames"]).issubset(original) for r in records)


def test_nearby_isolated_gaps_use_distinct_original_anchors():
    truth = translation_sequence()
    original = {i: frame for i, frame in truth.items() if i not in [4, 6]}
    _, records = fill_missing_frames(original, 10)
    anchors = [r["source_frames"][0] for r in records]
    assert len(set(anchors)) == 2
    assert set(anchors).issubset(original)


def test_stationary_scene_does_not_get_artificial_motion():
    truth = translation_sequence(1)[0]
    output, _ = fill_missing_frames({0: truth, 1: truth, 6: truth}, 7)
    assert all(np.array_equal(frame, truth) for frame in output.values())


def test_occluded_zero_confidence_pixels_do_not_turn_black():
    left = np.full((64, 64, 3), 180, dtype=np.uint8)
    right = np.full_like(left, 220)
    flow = np.zeros((64, 64, 2), dtype=np.float32)
    confidence = np.zeros((64, 64), dtype=np.float32)
    result = interpolate(left, right, 0.5, (flow, flow, confidence, confidence, {}))
    assert np.array_equal(result, np.full_like(left, 200))


def test_large_gaps_and_missing_edge_anchors_fail_explicitly():
    frames = translation_sequence()
    with pytest.raises(ValueError, match="exceeds"):
        fill_missing_frames({0: frames[0], 6: frames[6]}, 7)
    with pytest.raises(ValueError, match="two consecutive"):
        fill_missing_frames({0: frames[0], 2: frames[2]}, 5)


def test_source_hash_collision_and_invalid_frames_are_rejected(tmp_path):
    source = tmp_path / "source.mp4"
    encode_lossless(source, translation_sequence(), Fraction(50))
    with pytest.raises(FileExistsError):
        repair_video(source, source, bad_frames=[3])
    with pytest.raises(ValueError, match="SHA256"):
        repair_video(
            source, tmp_path / "out.mp4", bad_frames=[3], expected_sha256="wrong"
        )
    with pytest.raises(ValueError, match="Invalid frame"):
        fill_missing_frames(translation_sequence(), 10, bad_frames=[-1])


def test_lossless_roundtrip_preserves_fps_and_all_untouched_frames(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "repaired.mp4"
    truth = translation_sequence()
    encode_lossless(source, truth, Fraction(30000, 1001))
    before = source.read_bytes()
    result = repair_video(source, output, bad_frames=[3, 4, 8, 9])
    decoded, count, rate, errors = decode_video(output)
    assert count == 10 and rate == Fraction(30000, 1001) and not errors
    assert result["approximate"] and result["full_decode_verified"]
    assert source.read_bytes() == before
    for i in [0, 1, 2, 5, 6, 7]:
        assert np.array_equal(decoded[i], truth[i])
    assert result["replaced_frames"] == [3, 4, 8, 9]


def test_inconsistent_count_and_variable_rate_are_not_silently_retimed(tmp_path):
    source = tmp_path / "source.mp4"
    encode_lossless(source, translation_sequence(), Fraction(50))
    with pytest.raises(ValueError, match="counts disagree"):
        decode_video(source, 11)
