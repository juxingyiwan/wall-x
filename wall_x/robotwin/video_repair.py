"""Explicit, offline approximation of short corrupt-video gaps.

Outputs are derived observations, not recovered ground truth. Original readable
RGB frames and the original constant-rate timestamp grid are preserved exactly.
No training-time error suppression or automatic dataset replacement is provided.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class RepairPolicy:
    max_interior_gap: int = 4
    max_edge_gap: int = 2
    max_anchor_distance: int = 4
    max_motion_per_frame: float = 32.0
    min_flow_consistency: float = 0.80


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame):
    return hashlib.sha256(frame.tobytes()).hexdigest()


def decode_video(path, expected_frames=None):
    """Decode by PTS, continuing after packet errors without collapsing gaps."""
    frames, errors = {}, []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError("Expected one video stream and no audio")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate = stream.average_rate
        if rate is None or rate <= 0:
            raise ValueError("Missing positive frame rate")
        count = expected_frames if expected_frames is not None else stream.frames
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Expected frame count must be supplied for this video")
        if stream.frames and stream.frames != count:
            raise ValueError("Container and expected frame counts disagree")
        for packet in container.demux(stream):
            try:
                decoded = packet.decode()
            except av.FFmpegError as exc:
                errors.append({"pts": packet.pts, "error": str(exc)})
                continue
            for frame in decoded:
                if frame.pts is None:
                    raise ValueError("Frame has no timestamp")
                exact = frame.pts * frame.time_base * rate
                index = round(exact)
                if abs(exact - index) > Fraction(1, 1000):
                    raise ValueError(
                        "Only zero-based constant-rate videos are supported"
                    )
                if index in frames or not 0 <= index < count:
                    raise ValueError("Duplicate or out-of-range timestamp")
                rgb = (
                    frame.to_rgb(threads=1)
                    if int(av.__version__.split(".")[0]) >= 17
                    else frame.to_rgb()
                )
                frames[index] = rgb.to_ndarray()
    return frames, count, rate, errors


def _runs(indices):
    runs = []
    for index in sorted(indices):
        if not runs or index != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(index)
    return runs


def _grid(shape):
    y, x = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    return np.stack([x, y], axis=-1)


def _sample(array, positions):
    return cv2.remap(
        array,
        positions[..., 0],
        positions[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _flow_pair(left, right, separation, policy):
    gray_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    failures = []
    for estimator in ["dis", "farneback"]:
        if estimator == "dis":
            solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            forward = solver.calc(gray_left, gray_right, None)
            backward = solver.calc(gray_right, gray_left, None)
        else:
            # A second estimator is useful in low-texture close-up views. It
            # must pass exactly the same bounds; there is no unchecked fallback.
            forward = cv2.calcOpticalFlowFarneback(
                gray_left, gray_right, None, 0.5, 5, 25, 5, 7, 1.5, 0
            )
            backward = cv2.calcOpticalFlowFarneback(
                gray_right, gray_left, None, 0.5, 5, 25, 5, 7, 1.5, 0
            )
        try:
            checked = _check_flow(forward, backward, separation, policy)
            checked[-1]["estimator"] = estimator
            checked[-1]["spatial_model"] = "dense"
            regularized = _regularize_similarity(
                left, right, checked, separation, policy
            )
            if regularized is not None:
                return regularized
            return checked
        except ValueError as exc:
            failures.append(f"{estimator}: {exc}")
    raise ValueError("; ".join(failures))


def _regularize_similarity(left, right, flow, separation, policy):
    """Use a robust camera-motion model when both directions support it.

    Dense fields can tear rigid edges even when their overall consistency passes.
    A similarity map has no local folds. Require textured support and independently
    fitted forward/backward transforms; otherwise retain the dense field.
    """
    grid = _grid(left.shape)
    models, support = [], []
    for source, delta, confidence in [
        (left, flow[0], flow[2]),
        (right, flow[1], flow[3]),
    ]:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mask = (np.hypot(gx, gy) > 15) & (confidence > 0.5)
        mask[1::2, :] = False
        mask[:, 1::2] = False
        points = grid[mask]
        if len(points) < 64:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            points,
            (grid + delta)[mask],
            method=cv2.RANSAC,
            ransacReprojThreshold=2,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if matrix is None or inliers is None or float(inliers.mean()) < 0.50:
            return None
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        if not 0.8 <= scale <= 1.25:
            return None
        models.append((grid @ matrix[:, :2].T + matrix[:, 2] - grid).astype(np.float32))
        support.append(
            {
                "inlier_fraction": float(inliers.mean()),
                "textured_points": len(points),
                "matrix": matrix.tolist(),
            }
        )
    try:
        result = _check_flow(models[0], models[1], separation, policy)
    except ValueError:
        return None
    result[-1].update(
        estimator=flow[-1]["estimator"],
        spatial_model="robust_similarity",
        model_support=support,
    )
    return result


def _check_flow(forward, backward, separation, policy):
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError("Non-finite optical flow")
    grid = _grid(forward.shape)
    forward_error = np.linalg.norm(forward + _sample(backward, grid + forward), axis=-1)
    backward_error = np.linalg.norm(
        backward + _sample(forward, grid + backward), axis=-1
    )
    forward_norm = np.linalg.norm(forward, axis=-1)
    backward_norm = np.linalg.norm(backward, axis=-1)
    good_f = forward_error <= 1.0 + 0.05 * forward_norm
    good_b = backward_error <= 1.0 + 0.05 * backward_norm
    consistency = min(float(good_f.mean()), float(good_b.mean()))
    motion = (
        max(
            float(np.percentile(forward_norm, 99)),
            float(np.percentile(backward_norm, 99)),
        )
        / separation
    )
    if consistency < policy.min_flow_consistency:
        raise ValueError(f"Unreliable optical flow: consistency={consistency:.3f}")
    if motion > policy.max_motion_per_frame:
        raise ValueError(f"Motion exceeds configured bound: {motion:.3f} pixels/frame")
    confidence_f = np.exp(-forward_error / (1 + 0.05 * forward_norm)).astype(np.float32)
    confidence_b = np.exp(-backward_error / (1 + 0.05 * backward_norm)).astype(
        np.float32
    )
    return (
        forward,
        backward,
        confidence_f,
        confidence_b,
        {"flow_consistency": consistency, "motion_p99_per_frame": motion},
    )


def _inverse_map(flow, amount):
    # Solve donor + amount * flow(donor) = output coordinate.
    grid = _grid(flow.shape)
    donor = grid.copy()
    for _ in range(4):
        donor = grid - amount * _sample(flow, donor)
    return donor


def interpolate(left, right, fraction, flow):
    forward, backward, confidence_f, confidence_b, _ = flow
    map_left = _inverse_map(forward, fraction)
    map_right = _inverse_map(backward, 1 - fraction)
    a = _sample(left, map_left).astype(np.float32)
    b = _sample(right, map_right).astype(np.float32)
    wa = (1 - fraction) * _sample(confidence_f, map_left)
    wb = fraction * _sample(confidence_b, map_right)
    total = wa + wb
    mixed = (a * wa[..., None] + b * wb[..., None]) / np.maximum(total, 1e-20)[
        ..., None
    ]
    # Both directions may be occluded. Dividing tiny weights by a fixed epsilon
    # darkens those pixels; use the actual endpoints in uncertain regions.
    fallback = (1 - fraction) * left.astype(np.float32) + fraction * right.astype(
        np.float32
    )
    reliability = np.clip(total / 0.05, 0, 1)[..., None]
    mixed = reliability * mixed + (1 - reliability) * fallback
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def fill_missing_frames(original, count, bad_frames=(), policy=None):
    """Return completed frames and per-frame provenance; never reuse a repair."""
    policy = policy or RepairPolicy()
    if count <= 0 or not original:
        raise ValueError("No usable video frames")
    if (
        min(policy.max_interior_gap, policy.max_edge_gap, policy.max_anchor_distance)
        < 1
    ):
        raise ValueError("Gap bounds must be positive")
    if policy.max_motion_per_frame <= 0 or not 0 <= policy.min_flow_consistency <= 1:
        raise ValueError("Invalid optical flow thresholds")
    declared_bad = set(bad_frames)
    if any(
        not isinstance(i, int) or not 0 <= i < count
        for i in declared_bad | set(original)
    ):
        raise ValueError("Invalid frame index")
    good = {i: frame for i, frame in original.items() if i not in declared_bad}
    if not good:
        raise ValueError("No original normal frames remain")
    shapes = {frame.shape for frame in good.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3 or next(iter(shapes))[-1] != 3:
        raise ValueError("Expected consistent RGB image dimensions")
    if any(frame.dtype != np.uint8 for frame in good.values()):
        raise ValueError("Expected uint8 RGB images")
    missing = set(range(count)) - set(good)
    result, records = dict(good), []
    used_copies = set()
    for run in _runs(missing):
        left, right = run[0] - 1, run[-1] + 1
        interior = left in good and right in good
        bound = policy.max_interior_gap if interior else policy.max_edge_gap
        if len(run) > bound:
            raise ValueError(f"Gap {run} exceeds configured bound {bound}")
        if len(run) == 1:
            candidates = sorted(good, key=lambda i: (abs(i - run[0]), i))
            candidates = [
                i for i in candidates if abs(i - run[0]) <= policy.max_anchor_distance
            ]
            unused = [i for i in candidates if i not in used_copies]
            if not unused:
                raise ValueError(
                    "Nearby isolated gaps would repeatedly copy one anchor"
                )
            anchor = unused[0]
            result[run[0]] = good[anchor].copy()
            used_copies.add(anchor)
            records.append(
                {
                    "frame_index": run[0],
                    "method": "nearest_original",
                    "source_frames": [anchor],
                }
            )
        elif interior:
            flow = _flow_pair(good[left], good[right], right - left, policy)
            for i in run:
                fraction = (i - left) / (right - left)
                result[i] = interpolate(good[left], good[right], fraction, flow)
                records.append(
                    {
                        "frame_index": i,
                        "method": "bidirectional_flow_interpolation",
                        "source_frames": [left, right],
                        "fraction": fraction,
                        **flow[-1],
                    }
                )
        else:
            # Use two consecutive original frames: no recursive extrapolation and
            # no estimates obtained from another repaired gap.
            anchor = left if left in good else right
            earlier = anchor - 1 if left in good else anchor + 1
            if earlier not in good:
                raise ValueError(
                    "Edge gap requires two consecutive original anchor frames"
                )
            flow = _flow_pair(good[earlier], good[anchor], 1, policy)
            velocity = -flow[1]
            norm = np.linalg.norm(velocity, axis=-1)
            velocity *= np.minimum(
                1.0, policy.max_motion_per_frame / np.maximum(norm, 1e-6)
            )[..., None]
            velocity[norm < 0.15] = 0  # Avoid manufacturing motion from static noise.
            for i in run:
                steps = abs(i - anchor)
                donor = _inverse_map(velocity, steps)
                result[i] = _sample(good[anchor], donor)
                records.append(
                    {
                        "frame_index": i,
                        "method": "short_flow_extrapolation",
                        "source_frames": [earlier, anchor],
                        "steps": steps,
                        **flow[-1],
                    }
                )
    return result, sorted(records, key=lambda r: r["frame_index"])


def encode_lossless(path, frames, rate):
    shape = frames[0].shape
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=rate)
        stream.height, stream.width = shape[:2]
        stream.pix_fmt = "rgb24"
        stream.codec_context.thread_count = 1
        stream.options = {
            "crf": "0",
            "preset": "fast",
            "bf": "0",
            "g": str(max(1, round(rate))),
        }
        stream.time_base = 1 / rate
        for i in range(len(frames)):
            frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
            frame.pts, frame.time_base = i, 1 / rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def repair_video(
    source,
    destination,
    *,
    expected_frames=None,
    bad_frames=(),
    expected_sha256=None,
    policy=None,
):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise FileExistsError(
            "Repair destination must be new and different from source"
        )
    original_sha = file_sha256(source)
    if expected_sha256 is not None and original_sha != expected_sha256:
        raise ValueError("Source SHA256 mismatch")
    frames, count, rate, errors = decode_video(source, expected_frames)
    missing = sorted(set(range(count)) - set(frames))
    if errors and not missing and not bad_frames:
        raise ValueError(
            "Decode errors without identifiable missing frames require review"
        )
    completed, repairs = fill_missing_frames(frames, count, bad_frames, policy)
    if not repairs:
        raise ValueError("No identified bad or missing frames to repair")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        encode_lossless(temporary, completed, rate)
        decoded, decoded_count, decoded_rate, decode_errors = decode_video(
            temporary, count
        )
        hashes = [frame_sha256(completed[i]) for i in range(count)]
        if (
            decode_errors
            or decoded_count != count
            or decoded_rate != rate
            or set(decoded) != set(range(count))
        ):
            raise ValueError("Repaired file failed full decode validation")
        if [frame_sha256(decoded[i]) for i in range(count)] != hashes:
            raise ValueError("Lossless RGB verification failed")
        if file_sha256(source) != original_sha:
            raise ValueError("Source changed during repair")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "source_sha256": original_sha,
        "repaired_sha256": file_sha256(destination),
        "repaired_size": destination.stat().st_size,
        "frames": count,
        "fps_numerator": rate.numerator,
        "fps_denominator": rate.denominator,
        "approximate": True,
        "undecodable_frames": missing,
        "repairs": repairs,
        "replaced_frames": [r["frame_index"] for r in repairs],
        "intact_original_frames_preserved": count - len(repairs),
        "verified_rgb_sha256": hashes,
        "full_decode_verified": True,
        "implementation_sha256": file_sha256(__file__),
        "policy": asdict(policy or RepairPolicy()),
        "software": {
            "opencv": cv2.__version__,
            "av": av.__version__,
            "numpy": np.__version__,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, required=True, help="JSON object with videos list"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-approximation", action="store_true", required=True)
    parser.add_argument("--max-interior-gap", type=int, default=4)
    parser.add_argument("--max-edge-gap", type=int, default=2)
    parser.add_argument("--max-anchor-distance", type=int, default=4)
    parser.add_argument("--max-motion-per-frame", type=float, default=32.0)
    parser.add_argument("--min-flow-consistency", type=float, default=0.80)
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    av.logging.set_level(av.logging.PANIC)
    root = args.dataset_root.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output == root or output.is_relative_to(root):
        raise ValueError("Output must be a new staging directory outside the dataset")
    plan = json.loads(args.plan.read_text())
    rows = plan["videos"]
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Expected a nonempty plan with unique video paths")
    for row in rows:
        relative = Path(row["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".mp4"
        ):
            raise ValueError("Video paths must be dataset-relative MP4 files")
        if not (root / relative).resolve(strict=True).is_relative_to(root):
            raise ValueError("Source symlink escapes dataset")
    output.mkdir(parents=True)
    policy = RepairPolicy(
        max_interior_gap=args.max_interior_gap,
        max_edge_gap=args.max_edge_gap,
        max_anchor_distance=args.max_anchor_distance,
        max_motion_per_frame=args.max_motion_per_frame,
        min_flow_consistency=args.min_flow_consistency,
    )
    report = {
        "schema_version": 1,
        "approximate": True,
        "dataset_root": str(root),
        "dataset_modified": False,
        "videos": [],
        "state": "staging",
    }
    for row in rows:
        try:
            result = repair_video(
                root / row["path"],
                output / row["path"],
                expected_frames=row.get("expected_frames"),
                bad_frames=row.get("bad_frames", []),
                expected_sha256=row.get("sha256"),
                policy=policy,
            )
        except Exception as exc:
            report.update(state="failed", failed_video=row["path"], error=str(exc))
            (output / "repair-report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            raise
        report["videos"].append(dict(row, **result))
        (output / "repair-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"path": row["path"], "repaired_frames": len(result["repairs"])}
            ),
            flush=True,
        )
    report["state"] = "staged_and_rgb_verified"
    (output / "repair-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
