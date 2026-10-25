"""Smoke tests for the README asset generator.

Rendering the real banner/GIF takes ~15s, so these cover the cheap primitives and
the CLI surface only; CI separately runs ``scripts/make_assets.py`` end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

make_assets = pytest.importorskip("scripts.make_assets")
Image = pytest.importorskip("PIL.Image")


def test_gradient_image_ramps_between_the_two_colours():
    gradient = make_assets._gradient_image(120, 60, "#101828", "#2E90FA")
    assert gradient.shape == (60, 120, 3)
    # Dark at the top-left, blue at the bottom-right, and never flat in between.
    assert gradient[0, 0].sum() < 0.5
    assert gradient[-1, -1].sum() > 1.5
    assert len({tuple(pixel.round(3)) for pixel in gradient[0]}) > 20


def test_rounded_alpha_masks_the_corners():
    source = Image.new("RGB", (100, 50), (16, 24, 40))
    rounded = make_assets._rounded_alpha(source, radius=12)
    assert rounded.size == (100, 50)
    assert rounded.mode == "RGBA"
    assert rounded.getpixel((0, 0))[3] == 0  # corner is cut away
    assert rounded.getpixel((50, 25))[3] == 255  # centre is opaque


def test_demo_frame_storyboard_covers_every_stage():
    frames = make_assets._demo_frames()
    assert len(frames) >= 20
    assert frames[0].active_agent == 0
    assert {frame.active_agent for frame in frames} == {0, 1, 2, 3}
    captions = " ".join(frame.status for frame in frames)
    assert "Reviewing the draft" in captions
    assert "Report ready" in captions
    # Progress values only ever move forward, so the animation never rewinds.
    prices = [frame.price_progress for frame in frames]
    bars = [frame.bars_progress for frame in frames]
    assert prices == sorted(prices)
    assert bars == sorted(bars)


def test_human_size_formats_kilobytes(tmp_path):
    target = tmp_path / "blob.bin"
    target.write_bytes(b"x" * 2048)
    assert make_assets.human_size(target) == "2 KB"


def test_cli_rejects_unknown_flags():
    with pytest.raises(SystemExit):
        make_assets.main(["--definitely-not-a-flag"])
