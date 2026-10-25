"""Guards that keep the README (and its images) from rotting.

Documentation is part of the deliverable: broken image paths and dead anchors are
easy to introduce and hard to notice. These checks are cheap and run in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ASSETS = ROOT / "docs" / "assets"

EXPECTED_ASSETS = ("banner.png", "architecture.png", "demo.gif", "report-preview.png")


def github_slug(heading: str) -> str:
    """Reproduce GitHub's heading-anchor algorithm closely enough for our headings."""
    slug = "".join(char for char in heading.strip().lower() if char.isalnum() or char in " -_")
    return slug.replace(" ", "-")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


class TestReadmeAssets:
    @pytest.mark.parametrize("asset", EXPECTED_ASSETS)
    def test_asset_exists_and_is_not_empty(self, asset):
        path = ASSETS / asset
        assert path.exists(), f"missing README asset: {path.relative_to(ROOT)}"
        assert path.stat().st_size > 5_000, f"{asset} looks truncated"

    def test_every_local_reference_resolves(self, readme):
        targets = set(re.findall(r'(?:src|href)="([^"]+)"', readme))
        targets |= {
            group[1] or group[0]
            for group in re.findall(r"!?\[[^\]]*\]\(([^)#:]+)(#[^)]*)?\)", readme)
        }
        missing = [
            target
            for target in targets
            if not target.startswith(("http://", "https://", "#", "mailto:"))
            and not (ROOT / target.split("#")[0]).exists()
        ]
        assert missing == [], f"README points at missing files: {missing}"

    def test_animated_gif_is_actually_animated(self):
        from PIL import Image

        with Image.open(ASSETS / "demo.gif") as gif:
            assert gif.n_frames >= 10, "the demo GIF should contain several frames"
            assert gif.size[0] >= 640


class TestReadmeStructure:
    def test_all_anchor_links_resolve(self, readme):
        headings = {
            github_slug(match.group(2)) for match in re.finditer(r"^(#{1,6})\s+(.*)$", readme, re.M)
        }
        anchors = set(re.findall(r"\]\(#([^)]+)\)", readme))
        anchors |= set(re.findall(r'href="#([^"]+)"', readme))
        assert sorted(anchor for anchor in anchors if anchor not in headings) == []

    def test_code_fences_are_balanced(self, readme):
        assert readme.count("```") % 2 == 0

    def test_tables_are_well_formed(self, readme):
        malformed = [
            line
            for line in readme.splitlines()
            if line.startswith("|") and not line.rstrip().endswith("|")
        ]
        assert malformed == []

    def test_no_markdown_trapped_inside_raw_html(self, readme):
        """GitHub does not render Markdown inside block-level HTML."""
        offenders = []
        for block in re.finditer(r"<(div|table|tr|td)[^>]*>(.*?)</\1>", readme, re.S):
            inner = re.sub(r"<code>.*?</code>", "", block.group(2), flags=re.S)
            inner = re.sub(r"<img[^>]*>", "", inner)
            if re.search(r"(\*\*|\]\(|^#{1,6}\s|^\s*[-*]\s|`[^`]+`)", inner, re.M):
                offenders.append(re.sub(r"\s+", " ", inner)[:80])
        assert offenders == []

    @pytest.mark.parametrize(
        "section",
        [
            "Quick start",
            "Architecture",
            "Agent prompts & report contract",
            "Configuration",
            "Docker",
            "Deployment",
            "Testing & code quality",
            "Disclaimer",
        ],
    )
    def test_expected_sections_exist(self, readme, section):
        assert re.search(rf"^#{{1,6}}\s+\S*\s*{re.escape(section)}\s*$", readme, re.M)

    def test_quick_start_has_the_run_command(self, readme):
        assert "pip install -r requirements.txt" in readme
        assert "streamlit run app.py" in readme

    def test_disclaimer_is_stated(self, readme):
        assert "not investment advice" in readme.lower()


def test_asset_generator_is_documented(readme):
    """The generator that produced the images must be discoverable from the README."""
    assert "scripts/make_assets.py" in readme
