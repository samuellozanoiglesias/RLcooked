"""
generate_agent_sprite.py
------------------------
Render a randomised SpoiledBroth agent to a PNG, using the same sprite
logic as SimulationReplay / ReplayRenderer.

Usage
-----
    # One random agent → agent_<hash>.png next to the script
    python generate_agent_sprite.py

    # Specify a seed (reproducible result)
    python generate_agent_sprite.py --seed my_agent_42

    # Custom output path and tile size
    python generate_agent_sprite.py --seed hero --output hero.png --tile_size 128

    # Batch: 6 different agents in one call
    python generate_agent_sprite.py --batch 6 --output_dir ./agents

    # Show an agent holding a tomato
    python generate_agent_sprite.py --item tomato

    # Explicit indices (override randomisation)
    python generate_agent_sprite.py --skin 3 --hair 7 --mustache 0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required:  pip install Pillow")

# ---------------------------------------------------------------------------
# Paths  (mirrors the layout expected by ReplayRenderer)
# ---------------------------------------------------------------------------

# Walk up from this file until we find a 'static/sprites/agent' subtree.
def _find_sprites_root() -> Path:
    candidate = Path(__file__).resolve()
    for _ in range(8):                         # search up to 8 levels
        probe = candidate / "static" / "sprites"
        if (probe / "agent").is_dir():
            return probe
        candidate = candidate.parent
    # Fall back to a path next to the script (user can adjust)
    fallback = Path(__file__).resolve().parent / "static" / "sprites"
    return fallback


SPRITES_ROOT: Path = _find_sprites_root()

# ---------------------------------------------------------------------------
# Constants matching the engine
# ---------------------------------------------------------------------------

NUM_SKIN_VARIANTS     = 16
NUM_HAIR_VARIANTS     = 9
NUM_MUSTACHE_VARIANTS = 9

VALID_ITEMS = (
    "tomato", "pumpkin", "cabbage", "plate",
    "tomato_cut", "pumpkin_cut", "cabbage_cut",
    "tomato_salad", "pumpkin_salad", "cabbage_salad",
)

SOURCE_TILE_SIZE = 16  # px, native resolution of sprites

# Row offsets inside skin sprite sheet (rows of 16 px each)
_SKIN_ROW_IDLE        = 0   # no item
_SKIN_ROW_HOLD_ARMS   = 2   # arms-raised row used with held items (row index 2 → y=32)
_SKIN_ROW_IDLE_ONLY   = 1   # plain idle with arms (row index 1 → y=16)

_ITEM_SPRITE_Y = {
    "tomato":        0,
    "pumpkin":      16,
    "cabbage":      32,
    "plate":        48,
    "tomato_cut":    9 * 16,
    "pumpkin_cut":  10 * 16,
    "cabbage_cut":  11 * 16,
    "tomato_salad": 15 * 16,
    "pumpkin_salad":16 * 16,
    "cabbage_salad":17 * 16,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_index(token: str, modulo: int) -> int:
    """Stable deterministic index from a string token (same as ReplayRenderer)."""
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _load(relative_path: str) -> Image.Image:
    full = SPRITES_ROOT / relative_path
    if not full.exists():
        raise FileNotFoundError(
            f"Sprite not found: {full}\n"
            f"(SPRITES_ROOT = {SPRITES_ROOT})\n"
            "Check that SPRITES_ROOT points at the correct directory."
        )
    return Image.open(full).convert("RGBA")


def _crop16(sheet: Image.Image, y_offset: int) -> Image.Image:
    """Return a 16×16 tile from a sprite sheet at row y_offset."""
    return sheet.crop((0, y_offset, SOURCE_TILE_SIZE, y_offset + SOURCE_TILE_SIZE))


def _resize(tile: Image.Image, size: int) -> Image.Image:
    return tile.resize((size, size), resample=Image.NEAREST)

# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_agent(
    agent_id:      str,
    tile_size:     int  = 96,
    skin_index:    Optional[int] = None,
    hair_index:    Optional[int] = None,
    mustache_index:Optional[int] = None,
    item_type:     Optional[str] = None,
) -> Image.Image:
    """
    Compose an agent sprite using the same layering logic as ReplayRenderer.

    Parameters
    ----------
    agent_id : str
        Unique identifier used for deterministic randomisation (mirrors engine).
    tile_size : int
        Output image size in pixels (square).
    skin_index, hair_index, mustache_index : int, optional
        Override the randomised indices (0-based).  None → derive from agent_id.
    item_type : str, optional
        Item the agent is holding.  None → idle pose.

    Returns
    -------
    PIL.Image in RGBA mode, size (tile_size, tile_size).
    """
    # Resolve indices
    if skin_index is None:
        skin_index = _sha1_index(f"skin:{agent_id}", NUM_SKIN_VARIANTS)
    if hair_index is None:
        hair_index = _sha1_index(f"hair:{agent_id}", NUM_HAIR_VARIANTS)
    if mustache_index is None:
        mustache_index = _sha1_index(f"mustache:{agent_id}", NUM_MUSTACHE_VARIANTS)

    # Clamp to valid range
    skin_index     = max(0, min(skin_index,     NUM_SKIN_VARIANTS - 1))
    hair_index     = max(0, min(hair_index,     NUM_HAIR_VARIANTS - 1))
    mustache_index = max(0, min(mustache_index, NUM_MUSTACHE_VARIANTS - 1))

    # Build layer list  — same order as ReplayRenderer.render_agent
    # Each entry: (sprite_relative_path, y_offset_in_sheet)
    layers: List[Tuple[str, int]] = [
        ("agent/cook-husk.png",            0),
        (f"agent/hair/{hair_index}.png",   0),
        (f"agent/mustache/{mustache_index}.png", 0),
        (f"agent/skin/{skin_index}.png",   _SKIN_ROW_IDLE * SOURCE_TILE_SIZE),
    ]

    if item_type and item_type in _ITEM_SPRITE_Y:
        # Arms raised + item overlay
        layers.append((f"agent/skin/{skin_index}.png", _SKIN_ROW_HOLD_ARMS * SOURCE_TILE_SIZE))
        layers.append(("agent/items-held.png", _ITEM_SPRITE_Y[item_type]))
    else:
        # Idle with arms
        layers.append((f"agent/skin/{skin_index}.png", _SKIN_ROW_IDLE_ONLY * SOURCE_TILE_SIZE))

    # Composite
    canvas = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    for sprite_path, y_offset in layers:
        sheet = _load(sprite_path)
        tile  = _crop16(sheet, y_offset)
        canvas.alpha_composite(_resize(tile, tile_size))

    return canvas


def random_agent_id(rng: Optional[random.Random] = None) -> str:
    """Return a random agent-id-like string."""
    r = rng or random
    return "agent_" + "".join(r.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a SpoiledBroth agent PNG from sprite assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--seed",        type=str,  default=None,
                   help="Agent ID / seed string for deterministic look.")
    p.add_argument("--output",      type=str,  default=None,
                   help="Output PNG path.  Default: agent_<seed>.png")
    p.add_argument("--output_dir",  type=str,  default="./agents_png",
                   help="Directory for batch output (default: current dir).")
    p.add_argument("--tile_size",   type=int,  default=96,
                   help="Output image size in pixels (default: 96).")
    p.add_argument("--item",        type=str,  default=None,
                   choices=list(VALID_ITEMS) + ["none"],
                   help="Item the agent holds.")
    p.add_argument("--skin",        type=int,  default=None,
                   help=f"Skin index 0-{NUM_SKIN_VARIANTS-1} (overrides seed).")
    p.add_argument("--hair",        type=int,  default=None,
                   help=f"Hair index 0-{NUM_HAIR_VARIANTS-1} (overrides seed).")
    p.add_argument("--mustache",    type=int,  default=None,
                   help=f"Mustache index 0-{NUM_MUSTACHE_VARIANTS-1} (overrides seed).")
    p.add_argument("--batch",       type=int,  default=None,
                   help="Generate N random agents instead of one.")
    p.add_argument("--sprites_root", type=str, default=None,
                   help="Override auto-detected SPRITES_ROOT path.")
    return p.parse_args()


def _single(
    seed:           Optional[str],
    output:         Optional[str],
    tile_size:      int,
    item_type:      Optional[str],
    skin_index:     Optional[int],
    hair_index:     Optional[int],
    mustache_index: Optional[int],
    output_dir:     str = ".",
) -> Path:
    agent_id = seed if seed else random_agent_id()
    item     = None if (not item_type or item_type == "none") else item_type

    img = render_agent(
        agent_id       = agent_id,
        tile_size      = tile_size,
        skin_index     = skin_index,
        hair_index     = hair_index,
        mustache_index = mustache_index,
        item_type      = item,
    )

    if output:
        dest = Path(output)
    else:
        safe = agent_id.replace("/", "_").replace("\\", "_")
        dest = Path(output_dir) / f"agent_{safe}.png"

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG")
    return dest


def main() -> None:
    args = _parse_args()

    # Allow user to override sprite root at runtime
    if args.sprites_root:
        global SPRITES_ROOT
        SPRITES_ROOT = Path(args.sprites_root)

    if args.batch:
        rng = random.Random()          # unseeded → different each run
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.batch):
            seed = random_agent_id(rng)
            dest = _single(
                seed=seed, output=None,
                tile_size=args.tile_size,
                item_type=args.item,
                skin_index=args.skin,
                hair_index=args.hair,
                mustache_index=args.mustache,
                output_dir=str(out_dir),
            )
            print(f"[{i+1}/{args.batch}] {dest}")
        return

    # Single agent
    dest = _single(
        seed=args.seed,
        output=args.output,
        tile_size=args.tile_size,
        item_type=args.item,
        skin_index=args.skin,
        hair_index=args.hair,
        mustache_index=args.mustache,
        output_dir=args.output_dir,
    )
    print(f"Saved: {dest}")


if __name__ == "__main__":
    main()