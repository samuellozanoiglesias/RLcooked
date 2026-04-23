"""Helpers for resolving map text directories.

The codebase now stores map layouts in named subfolders such as
`maps_txt_classic` and `maps_txt_competition`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

MAPS_ROOT = Path(__file__).resolve().parent
DEFAULT_MAPS_FOLDER = "maps_txt_classic"


def normalize_maps_folder_name(
    game_version: Optional[str] = None,
    maps_folder: Optional[str] = None,
) -> str:
    """Return the canonical maps subfolder name for a game or explicit folder."""
    if maps_folder:
        folder_name = maps_folder.strip().rstrip("/")
        if not folder_name.startswith("maps_txt_"):
            folder_name = f"maps_txt_{folder_name}"
        return folder_name

    if game_version:
        normalized = game_version.strip().lower()
        if normalized.endswith("_collision"):
            normalized = normalized.removesuffix("_collision")
        if not normalized.startswith("maps_txt_"):
            normalized = f"maps_txt_{normalized}"
        return normalized

    return DEFAULT_MAPS_FOLDER


def get_maps_txt_dir(
    game_version: Optional[str] = None,
    maps_folder: Optional[str] = None,
) -> Path:
    """Resolve the directory that stores text maps for the requested variant."""
    return MAPS_ROOT / normalize_maps_folder_name(game_version, maps_folder)


def get_map_txt_path(
    map_nr: str,
    game_version: Optional[str] = None,
    maps_folder: Optional[str] = None,
) -> Path:
    """Resolve the text-file path for a map."""
    return get_maps_txt_dir(game_version=game_version, maps_folder=maps_folder) / f"{map_nr}.txt"


def iter_maps_txt_dirs() -> List[Path]:
    """Return all named map-text directories that currently exist."""
    return sorted(path for path in MAPS_ROOT.glob("maps_txt_*") if path.is_dir())
