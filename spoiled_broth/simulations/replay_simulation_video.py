"""Offline replay for simulation CSV outputs.

This module reconstructs a replay video from the CSV artifacts written by the
simulation pipeline. It uses:
- positions_ai_rl_*.csv for agent motion, carried item, and score state
- counters.csv for counter contents
- items.csv to translate item IDs into item types
- map text or PNG assets under spoiled_broth/maps
- sprite sheets under spoiled_broth/static/sprites

The resulting video does not require the live game engine.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .video_recorder import VideoRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPRITES_ROOT = PROJECT_ROOT / "static" / "sprites"
MAPS_ROOT = PROJECT_ROOT / "maps"


@dataclass
class AgentFrame:
    pixel_x: float
    pixel_y: float
    tile_x: int
    tile_y: int
    item: str
    item_id: str
    score: int


@dataclass
class Snapshot:
    frame: int
    second: float
    agents: Dict[str, AgentFrame]
    counters: Dict[Tuple[int, int], str]


class ReplayRenderer:
    """Render the replay scene using the same sprite sheets as the engine."""

    def __init__(self, tile_size: int = 96):
        self.tile_size = tile_size
        self.source_tile_size = 16.0
        self._sprite_cache: Dict[str, Image.Image] = {}

    def _sprite_path(self, relative_path: str) -> Path:
        return SPRITES_ROOT / relative_path

    def _load_sprite(self, relative_path: str) -> Image.Image:
        if relative_path not in self._sprite_cache:
            sprite_path = self._sprite_path(relative_path)
            if not sprite_path.exists():
                raise FileNotFoundError(f"Sprite not found: {sprite_path}")
            self._sprite_cache[relative_path] = Image.open(sprite_path).convert("RGBA")
        return self._sprite_cache[relative_path]

    def _stable_index(self, token: str, modulo: int) -> int:
        if modulo <= 0:
            return 0
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) % modulo

    def _resize_tile(self, sprite: Image.Image) -> Image.Image:
        return sprite.resize((self.tile_size, self.tile_size), resample=Image.NEAREST)

    def _paste_tile(self, canvas: Image.Image, sprite: Image.Image, x: int, y: int) -> None:
        canvas.alpha_composite(sprite, (x * self.tile_size, y * self.tile_size))

    def _scale_world_coord(self, value: float) -> float:
        return value * (self.tile_size / self.source_tile_size)

    def _item_sprite_y(self, item_type: str) -> int:
        return {
            "tomato": 0,
            "pumpkin": 16,
            "cabbage": 32,
            "plate": 48,
            "tomato_cut": 9 * 16,
            "pumpkin_cut": 10 * 16,
            "cabbage_cut": 11 * 16,
            "tomato_salad": 15 * 16,
            "pumpkin_salad": 16 * 16,
            "cabbage_salad": 17 * 16,
        }.get(item_type, 0)

    def render_map(self, map_lines: List[str], map_image: Optional[Image.Image] = None) -> Image.Image:
        if map_lines:
            height = len(map_lines)
            width = len(map_lines[0])
            canvas = Image.new("RGBA", (width * self.tile_size, height * self.tile_size), (0, 0, 0, 255))

            char_to_tile = {
                "W": "wall",
                "M": "counter",
                "B": "cutting_board",
                "T": "dispenser_tomato",
                "P": "dispenser_pumpkin",
                "X": "dispenser_plate",
                "C": "dispenser_cabbage",
                "D": "delivery",
                " ": "floor",
                "1": "floor",
                "2": "floor",
            }

            for y, line in enumerate(map_lines):
                for x, char in enumerate(line):
                    tile_type = char_to_tile.get(char, "floor")
                    if tile_type == "wall":
                        self._render_wall(canvas, x, y)
                    elif tile_type == "counter":
                        self._render_counter(canvas, x, y)
                    elif tile_type == "cutting_board":
                        self._render_cutting_board(canvas, x, y)
                    elif tile_type == "dispenser_tomato":
                        self._render_dispenser(canvas, x, y, "tomato")
                    elif tile_type == "dispenser_pumpkin":
                        self._render_dispenser(canvas, x, y, "pumpkin")
                    elif tile_type == "dispenser_plate":
                        self._render_dispenser(canvas, x, y, "plate")
                    elif tile_type == "dispenser_cabbage":
                        self._render_dispenser(canvas, x, y, "cabbage")
                    elif tile_type == "delivery":
                        self._render_delivery(canvas, x, y)
                    else:
                        self._render_floor(canvas, x, y)

            return canvas

        if map_image is not None:
            width = max(1, int(round(map_image.width * self.tile_size / 16.0)))
            height = max(1, int(round(map_image.height * self.tile_size / 16.0)))
            return map_image.resize((width, height), resample=Image.NEAREST).convert("RGBA")

        raise ValueError("Map is empty")

    def _render_floor(self, canvas: Image.Image, x: int, y: int) -> None:
        sprite = self._load_sprite("world/basic-floor.png")
        rows = max(1, sprite.height // 16)
        row = self._stable_index(f"floor:{x}:{y}", rows)
        crop = sprite.crop((0, row * 16, 16, row * 16 + 16))
        self._paste_tile(canvas, self._resize_tile(crop), x, y)

    def _render_wall(self, canvas: Image.Image, x: int, y: int) -> None:
        sprite = self._load_sprite("world/basic-wall.png")
        self._paste_tile(canvas, self._resize_tile(sprite.crop((0, 0, 16, 16))), x, y)

    def _render_counter(self, canvas: Image.Image, x: int, y: int) -> None:
        sprite = self._load_sprite("world/basic-counter.png")
        self._paste_tile(canvas, self._resize_tile(sprite.crop((0, 0, 16, 16))), x, y)

    def _render_cutting_board(self, canvas: Image.Image, x: int, y: int) -> None:
        self._render_counter(canvas, x, y)
        board = self._load_sprite("world/cutting-board.png")
        knife = self._load_sprite("world/knife.png")
        self._paste_tile(canvas, self._resize_tile(board.crop((0, 0, 16, 16))), x, y)
        self._paste_tile(canvas, self._resize_tile(knife.crop((0, 0, 16, 16))), x, y)

    def _render_dispenser(self, canvas: Image.Image, x: int, y: int, item_type: str) -> None:
        self._render_counter(canvas, x, y)
        sprite = self._load_sprite("world/item-dispenser.png")
        src_y = {"tomato": 0, "pumpkin": 16, "cabbage": 32, "plate": 48}.get(item_type, 0)
        crop = sprite.crop((0, src_y, 16, src_y + 16))
        self._paste_tile(canvas, self._resize_tile(crop), x, y)

    def _render_delivery(self, canvas: Image.Image, x: int, y: int) -> None:
        sprite = self._load_sprite("world/delivery.png")
        self._paste_tile(canvas, self._resize_tile(sprite.crop((0, 0, 16, 16))), x, y)

    def render_counter_item(self, canvas: Image.Image, x: int, y: int, item_type: str) -> None:
        sprite = self._load_sprite("world/item-on-counter.png")
        crop = sprite.crop((0, self._item_sprite_y(item_type), 16, self._item_sprite_y(item_type) + 16))
        self._paste_tile(canvas, self._resize_tile(crop), x, y)

    def render_agent(self, canvas: Image.Image, agent_id: str, pixel_x: float, pixel_y: float, item_type: str) -> None:
        skin_index = self._stable_index(f"skin:{agent_id}", 16)
        hair_index = self._stable_index(f"hair:{agent_id}", 9)
        mustache_index = self._stable_index(f"mustache:{agent_id}", 9)

        layers: List[Tuple[str, Tuple[int, int, int, int]]] = [
            ("agent/cook-husk.png", (0, 0, 16, 16)),
            (f"agent/hair/{hair_index}.png", (0, 0, 16, 16)),
            (f"agent/mustache/{mustache_index}.png", (0, 0, 16, 16)),
            (f"agent/skin/{skin_index}.png", (0, 0, 16, 16)),
        ]

        if item_type:
            layers.append((f"agent/skin/{skin_index}.png", (0, 32, 16, 48)))
            y0 = self._item_sprite_y(item_type)
            layers.append(("agent/items-held.png", (0, y0, 16, y0 + 16)))
        else:
            layers.append((f"agent/skin/{skin_index}.png", (0, 16, 16, 32)))

        agent_canvas = Image.new("RGBA", (self.tile_size, self.tile_size), (0, 0, 0, 0))
        for sprite_path, crop_box in layers:
            sprite = self._load_sprite(sprite_path)
            layer = sprite.crop(crop_box)
            agent_canvas.alpha_composite(self._resize_tile(layer))

        left = int(round(self._scale_world_coord(pixel_x) - self.tile_size / 2))
        top = int(round(self._scale_world_coord(pixel_y) - self.tile_size / 2))
        canvas.alpha_composite(agent_canvas, (left, top))


class SimulationReplay:
    """Replay a simulation directory into a video file."""

    def __init__(self, simulation_dir: Path | str, fps: int = 24, tile_size: int = 96, output_name: str = "replay_video.mp4"):
        self.simulation_dir = self._resolve_simulation_dir(Path(simulation_dir))
        self.fps = fps
        self.tile_size = tile_size
        self.output_name = output_name

        self.config = self._load_config(self.simulation_dir / "config.txt")
        self.map_nr = str(self.config.get("MAP_NR", ""))
        self.game_version = str(self.config.get("GAME_VERSION", "classic"))
        self.tick_rate = self._resolve_tick_rate()

        self.renderer = ReplayRenderer(tile_size=tile_size)
        self.positions_by_frame: Dict[int, Dict[str, AgentFrame]] = {}
        self.frame_seconds: Dict[int, float] = {}
        self.counters_by_frame: Dict[int, Dict[Tuple[int, int], str]] = {}
        self.item_types: Dict[str, str] = {}
        self.map_lines = self._load_map_lines()
        self.map_image = self._load_map_image()
        self._timeline_seconds: List[float] = []

    def _resolve_simulation_dir(self, path: Path) -> Path:
        if any(path.glob("positions_ai_rl_*.csv")):
            return path

        candidates = sorted(path.glob("simulation_*"))
        if candidates:
            return candidates[-1]

        csv_candidates = sorted(path.rglob("positions_ai_rl_*.csv"))
        if csv_candidates:
            return csv_candidates[0].parent

        raise FileNotFoundError(f"Could not locate a simulation directory under {path}")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if not config_path.exists():
            return config

        for line in config_path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            try:
                config[key] = ast.literal_eval(value)
            except Exception:
                config[key] = value
        return config

    def _resolve_tick_rate(self) -> float:
        for key in ("ENGINE_TICK_RATE", "TICK_RATE"):
            value = self.config.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)

        sample_file = next(iter(sorted(self.simulation_dir.glob("positions_ai_rl_*.csv"))), None)
        if sample_file and sample_file.exists():
            rows = self._read_csv_rows(sample_file)
            seconds = [float(row["second"]) for row in rows if row.get("second") not in (None, "")]
            if len(seconds) >= 2:
                deltas = [b - a for a, b in zip(seconds, seconds[1:]) if b > a]
                if deltas:
                    median_delta = sorted(deltas)[len(deltas) // 2]
                    if median_delta > 0:
                        return 1.0 / median_delta

        return 2.0

    def _read_csv_rows(self, path: Path) -> List[Dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _load_items(self) -> None:
        items_path = self.simulation_dir / "items.csv"
        if not items_path.exists():
            return

        for row in self._read_csv_rows(items_path):
            item_id = (row.get("item_id") or "").strip()
            item_type = (row.get("item_type") or "").strip()
            if item_id and item_type:
                self.item_types[item_id] = item_type

    def _load_positions(self) -> None:
        position_files = sorted(self.simulation_dir.glob("positions_ai_rl_*.csv"))
        for path in position_files:
            agent_id = path.stem.replace("positions_", "")
            for row in self._read_csv_rows(path):
                try:
                    frame = int(row.get("frame", "0"))
                except ValueError:
                    continue

                second = float(row.get("second", frame / self.tick_rate))
                self.frame_seconds.setdefault(frame, second)
                self.positions_by_frame.setdefault(frame, {})[agent_id] = AgentFrame(
                    pixel_x=float(row.get("pixel_x", 0) or 0),
                    pixel_y=float(row.get("pixel_y", 0) or 0),
                    tile_x=int(float(row.get("tile_x", 0) or 0)),
                    tile_y=int(float(row.get("tile_y", 0) or 0)),
                    item=(row.get("item", "") or "").strip(),
                    item_id=(row.get("item_id", "") or "").strip(),
                    score=int(float(row.get("score", 0) or 0)),
                )

        if not self.positions_by_frame:
            raise FileNotFoundError(f"No positions CSV files found in {self.simulation_dir}")

    def _load_counters(self) -> None:
        counters_path = self.simulation_dir / "counters.csv"
        if not counters_path.exists():
            return

        with counters_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return

            counter_columns = [name for name in reader.fieldnames if name.startswith("counter_") and name.endswith("_id")]
            counter_positions = [self._parse_counter_column(name) for name in counter_columns]

            for row in reader:
                try:
                    frame = int(row.get("frame", "0"))
                except ValueError:
                    continue

                counters: Dict[Tuple[int, int], str] = {}
                for column_name, position in zip(counter_columns, counter_positions):
                    item_id = (row.get(column_name, "") or "").strip()
                    if item_id:
                        counters[position] = item_id
                self.counters_by_frame[frame] = counters

    def _parse_counter_column(self, column_name: str) -> Tuple[int, int]:
        match = re.match(r"counter_(\d+)_(\d+)_id", column_name)
        if not match:
            return 0, 0
        return int(match.group(1)) - 1, int(match.group(2)) - 1

    def _load_map_lines(self) -> List[str]:
        if not self.map_nr:
            return []

        candidates = [
            MAPS_ROOT / "maps_txt" / f"{self.map_nr}.txt",
            MAPS_ROOT / f"{self.map_nr}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return [line.rstrip("\n") for line in candidate.read_text(encoding="utf-8").splitlines()]
        return []

    def _load_map_image(self) -> Optional[Image.Image]:
        if not self.map_nr:
            return None

        candidates = [
            MAPS_ROOT / "maps_png" / f"{self.map_nr}.png",
            MAPS_ROOT / f"{self.map_nr}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return Image.open(candidate).convert("RGBA")
        return None

    def _load_timeline(self) -> None:
        self._timeline_seconds = [self.frame_seconds[frame] for frame in self._sorted_frames()]

    def _sorted_frames(self) -> List[int]:
        return sorted(self.positions_by_frame.keys())

    def load(self) -> None:
        self._load_items()
        self._load_positions()
        self._load_counters()
        self._load_timeline()

    def _frame_snapshot(self, frame: int) -> Snapshot:
        return Snapshot(
            frame=frame,
            second=self.frame_seconds.get(frame, frame / self.tick_rate),
            agents=self.positions_by_frame.get(frame, {}),
            counters=self.counters_by_frame.get(frame, {}),
        )

    def _interpolate_agent(self, lower: AgentFrame, upper: AgentFrame, alpha: float) -> Tuple[float, float]:
        pixel_x = lower.pixel_x + (upper.pixel_x - lower.pixel_x) * alpha
        pixel_y = lower.pixel_y + (upper.pixel_y - lower.pixel_y) * alpha
        return pixel_x, pixel_y

    def _prepare_video_frame(self, snapshot: Snapshot, next_snapshot: Optional[Snapshot], alpha: float) -> np.ndarray:
        board = self.renderer.render_map(self.map_lines, self.map_image)

        for (x, y), item_id in snapshot.counters.items():
            item_type = self.item_types.get(item_id, "")
            if item_type:
                self.renderer.render_counter_item(board, x, y, item_type)

        for agent_id in sorted(snapshot.agents.keys()):
            lower = snapshot.agents[agent_id]
            upper = next_snapshot.agents.get(agent_id, lower) if next_snapshot else lower
            pixel_x, pixel_y = self._interpolate_agent(lower, upper, alpha)
            self.renderer.render_agent(board, agent_id, pixel_x, pixel_y, lower.item)

        frame = cv2.cvtColor(np.array(board), cv2.COLOR_RGBA2BGR)
        self._draw_overlay(frame, snapshot, alpha)
        return frame

    def _draw_overlay(self, frame: np.ndarray, snapshot: Snapshot, alpha: float) -> None:
        lines = [
            f"Map: {self.map_nr}",
            f"Frame: {snapshot.frame}",
            f"Second: {snapshot.second:.2f}",
            f"Mode: {self.game_version}",
        ]

        y = 24
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            y += 22

        if alpha > 0:
            line = f"interp: {alpha:.2f}"
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

    def _build_hud_game(self, snapshot: Snapshot) -> Any:
        game_objects: Dict[str, SimpleNamespace] = {}
        for agent_id, agent_frame in snapshot.agents.items():
            game_objects[agent_id] = SimpleNamespace(score=agent_frame.score)
        return SimpleNamespace(gameObjects=game_objects)

    def _iter_output_times(self, start_second: float, end_second: float) -> Iterable[float]:
        step = 1.0 / float(self.fps)
        current = start_second
        while current <= end_second + 1e-9:
            yield current
            current += step

    def replay(self) -> Path:
        self.load()

        frames = self._sorted_frames()
        if not frames:
            raise ValueError("No replay frames found")

        output_path = self.simulation_dir / self.output_name
        recorder = VideoRecorder(output_path, fps=self.fps)

        seconds = self._timeline_seconds
        start_second = seconds[0]
        end_second = seconds[-1]

        for output_second in self._iter_output_times(start_second, end_second):
            right_index = bisect_right(seconds, output_second)
            if right_index <= 0:
                lower_frame = frames[0]
                upper_frame = frames[0]
                alpha = 0.0
            elif right_index >= len(frames):
                lower_frame = frames[-1]
                upper_frame = frames[-1]
                alpha = 0.0
            else:
                lower_frame = frames[right_index - 1]
                upper_frame = frames[right_index]
                lower_second = seconds[right_index - 1]
                upper_second = seconds[right_index]
                alpha = 0.0 if upper_second <= lower_second else max(0.0, min(1.0, (output_second - lower_second) / (upper_second - lower_second)))

            snapshot = self._frame_snapshot(lower_frame)
            next_snapshot = self._frame_snapshot(upper_frame) if upper_frame != lower_frame else None
            frame = self._prepare_video_frame(snapshot, next_snapshot, alpha)
            recorder.write_frame_with_hud(frame, self._build_hud_game(snapshot))

        recorder.stop()
        return output_path


def replay_from_directory(simulation_dir: Path | str, fps: int = 24, tile_size: int = 96, output_name: str = "replay_video.mp4") -> Path:
    """Rebuild a replay video from a simulation output directory."""
    return SimulationReplay(simulation_dir, fps=fps, tile_size=tile_size, output_name=output_name).replay()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a replay video from simulation CSV outputs.")
    parser.add_argument("simulation_dir", type=str, help="Path to the simulation directory")
    parser.add_argument("--fps", type=int, default=24, help="Replay video FPS")
    parser.add_argument("--tile_size", type=int, default=96, help="Rendered tile size in pixels")
    parser.add_argument("--output_name", type=str, default="replay_video.mp4", help="Output video filename")
    args = parser.parse_args()

    video_path = replay_from_directory(args.simulation_dir, fps=args.fps, tile_size=args.tile_size, output_name=args.output_name)
    print(f"Saved replay video to: {video_path}")