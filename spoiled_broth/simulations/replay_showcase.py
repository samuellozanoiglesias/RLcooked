"""
Showcase replay: clean, presentation-ready videos from simulation CSV outputs.

Builds on replay_simulation_video.py (same CSVs, same sprites, same map
rendering) but produces videos meant for public screens:

  - no debug text (Map/Frame/Second/Mode/interp) and no HUD bar by default
  - one colour per player (soft halo under the chef)
  - a badge above each chef showing which ability is slowed
    (boot + snail = walks slowly, knife + snail = cuts slowly)
  - a floating "+1" every time a salad is delivered
  - optional speed-up (e.g. --speed 12 turns a 180-s game into 15 s)
  - H.264 / yuv420p output through ffmpeg (falls back to OpenCV mp4v)
  - a sidecar <name>_timeline.csv with the team score at every video frame,
    so the final composition can draw its own large score counter

It does NOT modify the simulation pipeline. Run it on existing simulation
folders (no need to re-run experimental_simulation.py):

  # one game, whole game at x12
  python -m spoiled_broth.simulations.replay_showcase <simulation_dir> --speed 12

  # an excerpt: seconds 60-84 of the game at x2 (12 s of video)
  python -m spoiled_broth.simulations.replay_showcase <simulation_dir> --start 60 --end 84 --speed 2

  # pick, among all games under a folder, the one whose final score is
  # closest to a target (e.g. the human mean for that condition)
  python -m spoiled_broth.simulations.replay_showcase <parent_dir> --pick_target 7 --speed 12

Author: Samuel Lozano (showcase additions)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .replay_simulation_video import SimulationReplay, Snapshot


VERSION = "3 (output_dir por defecto + comprobaciones)"

# Folder where every showcase video is saved unless --output_dir says otherwise
DEFAULT_OUTPUT_DIR = Path("/home/samuel_lozano/RLcooked/simulaciones_presentacion")

# --------------------------------------------------------------------------- #
#  Palette (RGB). Keep in sync with the final video composition.               #
# --------------------------------------------------------------------------- #
PLAYER_COLORS = {
    "ai_rl_1": (77, 182, 255),   # light blue
    "ai_rl_2": (255, 166, 77),   # amber
}
DEFAULT_PLAYER_COLOR = (200, 200, 200)
POPUP_COLOR = (120, 230, 140)    # soft green "+1"


# --------------------------------------------------------------------------- #
#  12x12 pixel-art icons (drawn here so no extra sprite files are needed)      #
# --------------------------------------------------------------------------- #
_ICON_PALETTE = {
    "o": (90, 55, 30, 255),      # shell outline
    "S": (230, 150, 60, 255),    # shell
    "b": (215, 200, 160, 255),   # snail body
    "e": (40, 40, 40, 255),      # eye
    "B": (150, 95, 55, 255),     # boot
    "s": (70, 45, 30, 255),      # sole
    "h": (130, 80, 40, 255),     # knife handle
    "k": (220, 225, 235, 255),   # blade
}

SNAIL = [
    "..........e.",
    "....oooo..b.",
    "...oSSSSo.b.",
    "..oSSooSSob.",
    "..oSoSSoSob.",
    "..oSoSoSSob.",
    "..oSSoooSob.",
    "...oSSSSobb.",
    "bbbboooobbb.",
    ".bbbbbbbbb..",
    "............",
    "............",
]

BOOT = [
    "............",
    "...BBBB.....",
    "...BBBB.....",
    "...BBBB.....",
    "...BBBB.....",
    "...BBBB.....",
    "...BBBBBB...",
    "...BBBBBBBB.",
    "...BBBBBBBB.",
    "..ssssssssss",
    "............",
    "............",
]

KNIFE = [
    "............",
    "..........hh",
    ".........hh.",
    "........hh..",
    ".......kk...",
    "......kkk...",
    ".....kkk....",
    "....kkk.....",
    "...kkk......",
    "..kkk.......",
    ".kk.........",
    "............",
]


def _icon(rows: List[str]) -> Image.Image:
    img = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
    px = img.load()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in _ICON_PALETTE:
                px[x, y] = _ICON_PALETTE[ch]
    return img


@dataclass
class Popup:
    start_t: float          # video time (s) when it appears
    x: float                # canvas pixel coords
    y: float


class ShowcaseReplay(SimulationReplay):
    """SimulationReplay with presentation-oriented rendering."""

    def __init__(
        self,
        simulation_dir: Path | str,
        fps: int = 30,
        tile_size: int = 96,
        output_name: str = "showcase.mp4",
        speed: float = 1.0,
        show_hud: bool = False,
        show_badges: bool = True,
        show_halos: bool = True,
        show_popups: bool = True,
        popup_duration: float = 1.2,
        start: float = 0.0,
        end: Optional[float] = None,
        output_dir: Optional[Path | str] = None,
    ):
        super().__init__(simulation_dir, fps=fps, tile_size=tile_size, output_name=output_name)
        self.speed = float(speed)
        self.show_hud = show_hud
        self.show_badges = show_badges
        self.show_halos = show_halos
        self.show_popups = show_popups
        self.popup_duration = popup_duration
        self.start = float(start)          # game seconds from the start of the game
        self.end = end                     # game seconds from the start of the game (None = until the end)
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR

        self._base_map: Optional[Image.Image] = None
        self._badges: Dict[str, Optional[Image.Image]] = {}
        self._popups: List[Popup] = []

        self.walking_speeds = self._speed_dict("WALKING_SPEEDS")
        self.cutting_speeds = self._speed_dict("CUTTING_SPEEDS")

    # ------------------------------------------------------------------ #
    #  Config helpers                                                       #
    # ------------------------------------------------------------------ #
    def _speed_dict(self, key: str) -> Dict[str, float]:
        value = self.config.get(key)
        return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------ #
    #  Badges                                                               #
    # ------------------------------------------------------------------ #
    def _badge_for(self, agent_id: str) -> Optional[Image.Image]:
        if agent_id in self._badges:
            return self._badges[agent_id]

        task_icons = []
        if float(self.walking_speeds.get(agent_id, 1.0)) < 1.0:
            task_icons.append(_icon(BOOT))
        if float(self.cutting_speeds.get(agent_id, 1.0)) < 1.0:
            task_icons.append(_icon(KNIFE))

        if not task_icons:
            self._badges[agent_id] = None
            return None

        # [task icon(s)] + [snail], pixel-art scaled with NEAREST
        icons = task_icons + [_icon(SNAIL)]
        scale = max(2, int(round(self.tile_size * 0.42 / 12)))
        icon_px = 12 * scale
        pad = max(4, scale * 2)
        gap = max(2, scale)
        w = pad * 2 + len(icons) * icon_px + (len(icons) - 1) * gap
        h = pad * 2 + icon_px

        color = PLAYER_COLORS.get(agent_id, DEFAULT_PLAYER_COLOR)
        badge = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(badge)
        radius = h // 2
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius,
                               fill=(20, 20, 24, 225), outline=color + (255,),
                               width=max(2, scale))
        x = pad
        for icon in icons:
            badge.alpha_composite(icon.resize((icon_px, icon_px), Image.NEAREST), (x, pad))
            x += icon_px + gap

        self._badges[agent_id] = badge
        return badge

    # ------------------------------------------------------------------ #
    #  Rendering                                                            #
    # ------------------------------------------------------------------ #
    def _board(self) -> Image.Image:
        if self._base_map is None:
            self._base_map = self.renderer.render_map(self.map_lines, self.map_image)
        return self._base_map.copy()

    def _agent_center(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        s = self.tile_size / self.renderer.source_tile_size
        return pixel_x * s, pixel_y * s

    def _draw_halo(self, board: Image.Image, agent_id: str, cx: float, cy: float) -> None:
        color = PLAYER_COLORS.get(agent_id, DEFAULT_PLAYER_COLOR)
        t = self.tile_size
        layer = Image.new("RGBA", board.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # ellipse at the chef's feet + thin ring
        fx, fy = cx, cy + t * 0.36
        rx, ry = t * 0.42, t * 0.16
        d.ellipse((fx - rx, fy - ry, fx + rx, fy + ry), fill=color + (110,))
        d.ellipse((fx - rx, fy - ry, fx + rx, fy + ry), outline=color + (255,), width=max(2, t // 32))
        board.alpha_composite(layer)

    def _draw_badge(self, board: Image.Image, agent_id: str, cx: float, cy: float) -> None:
        badge = self._badge_for(agent_id)
        if badge is None:
            return
        left = int(round(cx - badge.width / 2))
        top = int(round(cy - self.tile_size * 0.5 - badge.height - self.tile_size * 0.04))
        # keep inside the canvas
        left = max(0, min(board.width - badge.width, left))
        top = max(0, top)
        board.alpha_composite(badge, (left, top))

    def _render(self, snapshot: Snapshot, next_snapshot: Optional[Snapshot], alpha: float,
                video_t: float) -> np.ndarray:
        board = self._board()

        for (x, y), item_id in snapshot.counters.items():
            item_type = self.item_types.get(item_id, "")
            if item_type:
                self.renderer.render_counter_item(board, x, y, item_type)

        centers: Dict[str, Tuple[float, float]] = {}
        for agent_id in sorted(snapshot.agents.keys()):
            lower = snapshot.agents[agent_id]
            upper = next_snapshot.agents.get(agent_id, lower) if next_snapshot else lower
            px, py = self._interpolate_agent(lower, upper, alpha)
            centers[agent_id] = self._agent_center(px, py)
            if self.show_halos:
                self._draw_halo(board, agent_id, *centers[agent_id])
            self.renderer.render_agent(board, agent_id, px, py, lower.item)

        if self.show_badges:
            for agent_id, (cx, cy) in centers.items():
                self._draw_badge(board, agent_id, cx, cy)

        frame = cv2.cvtColor(np.array(board), cv2.COLOR_RGBA2BGR)

        if self.show_popups:
            self._draw_popups(frame, video_t)
        return frame

    def _draw_popups(self, frame: np.ndarray, video_t: float) -> None:
        alive = []
        for p in self._popups:
            age = video_t - p.start_t
            if age < 0 or age > self.popup_duration:
                continue
            alive.append(p)
            k = age / self.popup_duration
            rise = self.tile_size * 0.9 * k
            opacity = 1.0 if k < 0.6 else max(0.0, 1.0 - (k - 0.6) / 0.4)
            text = "+1"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = self.tile_size / 48.0
            thick = max(2, int(self.tile_size / 24))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            x = int(p.x - tw / 2)
            y = int(p.y - self.tile_size * 1.15 - rise)
            overlay = frame.copy()
            cv2.putText(overlay, text, (x, y), font, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
            bgr = POPUP_COLOR[::-1]
            cv2.putText(overlay, text, (x, y), font, scale, bgr, thick, cv2.LINE_AA)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, dst=frame)
        self._popups = alive

    # ------------------------------------------------------------------ #
    #  Writers                                                              #
    # ------------------------------------------------------------------ #
    def _open_writer(self, path: Path, w: int, h: int):
        if shutil.which("ffmpeg") and self._ffmpeg_has_x264():
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(self.fps),
                "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            return ("ffmpeg", proc)
        print("ffmpeg not found: falling back to OpenCV mp4v (lower quality).")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        return ("cv2", writer)

    @staticmethod
    def _ffmpeg_has_x264() -> bool:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=30).stdout
            return "libx264" in out
        except Exception:
            return False

    @staticmethod
    def _write(writer, frame: np.ndarray) -> None:
        kind, obj = writer
        if kind == "ffmpeg":
            obj.stdin.write(frame.tobytes())
        else:
            obj.write(frame)

    @staticmethod
    def _close(writer) -> None:
        kind, obj = writer
        if kind == "ffmpeg":
            obj.stdin.close()
            if obj.wait() != 0:
                raise RuntimeError("ffmpeg failed while encoding the video (see the error above)")
        else:
            obj.release()

    # ------------------------------------------------------------------ #
    #  Main loop                                                            #
    # ------------------------------------------------------------------ #
    def replay(self) -> Path:
        from bisect import bisect_right

        self.load()
        frames = self._sorted_frames()
        if not frames:
            raise ValueError("No replay frames found")

        seconds = self._timeline_seconds
        game0 = seconds[0]
        start_s = min(seconds[-1], game0 + max(0.0, self.start))
        end_s = seconds[-1] if self.end is None else min(seconds[-1], game0 + float(self.end))
        if end_s <= start_s:
            raise ValueError(f"Empty window: start={self.start}, end={self.end}")
        game_step = self.speed / float(self.fps)          # game seconds per video frame
        n_out = int(np.floor((end_s - start_s) / game_step)) + 1

        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / self.output_name
        (out_dir / (output_path.stem + "_source.txt")).write_text(
            f"simulation_dir: {self.simulation_dir}\n"
            f"window_s: {self.start}-{self.end}\nspeed: {self.speed}\n", encoding="utf-8")
        timeline_path = output_path.with_name(output_path.stem + "_timeline.csv")

        writer = None
        prev_scores: Dict[str, int] = {}
        hud_recorder = None
        if self.show_hud:
            from .video_recorder import VideoRecorder
            hud_recorder = VideoRecorder(output_path, fps=self.fps)

        with timeline_path.open("w", newline="", encoding="utf-8") as tf:
            tw = csv.writer(tf)
            tw.writerow(["video_frame", "video_second", "game_second", "team_score", "p1_score", "p2_score"])

            for i in range(n_out):
                game_t = start_s + i * game_step
                video_t = i / float(self.fps)

                r = bisect_right(seconds, game_t)
                if r <= 0:
                    lo = hi = frames[0]; alpha = 0.0
                elif r >= len(frames):
                    lo = hi = frames[-1]; alpha = 0.0
                else:
                    lo, hi = frames[r - 1], frames[r]
                    s0, s1 = seconds[r - 1], seconds[r]
                    alpha = 0.0 if s1 <= s0 else max(0.0, min(1.0, (game_t - s0) / (s1 - s0)))

                snap = self._frame_snapshot(lo)
                nxt = self._frame_snapshot(hi) if hi != lo else None

                # deliveries -> popups (score increase of an agent)
                for agent_id, af in snap.agents.items():
                    old = prev_scores.get(agent_id)
                    if old is not None and af.score > old:
                        cx, cy = self._agent_center(af.pixel_x, af.pixel_y)
                        for _ in range(af.score - old):
                            self._popups.append(Popup(video_t, cx, cy))
                    prev_scores[agent_id] = af.score

                frame = self._render(snap, nxt, alpha, video_t)

                p1 = prev_scores.get("ai_rl_1", 0)
                p2 = prev_scores.get("ai_rl_2", 0)
                tw.writerow([i, f"{video_t:.4f}", f"{game_t - game0:.3f}", p1 + p2, p1, p2])

                if hud_recorder is not None:
                    hud_recorder.write_frame_with_hud(frame, self._build_hud_game(snap))
                    continue

                # even dimensions for yuv420p
                h, w = frame.shape[:2]
                if h % 2 or w % 2:
                    frame = cv2.copyMakeBorder(frame, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                if writer is None:
                    writer = self._open_writer(output_path, frame.shape[1], frame.shape[0])
                self._write(writer, frame)

        if hud_recorder is not None:
            hud_recorder.stop()
        if writer is not None:
            self._close(writer)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"The video was NOT written: {output_path}")
        print(f"Saved showcase video: {output_path}  ({n_out} frames, {n_out / self.fps:.1f} s at {self.fps} fps, x{self.speed:g})")
        print(f"Saved score timeline: {timeline_path}")
        print(f"Window: {start_s - game0:.1f}-{end_s - game0:.1f} s of the game")
        print(f"Team score at the end of the window: {sum(prev_scores.values())}  "
              f"(whole game: {final_team_score(self.simulation_dir)})")
        return output_path


# --------------------------------------------------------------------------- #
#  Picking a representative game                                               #
# --------------------------------------------------------------------------- #
def final_team_score(sim_dir: Path) -> Optional[int]:
    total, found = 0, False
    for path in sorted(sim_dir.glob("positions_ai_rl_*.csv")):
        last = None
        with path.open("r", newline="", encoding="utf-8") as fh:
            for last in csv.DictReader(fh):
                pass
        if last is not None:
            total += int(float(last.get("score", 0) or 0))
            found = True
    return total if found else None


def find_simulation_dirs(root: Path) -> List[Path]:
    return sorted({p.parent for p in Path(root).rglob("positions_ai_rl_1.csv")})


def pick_closest(root: Path, target: float) -> Path:
    scored = [(d, final_team_score(d)) for d in find_simulation_dirs(root)]
    scored = [(d, s) for d, s in scored if s is not None]
    if not scored:
        raise FileNotFoundError(f"No simulations found under {root}")
    scored.sort(key=lambda ds: (abs(ds[1] - target), str(ds[0])))
    print(f"{len(scored)} games found under {root}. Closest to {target}:")
    for d, s in scored[:5]:
        print(f"  score {s:>3}  {d}")
    return scored[0][0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Presentation-ready replay video from simulation CSVs.")
    ap.add_argument("simulation_dir", type=str,
                    help="Simulation folder, or a parent folder when using --pick_target")
    ap.add_argument("--speed", type=float, default=1.0, help="Speed-up factor (e.g. 12: 180 s -> 15 s)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tile_size", type=int, default=96)
    ap.add_argument("--output_name", type=str, default="showcase.mp4")
    ap.add_argument("--pick_target", type=float, default=None,
                    help="Search simulation_dir recursively and render the game whose final score is closest to this value")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Start of the excerpt, in game seconds from the start of the game")
    ap.add_argument("--end", type=float, default=None,
                    help="End of the excerpt, in game seconds (default: end of the game)")
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                    help=f"Folder for the video (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--hud", action="store_true", help="Keep the old score bar under the map")
    ap.add_argument("--no_badges", action="store_true")
    ap.add_argument("--no_halos", action="store_true")
    ap.add_argument("--no_popups", action="store_true")
    args = ap.parse_args()

    print(f"replay_showcase version {VERSION}")
    print(f"  script:     {Path(__file__).resolve()}")
    print(f"  output_dir: {Path(args.output_dir).expanduser().resolve()}")
    sim_dir = Path(args.simulation_dir)
    if args.pick_target is not None:
        sim_dir = pick_closest(sim_dir, args.pick_target)

    ShowcaseReplay(
        sim_dir, fps=args.fps, tile_size=args.tile_size, output_name=args.output_name,
        speed=args.speed, show_hud=args.hud, show_badges=not args.no_badges,
        show_halos=not args.no_halos, show_popups=not args.no_popups,
        start=args.start, end=args.end, output_dir=args.output_dir,
    ).replay()


if __name__ == "__main__":
    main()