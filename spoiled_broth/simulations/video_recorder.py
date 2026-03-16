"""
Video recording utilities for simulation runs.

Author: Samuel Lozano
"""

import os
from pathlib import Path
from typing import List, Any

import numpy as np
import cv2


class VideoRecorder:
    """Handles video recording with HUD overlay."""
    
    def __init__(self, output_path: Path, fps: int = 24):
        self.output_path = str(output_path)
        self.fps = fps
        self.writer = None
        self.size = None
        self.hud_height = None  # Will be calculated based on first frame
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
    
    def start(self, frame: np.ndarray):
        """Initialize video writer with first frame."""
        src = self._prepare_frame(frame)
        h, w = src.shape[:2]
        
        # Calculate HUD height - must match _add_hud_to_frame calculation
        main_font = cv2.FONT_HERSHEY_SIMPLEX
        player_font = cv2.FONT_HERSHEY_SIMPLEX
        main_scale = 1.2
        player_scale = 0.9
        main_thickness = 2
        player_thickness = 2
        
        main_text_size = cv2.getTextSize('Score: 99', main_font, main_scale, main_thickness)[0]
        player_text_size = cv2.getTextSize('P1: 99', player_font, player_scale, player_thickness)[0]
        self.hud_height = max(100, main_text_size[1] + (player_text_size[1] * 2) + 60)  # Must match _add_hud_to_frame
        
        # Video dimensions include game area + HUD area
        video_width = w
        video_height = h + self.hud_height
        
        # Try different codecs in order of preference
        # mp4v: widely compatible, good compression
        # avc1/H264: high quality but requires proper FFmpeg build
        # XVID/MJPG: fallback options
        fourcc_candidates = ['mp4v', 'avc1', 'H264', 'X264', 'XVID', 'MJPG']
        
        for code in fourcc_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*code)
                self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (video_width, video_height))
                
                if self.writer and self.writer.isOpened():
                    print(f"  VideoWriter initialized with codec: {code}")
                    self.size = (video_width, video_height)
                    break
                elif self.writer:
                    self.writer.release()
                    self.writer = None
            except Exception as e:
                if self.writer:
                    self.writer.release()
                    self.writer = None
                continue
        
        if self.writer is None:
            print(f"Warning: could not open VideoWriter for {self.output_path}")
    
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for video encoding."""
        src = frame
        if src.dtype != np.uint8:
            src = (np.clip(src, 0.0, 1.0) * 255).astype(np.uint8)
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        return src
    
    def write_frame_with_hud(self, frame: np.ndarray, game: Any):
        """Write frame with HUD overlay showing scores."""
        if self.writer is None:
            self.start(frame)
        
        if self.writer is None:
            return
        
        try:
            src = self._prepare_frame(frame)
            
            # Calculate scores
            coop_score = 0
            scores = {}
            try:
                for agent_id, obj in game.gameObjects.items():
                    if agent_id.startswith('ai_rl_') and obj is not None:
                        score = int(getattr(obj, 'score', 0) or 0)
                        coop_score += score
                        scores[agent_id] = score
            except Exception:
                pass
            
            # Create enhanced HUD with main score and individual scores
            main_score = f"Score: {coop_score}"
            p1_score = f"P1: {scores.get('ai_rl_1', 0)}"
            p2_score = f"P2: {scores.get('ai_rl_2', 0)}"
            
            # Add HUD to frame (this preserves the original game dimensions)
            frame_with_hud = self._add_hud_to_frame(src, main_score, p1_score, p2_score)
            
            # Validate frame size matches video writer expectations
            if self.size and frame_with_hud.shape[:2][::-1] != self.size:
                print(f"Warning: Frame size mismatch! Expected {self.size}, got {frame_with_hud.shape[:2][::-1]}")
                return
            
            # Write the frame
            self.writer.write(frame_with_hud)
            
        except Exception as e:
            print(f"Warning: failed to write video frame: {e}")
    
    def _add_hud_to_frame(self, frame: np.ndarray, main_score: str, p1_score: str, p2_score: str) -> np.ndarray:
        """Add clean HUD overlay to frame with readable fonts."""
        h, w = frame.shape[:2]
        
        # Larger, more readable fonts
        main_font = cv2.FONT_HERSHEY_SIMPLEX
        player_font = cv2.FONT_HERSHEY_SIMPLEX
        
        main_scale = 1.2   # Larger main score
        player_scale = 0.9  # Larger player scores
        main_thickness = 2  # Bold main score
        player_thickness = 2  # Bold player scores
        
        # Calculate text dimensions
        main_text_size = cv2.getTextSize(main_score, main_font, main_scale, main_thickness)[0]
        player_text_size = cv2.getTextSize('P1: 99', player_font, player_scale, player_thickness)[0]
        
        # Use the pre-calculated hud_height from start() to ensure consistency
        pad_height = self.hud_height if self.hud_height else max(100, main_text_size[1] + (player_text_size[1] * 2) + 60)
        
        # Create canvas with HUD area - preserve original game frame exactly
        canvas = np.zeros((h + pad_height, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame
        
        # Draw clean black background for HUD area
        cv2.rectangle(canvas, (0, h), (w, h + pad_height), (0, 0, 0), -1)
        # Add subtle separator line
        cv2.line(canvas, (0, h), (w, h), (100, 100, 100), 2)
        
        # Position main score at top of HUD area (centered, just below video)
        main_x = (w - main_text_size[0]) // 2
        main_y = h + 25 + main_text_size[1]  # More space from video
        
        # Position player scores stacked vertically below main score
        # P1 aligned to left, P2 aligned to right
        p1_x = 30  # Left alignment with more padding
        p1_y = main_y + player_text_size[1] + 15  # Below main score
        p2_x = w - player_text_size[0] - 30  # Right alignment with more padding
        p2_y = p1_y + player_text_size[1] + 10  # Below P1
        
        # Soft color scheme (BGR format for OpenCV)
        main_color = (100, 200, 100)    # Soft green (BGR: B=100, G=200, R=100)
        p1_color = (200, 150, 100)      # Soft blue (BGR: B=200, G=150, R=100)
        p2_color = (100, 100, 200)      # Soft red (BGR: B=100, G=100, R=200)
        shadow_color = (0, 0, 0)        # Black shadow
        
        # Draw main score with minimal shadow
        cv2.putText(canvas, main_score, (main_x + 1, main_y + 1), main_font, main_scale, 
                   shadow_color, main_thickness, cv2.LINE_AA)
        cv2.putText(canvas, main_score, (main_x, main_y), main_font, main_scale, 
                   main_color, main_thickness, cv2.LINE_AA)
        
        # Draw player scores with minimal shadows
        # P1 score (left corner)
        cv2.putText(canvas, p1_score, (p1_x + 1, p1_y + 1), player_font, player_scale, 
                   shadow_color, player_thickness, cv2.LINE_AA)
        cv2.putText(canvas, p1_score, (p1_x, p1_y), player_font, player_scale, 
                   p1_color, player_thickness, cv2.LINE_AA)
        
        # P2 score (right corner)
        cv2.putText(canvas, p2_score, (p2_x + 1, p2_y + 1), player_font, player_scale, 
                   shadow_color, player_thickness, cv2.LINE_AA)
        cv2.putText(canvas, p2_score, (p2_x, p2_y), player_font, player_scale, 
                   p2_color, player_thickness, cv2.LINE_AA)
        
        return canvas
    
    def stop(self):
        """Stop video recording and release resources."""
        if self.writer:
            self.writer.release()
            self.writer = None