"""Wave detection using HSV thresholding and contour tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Deque, Optional

import cv2
import numpy as np
import pygame


@dataclass(frozen=True)
class DetectionROI:
    """Normalized rectangular region of interest."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class DetectionConfig:
    """Configuration for HSV thresholding and wave detection heuristics."""

    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    blur_kernel: int
    morph_kernel: int
    morph_iterations: int
    min_contour_area: float
    roi: DetectionROI
    buffer_duration: float
    min_wave_span: float
    min_wave_velocity: float
    cooldown: float
    ir_enabled: bool = True
    ir_buffer_duration: float = 0.35
    ir_score_threshold: float = 0.6
    ir_release_threshold: float = 0.4
    ir_debounce: float = 1.0


@dataclass
class DetectionResult:
    """Result from processing a single video frame."""

    wave_detected: bool
    centroid: Optional[tuple[int, int]] = None
    contour_area: float = 0.0
    ir_score: float = 0.0
    mode: str = "color"
    signal_strength: float = 0.0


def _ensure_odd(value: int) -> int:
    return int(value) | 1


class WaveDetector:
    """Detect deliberate waving motions using contour tracking."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._samples: Deque[tuple[float, float]] = deque()
        self._last_wave_time: float = 0.0
        self._debug_enabled = False
        self._ir_samples: Deque[tuple[float, float]] = deque()
        self._ir_active = False
        self._ir_last_trigger: float = float("-inf")
        self._paused = False

    @property
    def debug_enabled(self) -> bool:
        return self._debug_enabled

    @property
    def paused(self) -> bool:
        return self._paused

    def toggle_debug(self) -> None:
        self._debug_enabled = not self._debug_enabled
        print(f"[detector] Debug overlay {'enabled' if self._debug_enabled else 'disabled'}")

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        status = "paused" if self._paused else "resumed"
        print(f"[detector] Detection {status} by operator hotkey")

    def handle_key(self, key: int) -> None:
        """Handle key presses for tuning HSV thresholds at runtime."""

        adjustments = {
            pygame.K_1: ("lower", 0, -1),
            pygame.K_2: ("lower", 0, 1),
            pygame.K_3: ("upper", 0, -1),
            pygame.K_4: ("upper", 0, 1),
            pygame.K_5: ("lower", 1, -5),
            pygame.K_6: ("lower", 1, 5),
            pygame.K_7: ("upper", 1, -5),
            pygame.K_8: ("upper", 1, 5),
            pygame.K_9: ("lower", 2, -5),
            pygame.K_0: ("lower", 2, 5),
            pygame.K_MINUS: ("upper", 2, -5),
            pygame.K_EQUALS: ("upper", 2, 5),
        }

        if key == pygame.K_TAB:
            self.toggle_debug()
            return

        if key not in adjustments:
            return

        bound, channel, delta = adjustments[key]
        self._adjust_threshold(bound, channel, delta)

    def _adjust_threshold(self, bound: str, channel: int, delta: int) -> None:
        lower = list(self.config.hsv_lower)
        upper = list(self.config.hsv_upper)

        if bound == "lower":
            lower[channel] = self._clamp_channel(channel, lower[channel] + delta)
        else:
            upper[channel] = self._clamp_channel(channel, upper[channel] + delta)

        if channel == 0:
            lower[channel] = max(0, min(lower[channel], upper[channel]))
            upper[channel] = max(lower[channel], min(179, upper[channel]))
        else:
            lower[channel] = max(0, min(lower[channel], upper[channel]))
            upper[channel] = max(lower[channel], min(255, upper[channel]))

        self.config.hsv_lower = tuple(lower)
        self.config.hsv_upper = tuple(upper)
        print(f"[detector] HSV lower={self.config.hsv_lower} upper={self.config.hsv_upper}")

    @staticmethod
    def _clamp_channel(channel: int, value: int) -> int:
        if channel == 0:
            return max(0, min(179, value))
        return max(0, min(255, value))

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process ``frame`` and return the detection outcome."""

        frame_h, frame_w = frame.shape[:2]

        if self._paused:
            return DetectionResult(
                wave_detected=False,
                centroid=None,
                contour_area=0.0,
                ir_score=0.0,
                mode="paused",
                signal_strength=0.0,
            )
        roi_x0 = int(self.config.roi.x * frame_w)
        roi_y0 = int(self.config.roi.y * frame_h)
        roi_x1 = int(min(frame_w, roi_x0 + self.config.roi.width * frame_w))
        roi_y1 = int(min(frame_h, roi_y0 + self.config.roi.height * frame_h))

        roi_x1 = max(roi_x1, roi_x0 + 1)
        roi_y1 = max(roi_y1, roi_y0 + 1)

        roi_frame = frame[roi_y0:roi_y1, roi_x0:roi_x1]
        blurred = cv2.GaussianBlur(roi_frame, (_ensure_odd(self.config.blur_kernel),) * 2, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.config.hsv_lower), np.array(self.config.hsv_upper))

        kernel = np.ones((_ensure_odd(self.config.morph_kernel),) * 2, np.uint8)
        if self.config.morph_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.morph_iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour: Optional[np.ndarray] = None
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > best_area:
                best_area = area
                best_contour = contour

        centroid: Optional[tuple[int, int]] = None
        wave_detected = False

        if best_contour is not None and best_area >= self.config.min_contour_area:
            moments = cv2.moments(best_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                centroid = (roi_x0 + cx, roi_y0 + cy)
                roi_width = max(1, roi_x1 - roi_x0)
                wave_detected = self._update_motion_buffer(cx / roi_width)
            else:
                self._samples.clear()
        else:
            self._samples.clear()

        ir_score = self._update_ir_intensity(roi_frame) if self.config.ir_enabled else 0.0
        mode = "color"

        color_wave_detected = wave_detected
        ir_triggered = False

        if self.config.ir_enabled:
            now = time.monotonic()
            if ir_score >= self.config.ir_score_threshold:
                if not self._ir_active:
                    self._ir_active = True
                if now - self._ir_last_trigger >= self.config.ir_debounce:
                    ir_triggered = True
                    self._ir_last_trigger = now
            elif ir_score <= self.config.ir_release_threshold:
                self._ir_active = False

            if self._ir_active:
                mode = "ir"
                wave_detected = ir_triggered
                centroid = None
            else:
                mode = "color"
                wave_detected = color_wave_detected
        else:
            wave_detected = color_wave_detected

        signal_strength = ir_score if mode == "ir" else float(best_area)

        if self._debug_enabled:
            self._draw_debug(
                frame,
                (roi_x0, roi_y0, roi_x1, roi_y1),
                best_contour,
                centroid,
                best_area,
                mask,
                ir_score,
                mode,
            )

        return DetectionResult(
            wave_detected=wave_detected,
            centroid=centroid,
            contour_area=best_area,
            ir_score=ir_score,
            mode=mode,
            signal_strength=signal_strength,
        )

    def _update_ir_intensity(self, roi_frame: np.ndarray) -> float:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        intensity = float(np.mean(gray)) / 255.0
        timestamp = time.monotonic()
        self._ir_samples.append((timestamp, intensity))
        cutoff = timestamp - self.config.ir_buffer_duration
        while self._ir_samples and self._ir_samples[0][0] < cutoff:
            self._ir_samples.popleft()

        if not self._ir_samples:
            return intensity

        weighted = [value for _, value in self._ir_samples]
        return float(sum(weighted) / len(weighted))

    def _update_motion_buffer(self, normalized_x: float) -> bool:
        now = time.monotonic()
        self._samples.append((now, float(normalized_x)))

        buffer_cutoff = now - self.config.buffer_duration
        while self._samples and self._samples[0][0] < buffer_cutoff:
            self._samples.popleft()

        if len(self._samples) < 3:
            return False

        samples_list = list(self._samples)
        xs = [sample[1] for sample in samples_list]

        span = max(xs) - min(xs)
        if span < self.config.min_wave_span:
            return False

        velocities = []
        directions: list[int] = []
        for (t0, x0), (t1, x1) in zip(samples_list, samples_list[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            dx = x1 - x0
            velocities.append(abs(dx) / dt)
            if abs(dx) > 1e-2:
                directions.append(1 if dx > 0 else -1)

        if not velocities or max(velocities) < self.config.min_wave_velocity:
            return False

        direction_changes = sum(1 for a, b in zip(directions, directions[1:]) if a != b)
        if direction_changes == 0:
            return False

        now = time.monotonic()
        if now - self._last_wave_time < self.config.cooldown:
            return False

        self._last_wave_time = now
        self._samples.clear()
        return True

    def _draw_debug(
        self,
        frame: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
        contour: Optional[np.ndarray],
        centroid: Optional[tuple[int, int]],
        area: float,
        mask: np.ndarray,
        ir_score: float,
        mode: str,
    ) -> None:
        roi_x0, roi_y0, roi_x1, roi_y1 = roi_bounds
        cv2.rectangle(frame, (roi_x0, roi_y0), (roi_x1, roi_y1), (255, 255, 255), 2)

        if contour is not None:
            contour_offset = contour + np.array([[roi_x0, roi_y0]])
            cv2.drawContours(frame, [contour_offset], -1, (0, 255, 255), 2)

        if centroid is not None:
            cv2.circle(frame, centroid, 8, (0, 0, 255), -1)

        hsv_text = f"H[{self.config.hsv_lower[0]}-{self.config.hsv_upper[0]}] " \
            f"S[{self.config.hsv_lower[1]}-{self.config.hsv_upper[1]}] " \
            f"V[{self.config.hsv_lower[2]}-{self.config.hsv_upper[2]}]"
        cv2.putText(
            frame,
            hsv_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Area: {int(area)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"IR score: {ir_score:.2f} ({mode})",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if mask.size == 0:
            return

        mask_small = cv2.resize(mask, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)
        if mask_small.size == 0:
            return

        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        h, w = mask_bgr.shape[:2]
        y_end = min(frame.shape[0], roi_y0 + h)
        x_start = max(0, roi_x1 - w)
        x_end = min(frame.shape[1], roi_x1)
        height = max(0, y_end - roi_y0)
        width = max(0, x_end - x_start)
        if height > 0 and width > 0:
            overlay_region = frame[roi_y0:y_end, x_start:x_end]
            mask_crop = mask_bgr[:height, :width]
            if overlay_region.shape == mask_crop.shape:
                overlay_region[:] = mask_crop

