#!/usr/bin/env python3
"""Interactive calibration workflow for Terp Mirror."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "OpenCV (cv2) is required. Install it with 'pip install opencv-python' or "
        "'pip install -r requirements.txt'."
    ) from exc

import pygame

from terp_mirror.calibration import (
    CalibrationError,
    compute_calibration_mapping,
    save_calibration_mapping,
    map_point_to_screen,
)
from terp_mirror.detection import DetectionResult, WaveDetector
from terp_mirror.mirror_app import (
    MirrorConfig,
    MirrorConfigError,
    _capture_phase,
    _resolve_display_size,
    load_config,
)


ANCHOR_LAYOUTS = {
    2: [(0.25, 0.5), (0.75, 0.5)],
    3: [(0.25, 0.3), (0.75, 0.3), (0.5, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Terp Mirror overlays.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--points",
        type=int,
        default=4,
        choices=(2, 3, 4),
        help="Number of calibration points to record.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Location to store the calibration YAML (defaults to config settings).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="Override the monitor index when presenting the calibration UI.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run the calibration UI in a window instead of fullscreen.",
    )
    return parser.parse_args(argv)


def _resolve_output_path(config: MirrorConfig, override: Optional[Path]) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if config.calibration.file is not None:
        return config.calibration.file
    return (Path.cwd() / "calibration.yaml").resolve()


def _draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> pygame.Rect:
    rendered = font.render(text, True, color)
    rect = rendered.get_rect()
    rect.topleft = position
    surface.blit(rendered, rect)
    return rect


def _draw_anchor_markers(
    target: pygame.Surface,
    anchors: list[tuple[float, float]],
    active_index: int,
) -> tuple[int, int]:
    screen_w, screen_h = target.get_size()
    active_point = (screen_w // 2, screen_h // 2)
    for idx, anchor in enumerate(anchors):
        point = (int(anchor[0] * screen_w), int(anchor[1] * screen_h))
        color = (128, 128, 128)
        if idx < active_index:
            color = (0, 200, 120)
        elif idx == active_index:
            color = (255, 215, 0)
            active_point = point
        pygame.draw.circle(target, color, point, 18, 4)
        pygame.draw.circle(target, color, point, 4)
    return active_point


def _calibration_loop(
    config: MirrorConfig,
    screen: pygame.Surface,
    screen_size: tuple[int, int],
    anchors: list[tuple[float, float]],
    detector: WaveDetector,
    cap: "cv2.VideoCapture",
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], Optional[tuple[int, int]]]:
    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 64)
    status_font = pygame.font.Font(None, 40)

    captured: list[tuple[tuple[float, float], tuple[float, float]]] = []
    last_frame: Optional[pygame.Surface] = None
    last_detection: Optional[DetectionResult] = None
    last_frame_size: Optional[tuple[int, int]] = None
    message = "Align yourself with the highlighted marker and wave."
    message_time = time.monotonic()
    running = True

    while running and len(captured) < len(anchors):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if last_detection is None or last_detection.centroid is None:
                        message = "No detection centroid available; keep waving and try again."
                        message_time = time.monotonic()
                    elif last_frame_size is None:
                        message = "Waiting for a valid frame from the camera."
                        message_time = time.monotonic()
                    else:
                        anchor = anchors[len(captured)]
                        target_px = (
                            int(anchor[0] * screen_size[0]),
                            int(anchor[1] * screen_size[1]),
                        )
                        captured.append((last_detection.centroid, target_px))
                        message = f"Captured point {len(captured)} of {len(anchors)}."
                        message_time = time.monotonic()
                elif event.key == pygame.K_BACKSPACE and captured:
                    captured.pop()
                    message = "Removed the last captured point."
                    message_time = time.monotonic()

        frame_surface, detection_result, frame_dimensions = _capture_phase(
            cap, config, screen_size, detector
        )
        if frame_surface is not None:
            last_frame = frame_surface
        if detection_result is not None:
            last_detection = detection_result
        if frame_dimensions is not None:
            last_frame_size = frame_dimensions

        if last_frame is not None:
            screen.blit(last_frame, (0, 0))
        else:
            screen.fill((0, 0, 0))

        active_point = _draw_anchor_markers(screen, anchors, len(captured))

        if last_detection is not None and last_detection.centroid is not None and last_frame_size is not None:
            mapped_point = map_point_to_screen(
                last_detection.centroid, last_frame_size, screen_size, None
            )
            pygame.draw.circle(screen, (255, 0, 0), mapped_point, 12, 3)
            pygame.draw.line(screen, (255, 0, 0), mapped_point, active_point, 1)

        _draw_text(screen, title_font, "Terp Mirror Calibration", (40, 30))
        _draw_text(
            screen,
            status_font,
            f"Point {len(captured) + 1} of {len(anchors)}",
            (40, 110),
        )
        _draw_text(
            screen,
            status_font,
            "Press SPACE to capture, BACKSPACE to undo, ESC to cancel.",
            (40, screen_size[1] - 120),
        )

        if time.monotonic() - message_time < 4.0:
            _draw_text(screen, status_font, message, (40, screen_size[1] - 80), (255, 215, 0))

        pygame.display.flip()
        clock.tick(30)

    return captured, last_frame_size


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    try:
        config = load_config(args.config)
    except MirrorConfigError as exc:
        raise SystemExit(str(exc)) from exc

    monitor_index = args.monitor if args.monitor is not None else config.monitor_index
    screen_size = _resolve_display_size(monitor_index)
    output_path = _resolve_output_path(config, args.output)

    cap = cv2.VideoCapture(config.display_camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open camera index {config.display_camera_index}.")

    pygame.init()
    flags = 0 if args.windowed else pygame.FULLSCREEN
    pygame.display.set_caption("Terp Mirror Calibration")
    screen = pygame.display.set_mode(screen_size, flags, display=monitor_index)
    pygame.mouse.set_visible(True)

    detector = WaveDetector(config.detection)
    anchors = list(ANCHOR_LAYOUTS[args.points])

    try:
        captured_pairs, frame_size = _calibration_loop(
            config,
            screen,
            screen_size,
            anchors,
            detector,
            cap,
        )
    finally:
        cap.release()
        pygame.quit()

    if len(captured_pairs) != len(anchors) or frame_size is None:
        print("Calibration cancelled; no data saved.")
        return

    try:
        mapping = compute_calibration_mapping(captured_pairs, frame_size, screen_size)
    except CalibrationError as exc:
        raise SystemExit(f"Failed to compute calibration: {exc}") from exc

    save_calibration_mapping(output_path, mapping)
    print(f"Calibration saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])

