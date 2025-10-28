"""Mirror application for Terp Wizard projector setup."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import pygame
import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class MirrorConfigError(RuntimeError):
    """Raised when the mirror configuration is invalid."""


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the mirror configuration from YAML.

    Args:
        config_path: Optional override for the configuration file path.

    Returns:
        Parsed configuration dictionary.
    """

    path = config_path or CONFIG_PATH
    if not path.exists():
        raise MirrorConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    required_keys = {"monitor_index", "rotate_deg", "mirror", "camera", "target_fps"}
    missing = required_keys - data.keys()
    if missing:
        raise MirrorConfigError(f"Missing configuration keys: {', '.join(sorted(missing))}")

    camera_cfg = data.get("camera")
    if not isinstance(camera_cfg, dict) or "index" not in camera_cfg:
        raise MirrorConfigError("Camera configuration must include an 'index'.")

    return data


def _rotate_frame(frame, rotate_deg: int):
    if rotate_deg == 0:
        return frame
    if rotate_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate_deg == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_deg == 180 or rotate_deg == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise MirrorConfigError(
        "rotate_deg must be one of {0, 90, -90, 180, -180}."
    )


def run_mirror(config: Dict[str, Any], monitor_override: Optional[int] = None) -> None:
    """Run the mirror display loop using the supplied configuration."""

    monitor_index = monitor_override if monitor_override is not None else config["monitor_index"]
    camera_index = config["camera"]["index"]
    mirror_enabled = bool(config["mirror"])
    rotate_deg = int(config["rotate_deg"])
    target_fps = int(config["target_fps"])

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    pygame.init()
    pygame.display.set_caption("Terp Mirror")
    clock = pygame.time.Clock()
    screen = None

    try:
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                continue

            if mirror_enabled:
                frame = cv2.flip(frame, 1)

            frame = _rotate_frame(frame, rotate_deg)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame_rgb.shape[:2]

            if screen is None:
                screen = pygame.display.set_mode(
                    (frame_width, frame_height),
                    pygame.FULLSCREEN,
                    display=monitor_index,
                )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            surface = pygame.image.frombuffer(
                frame_rgb.tobytes(), (frame_width, frame_height), "RGB"
            )
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(target_fps)
    finally:
        cap.release()
        pygame.quit()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Terp Mirror display.")
    parser.add_argument(
        "--monitor",
        type=int,
        help="Override the monitor index specified in the config file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to configuration file (default: {CONFIG_PATH}).",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_mirror(config, monitor_override=args.monitor)


if __name__ == "__main__":
    main()
