"""Mirror application for Terp Wizard projector setup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class MirrorConfigError(RuntimeError):
    """Raised when the mirror configuration is invalid."""


@dataclass(frozen=True)
class MirrorConfig:
    """Normalized mirror configuration values."""

    monitor_index: int
    rotate_deg: int
    mirror: bool
    camera_index: int
    target_fps: int


def load_config(config_path: Optional[Path] = None) -> MirrorConfig:
    """Load the mirror configuration from YAML and validate it."""

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

    rotate_deg = int(data["rotate_deg"])
    if rotate_deg not in {0, 90, -90, 180, -180}:
        raise MirrorConfigError("rotate_deg must be one of {0, 90, -90, 180, -180}.")

    return MirrorConfig(
        monitor_index=int(data["monitor_index"]),
        rotate_deg=rotate_deg,
        mirror=bool(data["mirror"]),
        camera_index=int(camera_cfg["index"]),
        target_fps=max(1, int(data["target_fps"])),
    )


def _rotate_frame(frame, rotate_deg: int):
    if rotate_deg == 0:
        return frame
    if rotate_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate_deg == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_deg in {180, -180}:
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise MirrorConfigError("rotate_deg must be one of {0, 90, -90, 180, -180}.")


def _resolve_display_size(monitor_index: int) -> tuple[int, int]:
    """Return the full resolution for the requested monitor."""

    pygame.display.init()
    # ``get_num_video_displays`` only exists on some SDL builds; fall back to the
    # cross-version ``get_num_displays`` attribute when needed.
    if hasattr(pygame.display, "get_num_video_displays"):
        available = pygame.display.get_num_video_displays()
    else:
        available = pygame.display.get_num_displays()
    if available == 0:
        raise MirrorConfigError("No video displays detected for fullscreen output.")

    if monitor_index < 0 or monitor_index >= available:
        raise MirrorConfigError(
            f"Monitor index {monitor_index} is out of range (0-{available - 1})."
        )

    desktop_sizes = pygame.display.get_desktop_sizes()
    if monitor_index >= len(desktop_sizes):
        # Fall back to the primary monitor size if pygame did not report all displays.
        monitor_index = 0
    return desktop_sizes[monitor_index]


def run_mirror(config: MirrorConfig, monitor_override: Optional[int] = None) -> None:
    """Run the mirror display loop using the supplied configuration."""

    monitor_index = monitor_override if monitor_override is not None else config.monitor_index
    camera_index = config.camera_index

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    pygame.init()
    pygame.display.set_caption("Terp Mirror")
    pygame.mouse.set_visible(False)
    screen_size = _resolve_display_size(monitor_index)
    screen = pygame.display.set_mode(screen_size, pygame.FULLSCREEN, display=monitor_index)
    clock = pygame.time.Clock()

    try:
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                continue

            if config.mirror:
                frame = cv2.flip(frame, 1)

            frame = _rotate_frame(frame, config.rotate_deg)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            frame_surface = pygame.image.frombuffer(
                frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB"
            ).convert()
            if frame_surface.get_size() != screen_size:
                frame_surface = pygame.transform.smoothscale(frame_surface, screen_size)

            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            clock.tick(config.target_fps)
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
