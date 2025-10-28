"""Mirror application for Terp Wizard projector setup."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
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

from .states import MirrorState, MirrorStateMachine


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
    roll_duration: float
    result_duration: float
    cooldown_duration: float
    dry_run: bool = False


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

    timers_cfg = data.get("timers", {})
    try:
        roll_duration = float(timers_cfg.get("rolling", 3.0))
        result_duration = float(timers_cfg.get("result", 5.0))
        cooldown_duration = float(timers_cfg.get("cooldown", 2.0))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise MirrorConfigError("Timer values must be numeric.") from exc

    for name, value in {
        "rolling": roll_duration,
        "result": result_duration,
        "cooldown": cooldown_duration,
    }.items():
        if value <= 0:
            raise MirrorConfigError(f"Timer '{name}' must be greater than zero.")

    return MirrorConfig(
        monitor_index=int(data["monitor_index"]),
        rotate_deg=rotate_deg,
        mirror=bool(data["mirror"]),
        camera_index=int(camera_cfg["index"]),
        target_fps=max(1, int(data["target_fps"])),
        roll_duration=roll_duration,
        result_duration=result_duration,
        cooldown_duration=cooldown_duration,
        dry_run=bool(data.get("dry_run", False)),
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


class SpinnerOverlay:
    """Simple spinner animation rendered as a rotating arc."""

    def __init__(self, radius: int = 80, stroke: int = 12):
        self.radius = radius
        self.stroke = stroke

    def draw(self, target: pygame.Surface, angle: float) -> None:
        size = (self.radius * 2 + self.stroke * 2, self.radius * 2 + self.stroke * 2)
        overlay = pygame.Surface(size, pygame.SRCALPHA)
        rect = overlay.get_rect()
        arc_rect = rect.inflate(-self.stroke, -self.stroke)
        start_angle = angle % (2 * math.pi)
        end_angle = start_angle + math.pi * 1.5
        pygame.draw.arc(overlay, (255, 215, 0), arc_rect, start_angle, end_angle, self.stroke)
        target.blit(overlay, overlay.get_rect(center=target.get_rect().center))


class ResultOverlay:
    """Placeholder prize/result card overlay."""

    def __init__(self) -> None:
        self.title_font = pygame.font.Font(None, 96)
        self.body_font = pygame.font.Font(None, 48)

    def draw(self, target: pygame.Surface, prize_text: Optional[str]) -> None:
        overlay = pygame.Surface(target.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))

        card_width = int(target.get_width() * 0.5)
        card_height = int(target.get_height() * 0.35)
        card_rect = pygame.Rect(0, 0, card_width, card_height)
        card_rect.center = target.get_rect().center

        pygame.draw.rect(overlay, (255, 255, 255, 235), card_rect, border_radius=20)
        pygame.draw.rect(overlay, (128, 0, 128, 255), card_rect, width=6, border_radius=20)

        title_surface = self.title_font.render("Spin Complete!", True, (40, 0, 70))
        prize_message = f"Prize: {prize_text}" if prize_text else "Your prize awaits..."
        body_surface = self.body_font.render(prize_message, True, (40, 0, 70))

        title_rect = title_surface.get_rect(center=(card_rect.centerx, card_rect.centery - 40))
        body_rect = body_surface.get_rect(center=(card_rect.centerx, card_rect.centery + 40))

        overlay.blit(title_surface, title_rect)
        overlay.blit(body_surface, body_rect)

        target.blit(overlay, (0, 0))


def _capture_phase(
    cap: Optional["cv2.VideoCapture"], config: MirrorConfig, screen_size: tuple[int, int]
) -> Optional[pygame.Surface]:
    if cap is None:
        return None

    ret, frame = cap.read()
    if not ret:
        return None

    if config.mirror:
        frame = cv2.flip(frame, 1)

    frame = _rotate_frame(frame, config.rotate_deg)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_surface = pygame.image.frombuffer(
        frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB"
    ).convert()
    if frame_surface.get_size() != screen_size:
        frame_surface = pygame.transform.smoothscale(frame_surface, screen_size)
    return frame_surface


def _update_phase(
    events: list[pygame.event.Event], state_machine: MirrorStateMachine
) -> bool:
    running = True
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.key == pygame.K_r:
                state_machine.trigger_roll()
            elif event.key == pygame.K_f:
                state_machine.force_result()
            elif event.key == pygame.K_g:
                state_machine.force_grand_prize()

    state_machine.update()
    return running


def _render_phase(
    screen: pygame.Surface,
    frame_surface: Optional[pygame.Surface],
    state_machine: MirrorStateMachine,
    spinner: SpinnerOverlay,
    result_overlay: ResultOverlay,
) -> None:
    if frame_surface is not None:
        screen.blit(frame_surface, (0, 0))
    else:
        screen.fill((0, 0, 0))

    if state_machine.state is MirrorState.ROLLING:
        spinner.draw(screen, state_machine.time_in_state() * 2 * math.pi)
    elif state_machine.state is MirrorState.RESULT:
        result_overlay.draw(screen, state_machine.current_prize)


def run_mirror(config: MirrorConfig, monitor_override: Optional[int] = None) -> None:
    """Run the mirror display loop using the supplied configuration."""

    monitor_index = monitor_override if monitor_override is not None else config.monitor_index
    camera_index = config.camera_index
    dry_run = config.dry_run

    cap: Optional["cv2.VideoCapture"] = None
    if not dry_run:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}.")

    pygame.init()
    pygame.display.set_caption("Terp Mirror")
    pygame.mouse.set_visible(False)
    screen_size = _resolve_display_size(monitor_index)
    screen = pygame.display.set_mode(screen_size, pygame.FULLSCREEN, display=monitor_index)
    clock = pygame.time.Clock()

    state_machine = MirrorStateMachine(
        roll_duration=config.roll_duration,
        result_duration=config.result_duration,
        cooldown_duration=config.cooldown_duration,
        dry_run=dry_run,
    )
    spinner = SpinnerOverlay()
    result_overlay = ResultOverlay()

    last_frame: Optional[pygame.Surface] = None

    try:
        running = True
        while running:
            events = pygame.event.get()
            running = _update_phase(events, state_machine)

            frame_surface = _capture_phase(cap, config, screen_size)
            if frame_surface is not None:
                last_frame = frame_surface

            _render_phase(screen, last_frame, state_machine, spinner, result_overlay)

            pygame.display.flip()
            clock.tick(config.target_fps)
    finally:
        if cap is not None:
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without attempting to access the camera (testing mode).",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        config = replace(config, dry_run=True)
    run_mirror(config, monitor_override=args.monitor)


if __name__ == "__main__":
    main()
