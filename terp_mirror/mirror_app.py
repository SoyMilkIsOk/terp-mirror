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

from .detection import DetectionConfig, DetectionROI, DetectionResult, WaveDetector
from .calibration import (
    CalibrationConfig,
    CalibrationError,
    CalibrationMapping,
    load_calibration_mapping,
    map_point_to_screen,
)
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
    detection: DetectionConfig
    calibration: CalibrationConfig
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

    detection_cfg = _parse_detection_config(data.get("detection", {}))
    calibration_cfg = _parse_calibration_config(data.get("calibration", {}), path.parent)

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
        detection=detection_cfg,
        calibration=calibration_cfg,
    )


def _parse_detection_config(data: dict) -> DetectionConfig:
    def _require_iterable(name: str, value) -> tuple[int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise MirrorConfigError(f"Detection.{name} must be a sequence of three integers.")
        try:
            return tuple(int(v) for v in value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise MirrorConfigError(f"Detection.{name} must be numeric.") from exc

    hsv_lower = _require_iterable("hsv_min", data.get("hsv_min", (0, 120, 120)))
    hsv_upper = _require_iterable("hsv_max", data.get("hsv_max", (40, 255, 255)))

    roi_cfg = data.get(
        "roi",
        {
            "x": 0.2,
            "y": 0.1,
            "width": 0.6,
            "height": 0.8,
        },
    )
    if not isinstance(roi_cfg, dict):
        raise MirrorConfigError("Detection.roi must be a mapping with x/y/width/height.")

    try:
        roi = DetectionROI(
            x=float(roi_cfg.get("x", 0.0)),
            y=float(roi_cfg.get("y", 0.0)),
            width=float(roi_cfg.get("width", 1.0)),
            height=float(roi_cfg.get("height", 1.0)),
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise MirrorConfigError("Detection.roi values must be numeric.") from exc

    if not (0.0 <= roi.x <= 1.0 and 0.0 <= roi.y <= 1.0):
        raise MirrorConfigError("Detection.roi x and y must be within [0, 1].")
    if not (0.0 < roi.width <= 1.0 and 0.0 < roi.height <= 1.0):
        raise MirrorConfigError("Detection.roi width and height must be within (0, 1].")
    if roi.x + roi.width > 1.0 or roi.y + roi.height > 1.0:
        raise MirrorConfigError("Detection.roi must remain within the frame bounds.")

    try:
        blur_kernel = int(data.get("blur_kernel", 11))
        morph_kernel = int(data.get("morph_kernel", 5))
        morph_iterations = int(data.get("morph_iterations", 1))
        min_contour_area = float(data.get("min_contour_area", 1500.0))
        buffer_duration = float(data.get("buffer_duration", 0.6))
        min_wave_span = float(data.get("min_wave_span", 0.2))
        min_wave_velocity = float(data.get("min_wave_velocity", 0.4))
        cooldown = float(data.get("cooldown", 1.5))
        ir_target_frequency = float(data.get("ir_target_frequency", 200.0))
        ir_buffer_duration = float(data.get("ir_buffer_duration", 0.35))
        ir_score_threshold = float(data.get("ir_score_threshold", 0.6))
        ir_release_threshold = float(data.get("ir_release_threshold", 0.4))
        ir_debounce = float(data.get("ir_debounce", 1.0))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise MirrorConfigError("Detection numeric parameters must be valid numbers.") from exc

    if min_contour_area <= 0:
        raise MirrorConfigError("Detection.min_contour_area must be positive.")
    if buffer_duration <= 0:
        raise MirrorConfigError("Detection.buffer_duration must be positive.")
    if min_wave_span <= 0:
        raise MirrorConfigError("Detection.min_wave_span must be positive.")
    if min_wave_velocity <= 0:
        raise MirrorConfigError("Detection.min_wave_velocity must be positive.")
    if cooldown < 0:
        raise MirrorConfigError("Detection.cooldown must be zero or greater.")
    if ir_target_frequency <= 0:
        raise MirrorConfigError("Detection.ir_target_frequency must be positive.")
    if ir_buffer_duration <= 0:
        raise MirrorConfigError("Detection.ir_buffer_duration must be positive.")
    if not 0 <= ir_score_threshold <= 1:
        raise MirrorConfigError("Detection.ir_score_threshold must be within [0, 1].")
    if not 0 <= ir_release_threshold <= 1:
        raise MirrorConfigError("Detection.ir_release_threshold must be within [0, 1].")
    if ir_release_threshold > ir_score_threshold:
        raise MirrorConfigError("Detection.ir_release_threshold must be <= ir_score_threshold.")
    if ir_debounce < 0:
        raise MirrorConfigError("Detection.ir_debounce must be zero or greater.")

    return DetectionConfig(
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        blur_kernel=max(1, blur_kernel),
        morph_kernel=max(1, morph_kernel),
        morph_iterations=max(0, morph_iterations),
        min_contour_area=min_contour_area,
        roi=roi,
        buffer_duration=buffer_duration,
        min_wave_span=min_wave_span,
        min_wave_velocity=min_wave_velocity,
        cooldown=cooldown,
        ir_enabled=bool(data.get("ir_enabled", True)),
        ir_target_frequency=ir_target_frequency,
        ir_buffer_duration=ir_buffer_duration,
        ir_score_threshold=ir_score_threshold,
        ir_release_threshold=ir_release_threshold,
        ir_debounce=ir_debounce,
    )


def _parse_calibration_config(data: dict, base_dir: Path) -> CalibrationConfig:
    if not isinstance(data, dict):
        raise MirrorConfigError("Calibration configuration must be a mapping.")

    enabled = bool(data.get("enabled", False))
    file_value = data.get("file") or "calibration.yaml"
    try:
        raw_path = Path(file_value).expanduser()
    except TypeError as exc:  # pragma: no cover - defensive
        raise MirrorConfigError("Calibration.file must be a valid path string.") from exc

    file_path = (base_dir / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()

    return CalibrationConfig(
        enabled=enabled,
        file=file_path,
        apply_prompt=bool(data.get("apply_prompt", True)),
        apply_spinner=bool(data.get("apply_spinner", True)),
        apply_result=bool(data.get("apply_result", True)),
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

    def draw(self, target: pygame.Surface, angle: float, center: tuple[int, int]) -> None:
        size = (self.radius * 2 + self.stroke * 2, self.radius * 2 + self.stroke * 2)
        overlay = pygame.Surface(size, pygame.SRCALPHA)
        rect = overlay.get_rect()
        arc_rect = rect.inflate(-self.stroke, -self.stroke)
        start_angle = angle % (2 * math.pi)
        end_angle = start_angle + math.pi * 1.5
        pygame.draw.arc(overlay, (255, 215, 0), arc_rect, start_angle, end_angle, self.stroke)
        target.blit(overlay, overlay.get_rect(center=center))


class ResultOverlay:
    """Placeholder prize/result card overlay."""

    def __init__(self) -> None:
        self.title_font = pygame.font.Font(None, 96)
        self.body_font = pygame.font.Font(None, 48)

    def draw(self, target: pygame.Surface, prize_text: Optional[str], center: tuple[int, int]) -> None:
        overlay = pygame.Surface(target.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))

        card_width = int(target.get_width() * 0.5)
        card_height = int(target.get_height() * 0.35)
        card_rect = pygame.Rect(0, 0, card_width, card_height)
        card_rect.center = center

        # Keep the card within the screen bounds.
        card_rect.clamp_ip(target.get_rect())

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


class PromptOverlay:
    """Display guidance text positioned using calibration data."""

    def __init__(self) -> None:
        self.title_font = pygame.font.Font(None, 72)
        self.body_font = pygame.font.Font(None, 48)

    def draw(
        self,
        target: pygame.Surface,
        message: str,
        center: tuple[int, int],
        subtext: Optional[str] = None,
    ) -> None:
        if not message:
            return

        title_surface = self.title_font.render(message, True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=center)
        title_rect.y -= title_surface.get_height()

        target.blit(title_surface, title_rect)

        if subtext:
            body_surface = self.body_font.render(subtext, True, (255, 215, 0))
            body_rect = body_surface.get_rect(center=center)
            body_rect.y = title_rect.bottom + 8
            target.blit(body_surface, body_rect)


class DiagnosticsOverlay:
    """Render detection diagnostics (mode, signal strength, scores)."""

    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 36)

    def draw(self, target: pygame.Surface, detection: Optional[DetectionResult]) -> None:
        if detection is None:
            return

        lines = [f"Detection mode: {detection.mode.upper()}"]
        if detection.mode == "ir":
            lines.append(f"IR score: {detection.ir_score:.2f}")
            signal_text = f"Signal: {detection.signal_strength:.2f}"
        else:
            lines.append(f"Contour area: {int(detection.contour_area)}")
            signal_text = f"Signal: {int(detection.signal_strength)}"
        lines.append(signal_text)
        lines.append(f"Wave detected: {'YES' if detection.wave_detected else 'no'}")

        y = 20
        for line in lines:
            surface = self.font.render(line, True, (255, 255, 255))
            target.blit(surface, (20, y))
            y += surface.get_height() + 6


def _capture_phase(
    cap: Optional["cv2.VideoCapture"],
    config: MirrorConfig,
    screen_size: tuple[int, int],
    detector: Optional[WaveDetector],
) -> tuple[Optional[pygame.Surface], Optional[DetectionResult], Optional[tuple[int, int]]]:
    if cap is None:
        return None, None, None

    ret, frame = cap.read()
    if not ret:
        return None, None, None

    if config.mirror:
        frame = cv2.flip(frame, 1)

    frame = _rotate_frame(frame, config.rotate_deg)

    detection_result: Optional[DetectionResult] = None
    if detector is not None:
        detection_result = detector.process_frame(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_size = frame_rgb.shape[1], frame_rgb.shape[0]

    frame_surface = pygame.image.frombuffer(
        frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB"
    ).convert()
    if frame_surface.get_size() != screen_size:
        frame_surface = pygame.transform.smoothscale(frame_surface, screen_size)
    return frame_surface, detection_result, frame_size


def _update_phase(
    events: list[pygame.event.Event],
    state_machine: MirrorStateMachine,
    detector: Optional[WaveDetector],
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
            if detector is not None:
                detector.handle_key(event.key)

    state_machine.update()
    return running


def _render_phase(
    screen: pygame.Surface,
    frame_surface: Optional[pygame.Surface],
    state_machine: MirrorStateMachine,
    spinner: SpinnerOverlay,
    result_overlay: ResultOverlay,
    prompt_overlay: PromptOverlay,
    diagnostics_overlay: DiagnosticsOverlay,
    detection: Optional[DetectionResult],
    frame_size: Optional[tuple[int, int]],
    calibration: Optional[CalibrationMapping],
    calibration_config: CalibrationConfig,
) -> None:
    if frame_surface is not None:
        screen.blit(frame_surface, (0, 0))
    else:
        screen.fill((0, 0, 0))

    screen_size = screen.get_size()
    rect = screen.get_rect()

    default_prompt_anchor = (rect.centerx, max(80, rect.centery - rect.height // 4))
    default_spinner_anchor = rect.center
    default_result_anchor = rect.center

    calibrated_prompt_anchor = default_prompt_anchor
    calibrated_spinner_anchor = default_spinner_anchor
    calibrated_result_anchor = default_result_anchor

    if (
        calibration is not None
        and frame_size is not None
        and calibration.is_compatible(frame_size, screen_size)
        and calibration.target_points
    ):
        xs = [pt[0] for pt in calibration.target_points]
        ys = [pt[1] for pt in calibration.target_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_x = int(round((min_x + max_x) / 2.0))
        center_y = int(round((min_y + max_y) / 2.0))
        bbox_height = max(1, max_y - min_y)

        calibrated_spinner_anchor = (center_x, center_y)
        calibrated_result_anchor = (center_x, center_y)

        prompt_offset = max(80, int(round(bbox_height * 0.35)))
        prompt_y = max(40, min(screen_size[1] - 40, int(round(min_y - prompt_offset))))
        calibrated_prompt_anchor = (center_x, prompt_y)

    def _pick_anchor(use_calibration: bool, calibrated: tuple[int, int], fallback: tuple[int, int]) -> tuple[int, int]:
        return calibrated if use_calibration else fallback

    prompt_anchor = _pick_anchor(
        calibration_config.apply_prompt, calibrated_prompt_anchor, default_prompt_anchor
    )
    spinner_anchor = _pick_anchor(
        calibration_config.apply_spinner, calibrated_spinner_anchor, default_spinner_anchor
    )
    result_anchor = _pick_anchor(
        calibration_config.apply_result, calibrated_result_anchor, default_result_anchor
    )

    if state_machine.state is MirrorState.IDLE:
        prompt_overlay.draw(screen, "Wave to roll", prompt_anchor, "Raise your hand to start")
    elif state_machine.state is MirrorState.ROLLING:
        spinner.draw(screen, state_machine.time_in_state() * 2 * math.pi, spinner_anchor)
    elif state_machine.state is MirrorState.RESULT:
        result_overlay.draw(screen, state_machine.current_prize, result_anchor)

    diagnostics_overlay.draw(screen, detection)

    if detection is not None and detection.centroid is not None and frame_size is not None:
        try:
            mapped_point = map_point_to_screen(detection.centroid, frame_size, screen_size, calibration)
        except CalibrationError:
            mapped_point = None
        if mapped_point is not None:
            pygame.draw.circle(screen, (255, 0, 0), mapped_point, 14, 3)


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
    detector = WaveDetector(config.detection) if config.detection is not None else None
    spinner = SpinnerOverlay()
    result_overlay = ResultOverlay()
    prompt_overlay = PromptOverlay()
    diagnostics_overlay = DiagnosticsOverlay()
    calibration_mapping: Optional[CalibrationMapping] = None
    calibration_warning_emitted = False

    if config.calibration.enabled:
        if config.calibration.file is None:
            raise MirrorConfigError("Calibration enabled but no calibration file provided.")
        try:
            calibration_mapping = load_calibration_mapping(config.calibration.file)
        except CalibrationError as exc:
            print(f"[mirror] Unable to load calibration: {exc}. Using proportional scaling instead.")
            calibration_mapping = None
            calibration_warning_emitted = True

    last_frame: Optional[pygame.Surface] = None
    last_detection: Optional[DetectionResult] = None
    last_frame_size: Optional[tuple[int, int]] = None
    active_calibration: Optional[CalibrationMapping] = None

    try:
        running = True
        while running:
            events = pygame.event.get()
            running = _update_phase(events, state_machine, detector)

            frame_surface, detection_result, frame_size = _capture_phase(
                cap, config, screen_size, detector
            )
            if frame_surface is not None:
                last_frame = frame_surface
            if detection_result is not None:
                last_detection = detection_result
            if frame_size is not None:
                last_frame_size = frame_size
                if calibration_mapping is not None and active_calibration is None:
                    if calibration_mapping.is_compatible(last_frame_size, screen_size):
                        active_calibration = calibration_mapping
                    elif not calibration_warning_emitted:
                        print(
                            "[mirror] Calibration file does not match the current resolution; "
                            "falling back to proportional scaling."
                        )
                        calibration_warning_emitted = True

            if (
                detection_result is not None
                and detection_result.wave_detected
                and state_machine.state is MirrorState.IDLE
            ):
                state_machine.trigger_roll()

            _render_phase(
                screen,
                last_frame,
                state_machine,
                spinner,
                result_overlay,
                prompt_overlay,
                diagnostics_overlay,
                last_detection,
                last_frame_size,
                active_calibration,
                config.calibration,
            )

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
