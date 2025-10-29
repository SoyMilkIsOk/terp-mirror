"""Mirror application for Terp Wizard projector setup."""

from __future__ import annotations

import argparse
import math
import time
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
from .prizes import PrizeManager, PrizeStockConfig
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
    prizes: PrizeStockConfig
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
        detection=detection_cfg,
        calibration=calibration_cfg,
        prizes=_parse_prize_config(data.get("prizes", {})),
        dry_run=bool(data.get("dry_run", False)),
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
        min_circularity = float(data.get("min_circularity", 0.65))
        processing_interval = float(data.get("processing_interval", 0.04))
        buffer_duration = float(data.get("buffer_duration", 0.6))
        min_wave_span = float(data.get("min_wave_span", 0.2))
        min_wave_velocity = float(data.get("min_wave_velocity", 0.4))
        cooldown = float(data.get("cooldown", 1.5))
        ir_buffer_duration = float(data.get("ir_buffer_duration", 0.35))
        ir_score_threshold = float(data.get("ir_score_threshold", 0.6))
        ir_release_threshold = float(data.get("ir_release_threshold", 0.4))
        ir_debounce = float(data.get("ir_debounce", 1.0))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise MirrorConfigError("Detection numeric parameters must be valid numbers.") from exc

    if min_contour_area <= 0:
        raise MirrorConfigError("Detection.min_contour_area must be positive.")
    if not 0.0 <= min_circularity <= 1.0:
        raise MirrorConfigError("Detection.min_circularity must be within [0, 1].")
    if processing_interval < 0:
        raise MirrorConfigError("Detection.processing_interval must be zero or greater.")
    if buffer_duration <= 0:
        raise MirrorConfigError("Detection.buffer_duration must be positive.")
    if min_wave_span <= 0:
        raise MirrorConfigError("Detection.min_wave_span must be positive.")
    if min_wave_velocity <= 0:
        raise MirrorConfigError("Detection.min_wave_velocity must be positive.")
    if cooldown < 0:
        raise MirrorConfigError("Detection.cooldown must be zero or greater.")
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
        min_circularity=min_circularity,
        processing_interval=processing_interval,
        roi=roi,
        buffer_duration=buffer_duration,
        min_wave_span=min_wave_span,
        min_wave_velocity=min_wave_velocity,
        cooldown=cooldown,
        ir_enabled=bool(data.get("ir_enabled", True)),
        ir_buffer_duration=ir_buffer_duration,
        ir_score_threshold=ir_score_threshold,
        ir_release_threshold=ir_release_threshold,
        ir_debounce=ir_debounce,
    )


def _parse_prize_config(data: dict) -> PrizeStockConfig:
    if not isinstance(data, dict):
        raise MirrorConfigError("Prizes configuration must be a mapping.")

    track = bool(data.get("track_stock", False))
    counts_raw = data.get("stock", {})
    if counts_raw is None:
        counts_raw = {}
    if not isinstance(counts_raw, dict):
        raise MirrorConfigError("Prizes.stock must be a mapping of prize names to counts.")

    counts: dict[str, int] = {}
    for name, value in counts_raw.items():
        try:
            counts[str(name)] = max(0, int(value))
        except (TypeError, ValueError) as exc:
            raise MirrorConfigError(
                "Prizes.stock values must be integers greater than or equal to zero."
            ) from exc

    grand_raw = data.get("grand_prize")
    grand_value: Optional[int]
    if grand_raw is None:
        grand_value = None
    else:
        try:
            grand_value = max(0, int(grand_raw))
        except (TypeError, ValueError) as exc:
            raise MirrorConfigError("Prizes.grand_prize must be a positive integer.") from exc

    return PrizeStockConfig(track_stock=track, counts=counts, grand_prize=grand_value)


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


@dataclass
class RuntimeSettings:
    """Mutable runtime settings adjusted from the control panel."""

    target_fps: int
    mirror_enabled: bool


@dataclass
class RuntimeToggles:
    """Flags mutated in response to operator hotkeys."""

    diagnostics_visible: bool = False
    controls_visible: bool = True


@dataclass
class DiagnosticsData:
    """Aggregate data presented in the diagnostics overlay."""

    detection: Optional[DetectionResult]
    fps: float
    state: MirrorState
    detection_paused: bool
    last_trigger_latency: Optional[float]
    camera_status: str


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
        self.font = pygame.font.Font(None, 32)
        self.sub_font = pygame.font.Font(None, 26)

    def draw(self, target: pygame.Surface, data: Optional[DiagnosticsData]) -> None:
        if data is None:
            return

        lines: list[str] = []
        lines.append(f"FPS: {data.fps:5.1f}")
        lines.append(f"State: {data.state.value}")

        if data.camera_status:
            lines.append(f"Camera: {data.camera_status}")

        if data.detection_paused:
            lines.append("Detection: PAUSED")
        elif data.detection is None:
            lines.append("Detection: awaiting signal")
        else:
            det = data.detection
            lines.append(f"Detection mode: {det.mode.upper()}")
            if det.mode == "ir":
                lines.append(f"IR intensity: {det.ir_score:.2f}")
                lines.append(f"Signal: {det.signal_strength:.2f}")
            else:
                lines.append(f"Contour area: {int(det.contour_area)}")
                lines.append(f"Signal: {int(det.signal_strength)}")
            lines.append(f"Wave detected: {'YES' if det.wave_detected else 'no'}")

        if data.last_trigger_latency is not None:
            lines.append(f"Last trigger latency: {data.last_trigger_latency * 1000:.0f} ms")

        y = 20
        for idx, line in enumerate(lines):
            font = self.font if idx < 2 else self.sub_font
            surface = font.render(line, True, (255, 255, 255))
            target.blit(surface, (20, y))
            y += surface.get_height() + 6


class PrizeStatusOverlay:
    """Display prize selection state and stock levels."""

    def __init__(self) -> None:
        self.title_font = pygame.font.Font(None, 32)
        self.body_font = pygame.font.Font(None, 26)

    def draw(self, target: pygame.Surface, manager: PrizeManager) -> None:
        width = int(target.get_width() * 0.32)
        height = 160
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        pygame.draw.rect(overlay, (90, 0, 140, 220), overlay.get_rect(), border_radius=12)

        selected = manager.current_selection()
        stock = manager.stock_for(selected)
        queued = manager.queued_manual
        lines = [
            f"Selected: {selected}",
            f"Stock: {'∞' if stock is None else stock}",
            f"Queued: {queued or 'auto'}",
            "Hotkeys: [ ] select | +/- stock | G queue",
        ]

        y = 14
        for idx, line in enumerate(lines):
            font = self.title_font if idx == 0 else self.body_font
            text = font.render(line, True, (255, 255, 255))
            overlay.blit(text, (16, y))
            y += text.get_height() + 6

        target.blit(overlay, (16, target.get_height() - height - 20))


class ControlItem:
    """Single adjustable value rendered inside the control panel."""

    def __init__(
        self,
        label: str,
        getter,
        setter,
        *,
        step: float,
        fmt: str = "{:.2f}",
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        coerce=lambda value: value,
    ) -> None:
        self.label = label
        self._getter = getter
        self._setter = setter
        self.step = step
        self.fmt = fmt
        self.minimum = minimum
        self.maximum = maximum
        self._coerce = coerce

    def value(self):
        return self._getter()

    def formatted(self) -> str:
        value = self.value()
        if isinstance(value, bool):
            return "On" if value else "Off"
        try:
            return self.fmt.format(value)
        except (ValueError, TypeError):
            return str(value)

    def adjust(self, delta: int) -> None:
        if self._setter is None:
            return
        if delta == 0:
            return

        current_value = self.value()
        if isinstance(current_value, bool):
            self._setter(not current_value)
            return

        raw = float(current_value)
        new_value = raw + delta * self.step
        if self.minimum is not None:
            new_value = max(self.minimum, new_value)
        if self.maximum is not None:
            new_value = min(self.maximum, new_value)
        coerced = self._coerce(new_value)
        self._setter(coerced)


class ControlPanel:
    """Interactive panel allowing live configuration tweaks."""

    def __init__(
        self,
        settings: RuntimeSettings,
        state_machine: MirrorStateMachine,
        detector: Optional[WaveDetector],
    ) -> None:
        self.settings = settings
        self.state_machine = state_machine
        self.detector = detector
        self.font = pygame.font.Font(None, 26)
        self.header_font = pygame.font.Font(None, 32)
        self.items: list[ControlItem] = []
        self.selected_index = 0
        self._build_items()

    def _build_items(self) -> None:
        self.items = [
            ControlItem(
                "Mirror output",
                getter=lambda: self.settings.mirror_enabled,
                setter=lambda value: setattr(self.settings, "mirror_enabled", bool(value)),
                step=1,
                fmt="{}",
            ),
            ControlItem(
                "Target FPS",
                getter=lambda: self.settings.target_fps,
                setter=lambda value: setattr(self.settings, "target_fps", int(value)),
                step=1,
                fmt="{:d}",
                minimum=15,
                maximum=240,
                coerce=lambda value: int(round(value)),
            ),
            ControlItem(
                "Roll duration",
                getter=lambda: self.state_machine.roll_duration,
                setter=lambda value: setattr(self.state_machine, "roll_duration", float(value)),
                step=0.25,
                minimum=0.5,
            ),
            ControlItem(
                "Result duration",
                getter=lambda: self.state_machine.result_duration,
                setter=lambda value: setattr(self.state_machine, "result_duration", float(value)),
                step=0.25,
                minimum=0.5,
            ),
            ControlItem(
                "Cooldown duration",
                getter=lambda: self.state_machine.cooldown_duration,
                setter=lambda value: setattr(self.state_machine, "cooldown_duration", float(value)),
                step=0.25,
                minimum=0.0,
            ),
        ]

        if self.detector is None:
            return

        cfg = self.detector.config
        self.items.extend(
            [
                ControlItem(
                    "Min contour area",
                    getter=lambda cfg=cfg: cfg.min_contour_area,
                    setter=lambda value, cfg=cfg: setattr(cfg, "min_contour_area", float(value)),
                    step=100.0,
                    minimum=0.0,
                ),
                ControlItem(
                    "Min circularity",
                    getter=lambda cfg=cfg: cfg.min_circularity,
                    setter=lambda value, cfg=cfg: setattr(cfg, "min_circularity", float(value)),
                    step=0.02,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ControlItem(
                    "Processing interval (s)",
                    getter=lambda cfg=cfg: cfg.processing_interval,
                    setter=lambda value, cfg=cfg: setattr(
                        cfg, "processing_interval", float(value)
                    ),
                    step=0.01,
                    minimum=0.0,
                    maximum=0.5,
                ),
                ControlItem(
                    "Buffer duration",
                    getter=lambda cfg=cfg: cfg.buffer_duration,
                    setter=lambda value, cfg=cfg: setattr(cfg, "buffer_duration", float(value)),
                    step=0.05,
                    minimum=0.1,
                ),
                ControlItem(
                    "Min wave span",
                    getter=lambda cfg=cfg: cfg.min_wave_span,
                    setter=lambda value, cfg=cfg: setattr(cfg, "min_wave_span", float(value)),
                    step=0.02,
                    minimum=0.05,
                ),
                ControlItem(
                    "Min wave velocity",
                    getter=lambda cfg=cfg: cfg.min_wave_velocity,
                    setter=lambda value, cfg=cfg: setattr(cfg, "min_wave_velocity", float(value)),
                    step=0.05,
                    minimum=0.05,
                ),
                ControlItem(
                    "Detection cooldown",
                    getter=lambda cfg=cfg: cfg.cooldown,
                    setter=lambda value, cfg=cfg: setattr(cfg, "cooldown", float(value)),
                    step=0.1,
                    minimum=0.0,
                ),
                ControlItem(
                    "IR threshold",
                    getter=lambda cfg=cfg: cfg.ir_score_threshold,
                    setter=lambda value, cfg=cfg: setattr(cfg, "ir_score_threshold", float(value)),
                    step=0.02,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ControlItem(
                    "IR release",
                    getter=lambda cfg=cfg: cfg.ir_release_threshold,
                    setter=lambda value, cfg=cfg: setattr(cfg, "ir_release_threshold", float(value)),
                    step=0.02,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ControlItem(
                    "IR debounce",
                    getter=lambda cfg=cfg: cfg.ir_debounce,
                    setter=lambda value, cfg=cfg: setattr(cfg, "ir_debounce", float(value)),
                    step=0.1,
                    minimum=0.0,
                ),
                ControlItem(
                    "ROI X",
                    getter=lambda cfg=cfg: cfg.roi.x,
                    setter=lambda value, cfg=cfg: self._set_roi_value(cfg, "x", float(value)),
                    step=0.01,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ControlItem(
                    "ROI Y",
                    getter=lambda cfg=cfg: cfg.roi.y,
                    setter=lambda value, cfg=cfg: self._set_roi_value(cfg, "y", float(value)),
                    step=0.01,
                    minimum=0.0,
                    maximum=1.0,
                ),
                ControlItem(
                    "ROI width",
                    getter=lambda cfg=cfg: cfg.roi.width,
                    setter=lambda value, cfg=cfg: self._set_roi_value(cfg, "width", float(value)),
                    step=0.02,
                    minimum=0.1,
                    maximum=1.0,
                ),
                ControlItem(
                    "ROI height",
                    getter=lambda cfg=cfg: cfg.roi.height,
                    setter=lambda value, cfg=cfg: self._set_roi_value(cfg, "height", float(value)),
                    step=0.02,
                    minimum=0.1,
                    maximum=1.0,
                ),
                ControlItem(
                    "HSV H lower",
                    getter=lambda cfg=cfg: cfg.hsv_lower[0],
                    setter=lambda value, cfg=cfg: self._set_hsv(cfg, "lower", 0, int(round(value))),
                    step=1,
                    fmt="{:d}",
                    minimum=0,
                    maximum=179,
                    coerce=lambda value: int(round(value)),
                ),
                ControlItem(
                    "HSV H upper",
                    getter=lambda cfg=cfg: cfg.hsv_upper[0],
                    setter=lambda value, cfg=cfg: self._set_hsv(cfg, "upper", 0, int(round(value))),
                    step=1,
                    fmt="{:d}",
                    minimum=0,
                    maximum=179,
                    coerce=lambda value: int(round(value)),
                ),
                ControlItem(
                    "HSV S lower",
                    getter=lambda cfg=cfg: cfg.hsv_lower[1],
                    setter=lambda value, cfg=cfg: self._set_hsv(cfg, "lower", 1, int(round(value))),
                    step=2,
                    fmt="{:d}",
                    minimum=0,
                    maximum=255,
                    coerce=lambda value: int(round(value)),
                ),
                ControlItem(
                    "HSV S upper",
                    getter=lambda cfg=cfg: cfg.hsv_upper[1],
                    setter=lambda value, cfg=cfg: self._set_hsv(cfg, "upper", 1, int(round(value))),
                    step=2,
                    fmt="{:d}",
                    minimum=0,
                    maximum=255,
                    coerce=lambda value: int(round(value)),
                ),
                ControlItem(
                    "HSV V lower",
                    getter=lambda cfg=cfg: cfg.hsv_lower[2],
                    setter=lambda value, cfg=cfg: self._set_hsv(cfg, "lower", 2, int(round(value))),
                    step=2,
                    fmt="{:d}",
                    minimum=0,
                    maximum=255,
                    coerce=lambda value: int(round(value)),
                ),
                ControlItem(
                    "HSV V upper",
                    getter=lambda cfg=cfg: cfg.hsv_upper[2],
                    setter=lambda value, cfg=cfg: self._set_hsv(cfg, "upper", 2, int(round(value))),
                    step=2,
                    fmt="{:d}",
                    minimum=0,
                    maximum=255,
                    coerce=lambda value: int(round(value)),
                ),
            ]
        )

    @staticmethod
    def _set_hsv(config: DetectionConfig, bound: str, channel: int, value: int) -> None:
        lower = list(config.hsv_lower)
        upper = list(config.hsv_upper)
        if bound == "lower":
            lower[channel] = value
        else:
            upper[channel] = value
        if channel == 0:
            lower[channel] = max(0, min(lower[channel], upper[channel]))
            upper[channel] = max(lower[channel], min(179, upper[channel]))
        else:
            lower[channel] = max(0, min(lower[channel], upper[channel]))
            upper[channel] = max(lower[channel], min(255, upper[channel]))
        config.hsv_lower = tuple(lower)
        config.hsv_upper = tuple(upper)

    @staticmethod
    def _set_roi_value(config: DetectionConfig, attribute: str, value: float) -> None:
        value = float(value)
        if attribute in {"width", "height"}:
            value = max(0.05, min(1.0, value))
        else:
            value = max(0.0, min(1.0, value))

        setattr(config.roi, attribute, value)
        config.roi.x = max(0.0, min(config.roi.x, 1.0 - config.roi.width))
        config.roi.y = max(0.0, min(config.roi.y, 1.0 - config.roi.height))

    def move_selection(self, delta: int) -> None:
        if not self.items:
            return
        self.selected_index = (self.selected_index + delta) % len(self.items)

    def adjust_selection(self, delta: int) -> None:
        if not self.items:
            return
        self.items[self.selected_index].adjust(delta)

    def draw(self, target: pygame.Surface) -> None:
        if not self.items:
            return

        width = int(target.get_width() * 0.28)
        height = min(target.get_height() - 60, 80 + len(self.items) * 26)
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        pygame.draw.rect(overlay, (0, 90, 120, 210), overlay.get_rect(), border_radius=12)

        y = 16
        header = self.header_font.render("Controls Panel", True, (255, 255, 255))
        overlay.blit(header, (16, y))
        y += header.get_height() + 6
        instruction = self.font.render("↑↓ select | ←→ adjust", True, (220, 220, 220))
        overlay.blit(instruction, (16, y))
        y += instruction.get_height() + 10

        for idx, item in enumerate(self.items):
            text = f"{item.label}: {item.formatted()}"
            color = (255, 230, 180) if idx == self.selected_index else (255, 255, 255)
            surface = self.font.render(text, True, color)
            overlay.blit(surface, (20, y))
            y += surface.get_height() + 4

        target.blit(overlay, (target.get_width() - width - 16, 16))


class CameraManager:
    """Robust wrapper around ``cv2.VideoCapture`` with retry support."""

    def __init__(self, camera_index: int, *, enabled: bool = True, retry_interval: float = 2.5) -> None:
        self.camera_index = camera_index
        self.enabled = enabled
        self.retry_interval = retry_interval
        self._capture: Optional["cv2.VideoCapture"] = None
        self._last_attempt: float = 0.0
        self._status: str = "disabled" if not enabled else "initializing"
        if self.enabled:
            self._open()

    def _open(self) -> None:
        self.close()
        self._last_attempt = time.monotonic()
        capture = cv2.VideoCapture(self.camera_index)
        if capture.isOpened():
            self._capture = capture
            self._status = "connected"
        else:
            self._status = "retrying"

    def read(self) -> Optional[object]:
        if not self.enabled:
            self._status = "dry-run"
            return None

        if self._capture is None:
            if time.monotonic() - self._last_attempt >= self.retry_interval:
                self._open()
            return None

        ret, frame = self._capture.read()
        if not ret or frame is None:
            self.close()
            self._status = "lost"
            self._last_attempt = time.monotonic()
            return None

        self._status = "connected"
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def status_text(self) -> str:
        if self._status == "connected":
            return ""
        if self._status == "dry-run":
            return "Dry-run mode (camera disabled)"
        if self._status == "retrying":
            return "Camera unavailable, retrying..."
        if self._status == "initializing":
            return "Connecting to camera..."
        if self._status == "lost":
            remaining = max(0.0, self.retry_interval - (time.monotonic() - self._last_attempt))
            return f"Camera lost. Retrying in {remaining:.1f}s"
        return self._status


class CameraStatusOverlay:
    """Prominently display camera connection issues."""

    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 32)

    def draw(self, target: pygame.Surface, message: str) -> None:
        if not message:
            return

        text_surface = self.font.render(message, True, (255, 200, 200))
        padding = 14
        background = pygame.Surface(
            (text_surface.get_width() + padding * 2, text_surface.get_height() + padding),
            pygame.SRCALPHA,
        )
        background.fill((120, 0, 0, 180))
        rect = background.get_rect(center=(target.get_width() // 2, 40))
        target.blit(background, rect)
        target.blit(text_surface, text_surface.get_rect(center=rect.center))


def _capture_phase(
    camera: Optional[CameraManager],
    config: MirrorConfig,
    screen_size: tuple[int, int],
    detector: Optional[WaveDetector],
) -> tuple[Optional[pygame.Surface], Optional[DetectionResult], Optional[tuple[int, int]]]:
    if camera is None:
        return None, None, None

    raw_frame = camera.read()
    if raw_frame is None:
        return None, None, None

    frame = _rotate_frame(raw_frame, config.rotate_deg)

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
    prize_manager: PrizeManager,
    control_panel: ControlPanel,
    toggles: RuntimeToggles,
) -> tuple[bool, bool]:
    running = True
    manual_triggered = False

    for event in events:
        if event.type == pygame.QUIT:
            running = False
            continue

        if event.type != pygame.KEYDOWN:
            continue

        handled = False
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            running = False
            handled = True
        elif event.key == pygame.K_r:
            state_machine.trigger_roll()
            handled = True
        elif event.key == pygame.K_f:
            state_machine.force_result()
            handled = True
        elif event.key == pygame.K_g:
            if event.mod & pygame.KMOD_CTRL:
                state_machine.force_grand_prize()
            else:
                selection = prize_manager.queue_manual_selection()
                if selection is not None:
                    state_machine.queue_manual_prize(selection)
                    print(f"[prizes] Queued manual prize: {selection}")
                    manual_triggered = True
            handled = True
        elif event.key == pygame.K_p and detector is not None:
            detector.toggle_pause()
            handled = True
        elif event.key == pygame.K_d:
            toggles.diagnostics_visible = not toggles.diagnostics_visible
            handled = True
        elif event.key in (pygame.K_LALT, pygame.K_RALT):
            if not getattr(event, "repeat", False):
                toggles.controls_visible = not toggles.controls_visible
            handled = True
        elif event.key == pygame.K_LEFTBRACKET:
            prize_manager.cycle_selection(-1)
            print(f"[prizes] Selected prize: {prize_manager.current_selection()}")
            handled = True
        elif event.key == pygame.K_RIGHTBRACKET:
            prize_manager.cycle_selection(1)
            print(f"[prizes] Selected prize: {prize_manager.current_selection()}")
            handled = True
        elif event.unicode == "+":
            selection = prize_manager.current_selection()
            new_value = prize_manager.adjust_stock(selection, 1)
            if new_value is not None:
                print(f"[prizes] {selection} stock increased to {new_value}")
            handled = True
        elif event.unicode == "-":
            selection = prize_manager.current_selection()
            new_value = prize_manager.adjust_stock(selection, -1)
            if new_value is not None:
                print(f"[prizes] {selection} stock decreased to {new_value}")
            handled = True
        elif event.key == pygame.K_UP:
            control_panel.move_selection(-1)
            handled = True
        elif event.key == pygame.K_DOWN:
            control_panel.move_selection(1)
            handled = True
        elif event.key == pygame.K_LEFT:
            control_panel.adjust_selection(-1)
            handled = True
        elif event.key == pygame.K_RIGHT:
            control_panel.adjust_selection(1)
            handled = True

        if detector is not None and not handled:
            detector.handle_key(event.key)

    state_machine.update()
    return running, manual_triggered


def _render_phase(
    target: pygame.Surface,
    frame_surface: Optional[pygame.Surface],
    state_machine: MirrorStateMachine,
    spinner: SpinnerOverlay,
    result_overlay: ResultOverlay,
    prompt_overlay: PromptOverlay,
    diagnostics_overlay: DiagnosticsOverlay,
    diagnostics_data: Optional[DiagnosticsData],
    prize_overlay: PrizeStatusOverlay,
    control_panel: ControlPanel,
    camera_overlay: CameraStatusOverlay,
    toggles: RuntimeToggles,
    detection: Optional[DetectionResult],
    frame_size: Optional[tuple[int, int]],
    calibration: Optional[CalibrationMapping],
    calibration_config: CalibrationConfig,
    camera_status: str,
    detection_paused: bool,
) -> None:
    if frame_surface is not None:
        target.blit(frame_surface, (0, 0))
    else:
        target.fill((0, 0, 0))

    screen_size = target.get_size()
    rect = target.get_rect()

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

    if detection_paused:
        prompt_overlay.draw(target, "Detection paused", prompt_anchor, "Press P to resume")
    elif state_machine.state is MirrorState.IDLE:
        prompt_overlay.draw(target, "Wave to roll", prompt_anchor, "Raise your hand to start")
    elif state_machine.state is MirrorState.ROLLING:
        spinner.draw(target, state_machine.time_in_state() * 2 * math.pi, spinner_anchor)
    elif state_machine.state is MirrorState.RESULT:
        result_overlay.draw(target, state_machine.current_prize, result_anchor)

    if detection is not None and detection.centroid is not None and frame_size is not None:
        try:
            mapped_point = map_point_to_screen(detection.centroid, frame_size, screen_size, calibration)
        except CalibrationError:
            mapped_point = None
        if mapped_point is not None:
            pygame.draw.circle(target, (255, 0, 0), mapped_point, 14, 3)

    prize_overlay.draw(target, state_machine.prize_manager)
    if toggles.controls_visible:
        control_panel.draw(target)
    camera_overlay.draw(target, camera_status)

    if diagnostics_data is not None:
        diagnostics_overlay.draw(target, diagnostics_data)


def run_mirror(config: MirrorConfig, monitor_override: Optional[int] = None) -> None:
    """Run the mirror display loop using the supplied configuration."""

    monitor_index = monitor_override if monitor_override is not None else config.monitor_index
    camera_index = config.camera_index
    dry_run = config.dry_run

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
    prize_manager = PrizeManager(stock_config=config.prizes, dry_run=dry_run)
    state_machine.prize_manager = prize_manager
    detector = WaveDetector(config.detection) if config.detection is not None else None
    settings = RuntimeSettings(
        target_fps=config.target_fps,
        mirror_enabled=config.mirror,
    )
    control_panel = ControlPanel(settings, state_machine, detector)
    spinner = SpinnerOverlay()
    result_overlay = ResultOverlay()
    prompt_overlay = PromptOverlay()
    diagnostics_overlay = DiagnosticsOverlay()
    camera_overlay = CameraStatusOverlay()
    prize_overlay = PrizeStatusOverlay()
    toggles = RuntimeToggles()
    camera_manager = CameraManager(camera_index, enabled=not dry_run)
    composite_surface = pygame.Surface(screen_size)

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
    last_trigger_latency: Optional[float] = None
    last_wave_trigger: Optional[float] = None

    try:
        running = True
        while running:
            events = pygame.event.get()
            running, manual_triggered = _update_phase(
                events, state_machine, detector, prize_manager, control_panel, toggles
            )

            frame_surface, detection_result, frame_size = _capture_phase(
                camera_manager, config, screen_size, detector
            )
            if frame_surface is not None:
                last_frame = frame_surface
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

            if manual_triggered:
                last_trigger_latency = 0.0
                last_wave_trigger = None

            if (
                detection_result is not None
                and detection_result.wave_detected
                and state_machine.state is MirrorState.IDLE
                and detector is not None
                and not detector.paused
            ):
                last_wave_trigger = time.monotonic()
                state_machine.trigger_roll()
                last_trigger_latency = (
                    state_machine.last_trigger_timestamp - last_wave_trigger
                    if last_wave_trigger is not None
                    else None
                )
                last_wave_trigger = None

            camera_status = camera_manager.status_text()
            diagnostics_data: Optional[DiagnosticsData] = None
            if toggles.diagnostics_visible:
                diagnostics_data = DiagnosticsData(
                    detection=last_detection,
                    fps=clock.get_fps(),
                    state=state_machine.state,
                    detection_paused=detector.paused if detector is not None else False,
                    last_trigger_latency=last_trigger_latency,
                    camera_status=camera_status,
                )

            target_surface = composite_surface
            if last_frame is None:
                target_surface.fill((0, 0, 0))

            _render_phase(
                target_surface,
                last_frame,
                state_machine,
                spinner,
                result_overlay,
                prompt_overlay,
                diagnostics_overlay,
                diagnostics_data,
                prize_overlay,
                control_panel,
                camera_overlay,
                toggles,
                last_detection,
                last_frame_size,
                active_calibration,
                config.calibration,
                camera_status,
                detector.paused if detector is not None else False,
            )

            if settings.mirror_enabled:
                flipped = pygame.transform.flip(target_surface, True, False)
                screen.blit(flipped, (0, 0))
            else:
                screen.blit(target_surface, (0, 0))

            pygame.display.flip()
            clock.tick(settings.target_fps)
    finally:
        camera_manager.close()
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
