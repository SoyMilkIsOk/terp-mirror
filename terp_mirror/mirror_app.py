"""Mirror application for Terp Wizard projector setup."""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "OpenCV (cv2) is required. Install it with 'pip install opencv-python' or "
        "'pip install -r requirements.txt'."
    ) from exc
import pygame
import yaml

try:
    from pygame._sdl2.video import Renderer, Texture, Window
except (ImportError, AttributeError):  # pragma: no cover - optional dependency guard
    Renderer = Texture = Window = None  # type: ignore[assignment]

from .detection import DetectionConfig, DetectionROI, DetectionResult, WaveDetector
from .calibration import (
    CalibrationConfig,
    CalibrationError,
    CalibrationMapping,
    load_calibration_mapping,
    map_point_to_screen,
)
from .prizes import GRAND_PRIZE_NAME, PrizeManager, PrizeStockConfig
from .states import MirrorState, MirrorStateMachine


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

HALLOWEEN_RED = (215, 0, 0)
PROMPT_OFFSET_LIMIT = 1500
_FONT_CANDIDATES: Sequence[str] = (
    "MyFont-v2",
    "Chalkduster",
    "Trattatello",
    "Party LET",
    "Papyrus",
)


LOGGER = logging.getLogger(__name__)


class MirrorConfigError(RuntimeError):
    """Raised when the mirror configuration is invalid."""


@dataclass(frozen=True)
class MirrorConfig:
    """Normalized mirror configuration values."""

    monitor_index: int
    control_monitor_index: int
    rotate_deg: int
    ui_rotate_deg: int
    prompt_offset: tuple[int, int]
    mirror: bool
    tracking_camera_index: int
    display_camera_index: int
    target_fps: int
    roll_duration: float
    result_duration: float
    cooldown_duration: float
    detection: DetectionConfig
    calibration: CalibrationConfig
    prizes: PrizeStockConfig
    dry_run: bool = False


def _parse_offset_entry(value, name: str) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    if not isinstance(value, dict):
        raise MirrorConfigError(f"{name} must be a mapping with 'x' and 'y' keys.")
    try:
        x = int(value.get("x", 0))
        y = int(value.get("y", 0))
    except (TypeError, ValueError) as exc:
        raise MirrorConfigError(f"{name}.x and {name}.y must be integers.") from exc
    return (x, y)


def _clamp_prompt_offset(value: int) -> int:
    return max(-PROMPT_OFFSET_LIMIT, min(PROMPT_OFFSET_LIMIT, int(value)))


def _load_spooky_font(size: int) -> pygame.font.Font:
    for name in _FONT_CANDIDATES:
        try:
            font = pygame.font.SysFont(name, size)
        except Exception:
            continue
        if font is not None:
            return font
    return pygame.font.Font(None, size)


def load_config(config_path: Optional[Path] = None) -> MirrorConfig:
    """Load the mirror configuration from YAML and validate it."""

    raw_path = config_path or CONFIG_PATH
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise MirrorConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    required_keys = {"monitor_index", "rotate_deg", "mirror", "camera", "target_fps"}
    missing = required_keys - data.keys()
    if missing:
        raise MirrorConfigError(f"Missing configuration keys: {', '.join(sorted(missing))}")

    camera_cfg = data.get("camera")
    if not isinstance(camera_cfg, dict):
        raise MirrorConfigError("Camera configuration must be a mapping.")

    base_camera_index = camera_cfg.get("index")
    tracking_camera_raw = camera_cfg.get("tracking_index", base_camera_index)
    display_camera_raw = camera_cfg.get("display_index", base_camera_index)
    if tracking_camera_raw is None or display_camera_raw is None:
        raise MirrorConfigError(
            "Camera configuration must include either 'index' or both 'tracking_index' and 'display_index'."
        )
    try:
        tracking_camera_index = int(tracking_camera_raw)
        display_camera_index = int(display_camera_raw)
    except (TypeError, ValueError) as exc:
        raise MirrorConfigError("Camera indexes must be integers.") from exc

    rotate_deg = int(data["rotate_deg"])
    if rotate_deg not in {0, 90, -90, 180, -180}:
        raise MirrorConfigError("rotate_deg must be one of {0, 90, -90, 180, -180}.")

    ui_rotate_raw = data.get("ui_rotate_deg", 0)
    try:
        ui_rotate_deg = int(ui_rotate_raw)
    except (TypeError, ValueError) as exc:
        raise MirrorConfigError("ui_rotate_deg must be an integer.") from exc
    if ui_rotate_deg not in {0, 90, -90, 180, -180}:
        raise MirrorConfigError("ui_rotate_deg must be one of {0, 90, -90, 180, -180}.")

    prompt_offset = (0, 0)
    ui_offsets_cfg = data.get("ui_offsets")
    if ui_offsets_cfg is not None:
        if not isinstance(ui_offsets_cfg, dict):
            raise MirrorConfigError("ui_offsets must be a mapping with a 'prompt' entry.")
        prompt_offset = _parse_offset_entry(ui_offsets_cfg.get("prompt"), "ui_offsets.prompt")
    if "prompt_offset" in data:
        prompt_offset = _parse_offset_entry(data.get("prompt_offset"), "prompt_offset")

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
        control_monitor_index=int(data.get("control_monitor_index", data["monitor_index"])),
        rotate_deg=rotate_deg,
        ui_rotate_deg=ui_rotate_deg,
        prompt_offset=prompt_offset,
        mirror=bool(data["mirror"]),
        tracking_camera_index=tracking_camera_index,
        display_camera_index=display_camera_index,
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


def _rotate_ui_surface(surface: pygame.Surface, rotate_deg: int) -> pygame.Surface:
    """Rotate a UI surface while preserving alpha transparency."""

    if rotate_deg == 0:
        return surface
    if rotate_deg not in {0, 90, -90, 180, -180}:
        raise MirrorConfigError("ui_rotate_deg must be one of {0, 90, -90, 180, -180}.")

    return pygame.transform.rotate(surface, -rotate_deg)


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
    ui_rotation_deg: int
    prompt_offset_x: int
    prompt_offset_y: int


@dataclass
class RuntimeToggles:
    """Flags mutated in response to operator hotkeys."""

    diagnostics_visible: bool = False
    controls_visible: bool = True
    dual_camera_preview: bool = False


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
        pygame.draw.arc(overlay, HALLOWEEN_RED, arc_rect, start_angle, end_angle, self.stroke)
        target.blit(overlay, overlay.get_rect(center=center))


class ResultOverlay:
    """Placeholder prize/result card overlay."""

    def __init__(self) -> None:
        self.title_font = _load_spooky_font(88)
        self.prize_font = _load_spooky_font(60)
        self.text_color = HALLOWEEN_RED
        self.border_color = HALLOWEEN_RED
        self._image_dir = Path(__file__).resolve().parent / "pics"
        self._image_keys: dict[str, pygame.Surface] = {}
        self._keyword_lookup: Sequence[tuple[str, str]] = (
            (GRAND_PRIZE_NAME.lower(), "bong"),
            ("grand", "bong"),
            ("bong", "bong"),
            ("shirt", "shirt"),
            ("lighter", "lighters"),
            ("candy", "candy"),
            ("scoop", "2pk"),
            ("2pk", "2pk"),
        )
        self._load_prize_images()

    def _load_prize_images(self) -> None:
        """Load prize artwork assets bundled with the app."""

        asset_files = {
            "shirt": "shirt.png",
            "lighters": "lighters.png",
            "2pk": "2pk.png",
            "bong": "bong.png",
            "candy": "candy.png",
        }
        for key, filename in asset_files.items():
            path = self._image_dir / filename
            if not path.exists():
                LOGGER.warning("Prize artwork missing: %s", path)
                continue
            try:
                surface = pygame.image.load(str(path)).convert_alpha()
            except pygame.error as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to load prize artwork %s: %s", path, exc)
                continue
            self._image_keys[key] = surface

    def _lookup_image(self, prize_text: Optional[str]) -> Optional[pygame.Surface]:
        if not prize_text:
            return None
        normalized = prize_text.lower()
        for keyword, key in self._keyword_lookup:
            if keyword in normalized:
                return self._image_keys.get(key)
        return None

    @staticmethod
    def _scale_image(
        surface: pygame.Surface, max_width: int, max_height: int
    ) -> pygame.Surface:
        width, height = surface.get_size()
        if width <= max_width and height <= max_height:
            return surface
        scale = min(max_width / max(1, width), max_height / max(1, height))
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return pygame.transform.smoothscale(surface, new_size)

    def draw(self, target: pygame.Surface, prize_text: Optional[str], center: tuple[int, int]) -> None:
        overlay = pygame.Surface(target.get_size(), pygame.SRCALPHA)

        title_surface = self.title_font.render("YOU WIN:", True, self.text_color)
        prize_message = prize_text if prize_text else "Mystery treat incoming!"
        body_surface = self.prize_font.render(prize_message, True, self.text_color)

        horizontal_padding = 48
        top_padding = 36
        bottom_padding = 48
        text_spacing = 24
        image_spacing = 36
        max_card_width = target.get_width() - 80
        max_card_height = target.get_height() - 80

        image_surface = self._lookup_image(prize_text)
        scaled_image: Optional[pygame.Surface] = None
        if image_surface is not None:
            max_image_width = max(180, int(target.get_width() * 0.35))
            if max_card_width > 0:
                inner_width_limit = max_card_width - horizontal_padding * 2
                if inner_width_limit > 0:
                    max_image_width = min(max_image_width, inner_width_limit)
            max_image_height = max(160, int(target.get_height() * 0.35))
            if max_card_height > 0:
                height_budget = (
                    max_card_height
                    - (top_padding + bottom_padding + title_surface.get_height() + text_spacing + body_surface.get_height())
                )
                if height_budget > 0:
                    max_image_height = min(max_image_height, height_budget)
            scaled_image = self._scale_image(image_surface, max_image_width, max_image_height)

        elements: list[tuple[str, pygame.Surface]] = [
            ("text", title_surface),
            ("text", body_surface),
        ]
        if scaled_image is not None:
            elements.append(("image", scaled_image))

        content_width = max(surface.get_width() for _, surface in elements)
        content_height = sum(surface.get_height() for _, surface in elements)

        for idx in range(len(elements) - 1):
            next_kind = elements[idx + 1][0]
            content_height += image_spacing if next_kind == "image" else text_spacing

        if max_card_width > 0:
            horizontal_padding = min(
                horizontal_padding,
                max(24, (max_card_width - content_width) // 2),
            )
        card_width = content_width + horizontal_padding * 2

        card_height = content_height + top_padding + bottom_padding
        if max_card_height > 0 and card_height > max_card_height:
            available_padding = max_card_height - content_height
            available_padding = max(40, available_padding)
            top_padding = available_padding // 2
            bottom_padding = max(32, available_padding - top_padding)
            card_height = content_height + top_padding + bottom_padding

        card_rect = pygame.Rect(0, 0, card_width, card_height)
        card_rect.center = center

        pygame.draw.rect(
            overlay,
            self.border_color,
            card_rect,
            width=8,
            border_radius=24,
        )

        cursor_y = card_rect.top + top_padding
        for idx, (kind, surface) in enumerate(elements):
            element_rect = surface.get_rect()
            element_rect.centerx = card_rect.centerx
            element_rect.top = cursor_y
            overlay.blit(surface, element_rect)

            cursor_y = element_rect.bottom
            if idx < len(elements) - 1:
                next_kind = elements[idx + 1][0]
                cursor_y += image_spacing if next_kind == "image" else text_spacing

        target.blit(overlay, (0, 0))


class PromptOverlay:
    """Display guidance text positioned using calibration data."""

    def __init__(self) -> None:
        self.title_font = _load_spooky_font(96)
        self.body_font = _load_spooky_font(56)
        self.text_color = HALLOWEEN_RED
        self.subtext_color = HALLOWEEN_RED

    def draw(
        self,
        target: pygame.Surface,
        message: str,
        center: tuple[int, int],
        subtext: Optional[str] = None,
        *,
        color: Optional[tuple[int, int, int]] = None,
        subtext_color: Optional[tuple[int, int, int]] = None,
    ) -> None:
        if not message:
            return

        title_color = color or self.text_color
        title_surface = self.title_font.render(message, True, title_color)
        title_rect = title_surface.get_rect(center=center)
        title_rect.y -= title_surface.get_height()

        target.blit(title_surface, title_rect)

        if subtext:
            body_color = subtext_color or self.subtext_color
            body_surface = self.body_font.render(subtext, True, body_color)
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
                lines.append(f"Max area: {int(det.max_contour_area)}")
            else:
                lines.append(f"Contour area: {int(det.contour_area)}")
                lines.append(f"Max area: {int(det.max_contour_area)}")
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

    def adjust(self, delta: int, multiplier: int = 1) -> None:
        if self._setter is None:
            return
        if delta == 0:
            return

        current_value = self.value()
        if isinstance(current_value, bool):
            self._setter(not current_value)
            return

        raw = float(current_value)
        new_value = raw + delta * self.step * multiplier
        if self.minimum is not None:
            new_value = max(self.minimum, new_value)
        if self.maximum is not None:
            new_value = min(self.maximum, new_value)
        try:
            coerced = self._coerce(new_value, delta, multiplier)
        except TypeError:
            coerced = self._coerce(new_value)
        self._setter(coerced)


class ControlAction:
    """Action-style control item that triggers a callable."""

    def __init__(
        self,
        label: str,
        action: Callable[[], None],
        status_getter: Optional[Callable[[], str]] = None,
    ) -> None:
        self.label = label
        self._action = action
        self._status_getter = status_getter or (lambda: "")

    def value(self):
        return self._status_getter()

    def formatted(self) -> str:
        return self._status_getter()

    def adjust(self, delta: int, multiplier: int = 1) -> None:
        if delta == 0:
            return
        self._action()


class ControlPanel:
    """Interactive panel allowing live configuration tweaks."""

    def __init__(
        self,
        settings: RuntimeSettings,
        state_machine: MirrorStateMachine,
        detector: Optional[WaveDetector],
        *,
        config: MirrorConfig,
        config_path: Path,
        prize_manager: PrizeManager,
    ) -> None:
        self.settings = settings
        self.state_machine = state_machine
        self.detector = detector
        self._base_config = config
        if config_path is not None:
            expanded = Path(config_path).expanduser()
            try:
                self.config_path = expanded.resolve(strict=False)
            except OSError:
                self.config_path = expanded
        else:
            self.config_path = None
        self.prize_manager = prize_manager
        self.font = pygame.font.Font(None, 26)
        self.header_font = pygame.font.Font(None, 32)
        self.items: list[ControlItem | ControlAction] = []
        self.selected_index = 0
        self._default_save_prompt = "Press Left/Right or Enter to save"
        self._dirty_save_prompt = "Unsaved changes - press Enter to save"
        self._save_message = self._default_save_prompt
        self._save_message_expires = 0.0
        self._dirty = False
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
                "UI rotation",
                getter=lambda: self.settings.ui_rotation_deg,
                setter=lambda value: setattr(self.settings, "ui_rotation_deg", int(value)),
                step=90,
                fmt="{:d}°",
                coerce=lambda value, delta, multiplier, steps=(-180, -90, 0, 90, 180), settings=self.settings: (
                    steps[
                        (
                            (steps.index(settings.ui_rotation_deg)
                            if settings.ui_rotation_deg in steps
                            else steps.index(0))
                            + int(delta or 0)
                        )
                        % len(steps)
                    ]
                ),
            ),
            ControlItem(
                "Wave text X offset",
                getter=lambda settings=self.settings: settings.prompt_offset_x,
                setter=lambda value, settings=self.settings: setattr(
                    settings, "prompt_offset_x", int(round(value))
                ),
                step=5,
                fmt="{:d}px",
                minimum=-PROMPT_OFFSET_LIMIT,
                maximum=PROMPT_OFFSET_LIMIT,
                coerce=lambda value: int(round(value)),
            ),
            ControlItem(
                "Wave text Y offset",
                getter=lambda settings=self.settings: settings.prompt_offset_y,
                setter=lambda value, settings=self.settings: setattr(
                    settings, "prompt_offset_y", int(round(value))
                ),
                step=5,
                fmt="{:d}px",
                minimum=-PROMPT_OFFSET_LIMIT,
                maximum=PROMPT_OFFSET_LIMIT,
                coerce=lambda value: int(round(value)),
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

        if self.detector is not None:
            cfg = self.detector.config

            def _odd_kernel(value: float, delta: int, multiplier: int) -> int:
                candidate = max(1, int(round(value)))
                if candidate % 2 == 0:
                    if delta < 0:
                        candidate = max(1, candidate - 1)
                    else:
                        candidate += 1
                return candidate

            self.items.extend(
                [
                    ControlItem(
                        "Min contour area",
                        getter=lambda cfg=cfg: cfg.min_contour_area,
                        setter=lambda value, cfg=cfg: setattr(cfg, "min_contour_area", float(value)),
                        step=1.0,
                        minimum=0.0,
                    ),
                    ControlItem(
                        "Blur kernel",
                        getter=lambda cfg=cfg: cfg.blur_kernel,
                        setter=lambda value, cfg=cfg: setattr(cfg, "blur_kernel", int(value)),
                        step=1,
                        fmt="{:d}",
                        minimum=1,
                        coerce=_odd_kernel,
                    ),
                    ControlItem(
                        "Morph kernel",
                        getter=lambda cfg=cfg: cfg.morph_kernel,
                        setter=lambda value, cfg=cfg: setattr(cfg, "morph_kernel", int(value)),
                        step=1,
                        fmt="{:d}",
                        minimum=1,
                        coerce=_odd_kernel,
                    ),
                    ControlItem(
                        "Morph iterations",
                        getter=lambda cfg=cfg: cfg.morph_iterations,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "morph_iterations", int(value)
                        ),
                        step=1,
                        fmt="{:d}",
                        minimum=0,
                        coerce=lambda value: max(0, int(round(value))),
                    ),
                    ControlItem(
                        "Min circularity",
                        getter=lambda cfg=cfg: cfg.min_circularity,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "min_circularity", float(value)
                        ),
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
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "buffer_duration", float(value)
                        ),
                        step=0.05,
                        minimum=0.1,
                    ),
                    ControlItem(
                        "Min wave span",
                        getter=lambda cfg=cfg: cfg.min_wave_span,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "min_wave_span", float(value)
                        ),
                        step=0.02,
                        minimum=0.05,
                    ),
                    ControlItem(
                        "Min wave velocity",
                        getter=lambda cfg=cfg: cfg.min_wave_velocity,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "min_wave_velocity", float(value)
                        ),
                        step=0.05,
                        minimum=0.05,
                    ),
                    ControlItem(
                        "Detection cooldown",
                        getter=lambda cfg=cfg: cfg.cooldown,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "cooldown", float(value)
                        ),
                        step=0.1,
                        minimum=0.0,
                    ),
                    ControlItem(
                        "IR enabled",
                        getter=lambda cfg=cfg: cfg.ir_enabled,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "ir_enabled", bool(value)
                        ),
                        step=1,
                    ),
                    ControlItem(
                        "IR buffer duration",
                        getter=lambda cfg=cfg: cfg.ir_buffer_duration,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "ir_buffer_duration", float(value)
                        ),
                        step=0.05,
                        minimum=0.05,
                    ),
                    ControlItem(
                        "IR threshold",
                        getter=lambda cfg=cfg: cfg.ir_score_threshold,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "ir_score_threshold", float(value)
                        ),
                        step=0.02,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    ControlItem(
                        "IR release",
                        getter=lambda cfg=cfg: cfg.ir_release_threshold,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "ir_release_threshold", float(value)
                        ),
                        step=0.02,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    ControlItem(
                        "IR debounce",
                        getter=lambda cfg=cfg: cfg.ir_debounce,
                        setter=lambda value, cfg=cfg: setattr(
                            cfg, "ir_debounce", float(value)
                        ),
                        step=0.1,
                        minimum=0.0,
                    ),
                    ControlItem(
                        "ROI X",
                        getter=lambda cfg=cfg: cfg.roi.x,
                        setter=lambda value, cfg=cfg: self._set_roi_value(
                            cfg, "x", float(value)
                        ),
                        step=0.01,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    ControlItem(
                        "ROI Y",
                        getter=lambda cfg=cfg: cfg.roi.y,
                        setter=lambda value, cfg=cfg: self._set_roi_value(
                            cfg, "y", float(value)
                        ),
                        step=0.01,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    ControlItem(
                        "ROI width",
                        getter=lambda cfg=cfg: cfg.roi.width,
                        setter=lambda value, cfg=cfg: self._set_roi_value(
                            cfg, "width", float(value)
                        ),
                        step=0.02,
                        minimum=0.1,
                        maximum=1.0,
                    ),
                    ControlItem(
                        "ROI height",
                        getter=lambda cfg=cfg: cfg.roi.height,
                        setter=lambda value, cfg=cfg: self._set_roi_value(
                            cfg, "height", float(value)
                        ),
                        step=0.02,
                        minimum=0.1,
                        maximum=1.0,
                    ),
                    ControlItem(
                        "HSV H lower",
                        getter=lambda cfg=cfg: cfg.hsv_lower[0],
                        setter=lambda value, cfg=cfg: self._set_hsv(
                            cfg, "lower", 0, int(round(value))
                        ),
                        step=1,
                        fmt="{:d}",
                        minimum=0,
                        maximum=179,
                        coerce=lambda value: int(round(value)),
                    ),
                    ControlItem(
                        "HSV H upper",
                        getter=lambda cfg=cfg: cfg.hsv_upper[0],
                        setter=lambda value, cfg=cfg: self._set_hsv(
                            cfg, "upper", 0, int(round(value))
                        ),
                        step=1,
                        fmt="{:d}",
                        minimum=0,
                        maximum=179,
                        coerce=lambda value: int(round(value)),
                    ),
                    ControlItem(
                        "HSV S lower",
                        getter=lambda cfg=cfg: cfg.hsv_lower[1],
                        setter=lambda value, cfg=cfg: self._set_hsv(
                            cfg, "lower", 1, int(round(value))
                        ),
                        step=2,
                        fmt="{:d}",
                        minimum=0,
                        maximum=255,
                        coerce=lambda value: int(round(value)),
                    ),
                    ControlItem(
                        "HSV S upper",
                        getter=lambda cfg=cfg: cfg.hsv_upper[1],
                        setter=lambda value, cfg=cfg: self._set_hsv(
                            cfg, "upper", 1, int(round(value))
                        ),
                        step=2,
                        fmt="{:d}",
                        minimum=0,
                        maximum=255,
                        coerce=lambda value: int(round(value)),
                    ),
                    ControlItem(
                        "HSV V lower",
                        getter=lambda cfg=cfg: cfg.hsv_lower[2],
                        setter=lambda value, cfg=cfg: self._set_hsv(
                            cfg, "lower", 2, int(round(value))
                        ),
                        step=2,
                        fmt="{:d}",
                        minimum=0,
                        maximum=255,
                        coerce=lambda value: int(round(value)),
                    ),
                    ControlItem(
                        "HSV V upper",
                        getter=lambda cfg=cfg: cfg.hsv_upper[2],
                        setter=lambda value, cfg=cfg: self._set_hsv(
                            cfg, "upper", 2, int(round(value))
                        ),
                        step=2,
                        fmt="{:d}",
                        minimum=0,
                        maximum=255,
                        coerce=lambda value: int(round(value)),
                    ),
                ]
            )

        self.items.append(
            ControlAction(
                "Save config",
                action=self.save_config,
                status_getter=self._current_save_status,
            )
        )

    @staticmethod
    def _values_changed(before, after) -> bool:
        if isinstance(before, bool) or isinstance(after, bool):
            return bool(before) != bool(after)
        numeric_types = (int, float)
        if isinstance(before, numeric_types) or isinstance(after, numeric_types):
            try:
                return not math.isclose(float(before), float(after), rel_tol=1e-9, abs_tol=1e-6)
            except (TypeError, ValueError):
                return before != after
        return before != after

    def mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
        self._set_save_status(self._dirty_save_prompt)

    @property
    def has_unsaved_changes(self) -> bool:
        return self._dirty

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

        roi = config.roi
        new_roi = replace(roi, **{attribute: value})

        max_x = max(0.0, 1.0 - new_roi.width)
        max_y = max(0.0, 1.0 - new_roi.height)
        clamped_x = min(max(new_roi.x, 0.0), max_x)
        clamped_y = min(max(new_roi.y, 0.0), max_y)

        if clamped_x != new_roi.x or clamped_y != new_roi.y:
            new_roi = replace(new_roi, x=clamped_x, y=clamped_y)

        config.roi = new_roi

    def move_selection(self, delta: int) -> None:
        if not self.items:
            return
        self.selected_index = (self.selected_index + delta) % len(self.items)

    def adjust_selection(self, delta: int, multiplier: int = 1) -> None:
        if not self.items or delta == 0:
            return
        current = self.items[self.selected_index]
        if isinstance(current, ControlAction):
            current.adjust(delta, multiplier)
            return
        before = current.value()
        current.adjust(delta, multiplier)
        after = current.value()
        if self._values_changed(before, after):
            self.mark_dirty()

    def activate_selection(self) -> bool:
        if not self.items:
            return False

        current = self.items[self.selected_index]
        if isinstance(current, ControlAction):
            current.adjust(1)
            return True

        value = current.value()
        if isinstance(value, bool):
            before = value
            current.adjust(1)
            after = current.value()
            if self._values_changed(before, after):
                self.mark_dirty()
            return True

        return False

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
        instruction = self.font.render(
            "Up/Down select | Left/Right adjust (hold Shift for x5)",
            True,
            (220, 220, 220),
        )
        overlay.blit(instruction, (16, y))
        y += instruction.get_height() + 10

        for idx, item in enumerate(self.items):
            text = f"{item.label}: {item.formatted()}"
            color = (255, 230, 180) if idx == self.selected_index else (255, 255, 255)
            surface = self.font.render(text, True, color)
            overlay.blit(surface, (20, y))
            y += surface.get_height() + 4

        target.blit(overlay, (target.get_width() - width - 16, 16))

    def _current_save_status(self) -> str:
        if self._save_message_expires and time.monotonic() > self._save_message_expires:
            if self._dirty:
                self._save_message = self._dirty_save_prompt
            else:
                self._save_message = self._default_save_prompt
            self._save_message_expires = 0.0
        if self._dirty and self._save_message == self._default_save_prompt:
            self._save_message = self._dirty_save_prompt
        return self._save_message

    def _set_save_status(self, message: str, duration: float = 0.0) -> None:
        self._save_message = message
        self._save_message_expires = (
            time.monotonic() + duration if duration > 0.0 else 0.0
        )

    def save_config(self, *, auto: bool = False) -> None:
        if not self.config_path:
            print("[controls] No configuration path available to save.")
            if not auto:
                self._set_save_status("No config path", duration=3.0)
            return

        payload = self._build_config_payload()
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
        except (OSError, yaml.YAMLError) as exc:
            print(f"[controls] Failed to save config: {exc}")
            if not auto:
                self._set_save_status("Save failed", duration=4.0)
            self._dirty = True
        else:
            if auto:
                print(f"[controls] Auto-saved configuration to {self.config_path}")
                self._save_message = self._default_save_prompt
                self._save_message_expires = 0.0
            else:
                print(f"[controls] Saved configuration to {self.config_path}")
                self._set_save_status("Saved!", duration=2.5)
            self._dirty = False

    def _build_config_payload(self) -> dict:
        data: dict[str, object] = {
            "monitor_index": self._base_config.monitor_index,
            "control_monitor_index": self._base_config.control_monitor_index,
            "rotate_deg": self._base_config.rotate_deg,
            "ui_rotate_deg": int(self.settings.ui_rotation_deg),
            "ui_offsets": {
                "prompt": {
                    "x": int(self.settings.prompt_offset_x),
                    "y": int(self.settings.prompt_offset_y),
                },
            },
            "mirror": bool(self.settings.mirror_enabled),
            "camera": self._build_camera_payload(),
            "target_fps": int(self.settings.target_fps),
            "dry_run": bool(self._base_config.dry_run),
            "timers": {
                "rolling": float(self.state_machine.roll_duration),
                "result": float(self.state_machine.result_duration),
                "cooldown": float(self.state_machine.cooldown_duration),
            },
        }

        detection_cfg = None
        if self.detector is not None:
            detection_cfg = self.detector.config
        elif self._base_config.detection is not None:
            detection_cfg = self._base_config.detection

        if detection_cfg is not None:
            roi = detection_cfg.roi
            data["detection"] = {
                "hsv_min": list(detection_cfg.hsv_lower),
                "hsv_max": list(detection_cfg.hsv_upper),
                "blur_kernel": int(detection_cfg.blur_kernel),
                "morph_kernel": int(detection_cfg.morph_kernel),
                "morph_iterations": int(detection_cfg.morph_iterations),
                "min_contour_area": float(detection_cfg.min_contour_area),
                "min_circularity": float(detection_cfg.min_circularity),
                "processing_interval": float(detection_cfg.processing_interval),
                "buffer_duration": float(detection_cfg.buffer_duration),
                "min_wave_span": float(detection_cfg.min_wave_span),
                "min_wave_velocity": float(detection_cfg.min_wave_velocity),
                "cooldown": float(detection_cfg.cooldown),
                "ir_enabled": bool(detection_cfg.ir_enabled),
                "ir_buffer_duration": float(detection_cfg.ir_buffer_duration),
                "ir_score_threshold": float(detection_cfg.ir_score_threshold),
                "ir_release_threshold": float(detection_cfg.ir_release_threshold),
                "ir_debounce": float(detection_cfg.ir_debounce),
                "roi": {
                    "x": float(roi.x),
                    "y": float(roi.y),
                    "width": float(roi.width),
                    "height": float(roi.height),
                },
            }

        calibration = self._base_config.calibration
        data["calibration"] = {
            "enabled": bool(calibration.enabled),
            "file": str(calibration.file) if calibration.file is not None else None,
            "apply_prompt": bool(calibration.apply_prompt),
            "apply_spinner": bool(calibration.apply_spinner),
            "apply_result": bool(calibration.apply_result),
        }

        if self.prize_manager is not None:
            prize_block: dict[str, object] = {
                "track_stock": bool(self.prize_manager.track_stock_enabled),
            }
            stock_snapshot = self.prize_manager.stock_snapshot()
            stock_entries = {
                name: int(count)
                for name, count in stock_snapshot.items()
                if name != GRAND_PRIZE_NAME and count is not None
            }
            if stock_entries or prize_block["track_stock"]:
                prize_block["stock"] = stock_entries
            grand_count = stock_snapshot.get(GRAND_PRIZE_NAME)
            if grand_count is not None:
                prize_block["grand_prize"] = int(grand_count)
            data["prizes"] = prize_block

        return data

    def _build_camera_payload(self) -> dict[str, int]:
        tracking_index = int(self._base_config.tracking_camera_index)
        display_index = int(self._base_config.display_camera_index)
        payload = {
            "tracking_index": tracking_index,
            "display_index": display_index,
        }
        if tracking_index == display_index:
            payload["index"] = tracking_index
        return payload


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

    def draw(self, target: pygame.Surface, messages: Sequence[str] | str) -> None:
        if isinstance(messages, str):
            lines = [messages]
        else:
            lines = list(messages)

        filtered = [line for line in lines if line and line.lower() != "connected"]
        if not filtered:
            return

        surfaces = [self.font.render(line, True, (255, 200, 200)) for line in filtered]
        padding_x = 18
        padding_y = 12
        width = max(surface.get_width() for surface in surfaces) + padding_x * 2
        height = sum(surface.get_height() for surface in surfaces) + padding_y * 2

        background = pygame.Surface((width, height), pygame.SRCALPHA)
        background.fill((120, 0, 0, 180))
        rect = background.get_rect(center=(target.get_width() // 2, 50))
        target.blit(background, rect)

        y = rect.top + padding_y
        for surface in surfaces:
            surface_rect = surface.get_rect(centerx=rect.centerx, top=y)
            target.blit(surface, surface_rect)
            y = surface_rect.bottom


class ControlDisplay:
    """Handle the operator-facing window rendered with SDL2."""

    def __init__(self, size: tuple[int, int], *, title: str = "Terp Mirror Controls") -> None:
        if Window is None or Renderer is None or Texture is None:
            raise MirrorConfigError(
                "pygame._sdl2 is required for the dual-display layout. Please upgrade pygame to 2.0+"
            )
        self.window = Window(title, size=size, fullscreen_desktop=True)
        self.renderer = Renderer(self.window)
        self.texture: Optional[Texture] = None
        self.window.focus()

    def present(self, surface: pygame.Surface) -> None:
        if self.texture is None or (self.texture.width, self.texture.height) != surface.get_size():
            self.texture = Texture.from_surface(self.renderer, surface)
        else:
            self.texture.update(surface)
        self.renderer.clear()
        self.renderer.blit(self.texture)
        self.renderer.present()

    def close(self) -> None:
        self.texture = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
        self.renderer = None


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
                control_panel.mark_dirty()
            handled = True
        elif event.unicode == "-":
            selection = prize_manager.current_selection()
            new_value = prize_manager.adjust_stock(selection, -1)
            if new_value is not None:
                print(f"[prizes] {selection} stock decreased to {new_value}")
                control_panel.mark_dirty()
            handled = True
        elif event.key == pygame.K_UP:
            control_panel.move_selection(-1)
            handled = True
        elif event.key == pygame.K_DOWN:
            control_panel.move_selection(1)
            handled = True
        elif event.key == pygame.K_LEFT:
            multiplier = 5 if event.mod & pygame.KMOD_SHIFT else 1
            control_panel.adjust_selection(-1, multiplier)
            handled = True
        elif event.key == pygame.K_RIGHT:
            multiplier = 5 if event.mod & pygame.KMOD_SHIFT else 1
            control_panel.adjust_selection(1, multiplier)
            handled = True
        elif event.key == pygame.K_F5:
            toggles.dual_camera_preview = not toggles.dual_camera_preview
            state = "enabled" if toggles.dual_camera_preview else "disabled"
            print(f"[mirror] Dual camera preview {state}.")
            handled = True
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            handled = control_panel.activate_selection()

        if detector is not None and not handled:
            if detector.handle_key(event.key, event.mod):
                control_panel.mark_dirty()

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
    camera_status: Sequence[str] | str,
    detection_paused: bool,
    mirror_enabled: bool,
    prompt_offset: tuple[int, int],
    ui_rotation_deg: int,
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

        prompt_margin = max(80, int(round(bbox_height * 0.35)))
        prompt_y = max(40, min(screen_size[1] - 40, int(round(min_y - prompt_margin))))
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

    def _apply_offset(anchor: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
        return (int(anchor[0] + offset[0]), int(anchor[1] + offset[1]))

    prompt_anchor = _apply_offset(prompt_anchor, prompt_offset)
    overlay_target = target if ui_rotation_deg == 0 else pygame.Surface(target.get_size(), pygame.SRCALPHA)
    if overlay_target is not target:
        overlay_target.fill((0, 0, 0, 0))
    prompt_layer = overlay_target
    prompt_anchor_draw = prompt_anchor
    prompt_layer_separate = False

    if ui_rotation_deg != 0:
        layer_size = (screen_size[0] * 2, screen_size[1] * 2)
        layer_offset = (screen_size[0] // 2, screen_size[1] // 2)
        prompt_layer = pygame.Surface(layer_size, pygame.SRCALPHA)
        prompt_layer.fill((0, 0, 0, 0))
        prompt_anchor_draw = (
            int(prompt_anchor[0] + layer_offset[0]),
            int(prompt_anchor[1] + layer_offset[1]),
        )
        prompt_layer_separate = True

    if detection_paused:
        prompt_overlay.draw(
            prompt_layer,
            "Detection paused",
            prompt_anchor_draw,
            "Press P to resume",
            color=(255, 255, 255),
            subtext_color=(255, 215, 0),
        )
    elif state_machine.state is MirrorState.IDLE:
        prompt_overlay.draw(
            prompt_layer,
            "Cast a Spell",
            prompt_anchor_draw,
            "Use the Terpwand",
        )
    elif state_machine.state is MirrorState.ROLLING:
        spinner.draw(
            overlay_target, state_machine.time_in_state() * 2 * math.pi, spinner_anchor
        )
    elif state_machine.state is MirrorState.RESULT:
        result_overlay.draw(overlay_target, state_machine.current_prize, result_anchor)

    if detection is not None and detection.centroid is not None and frame_size is not None:
        try:
            mapped_point = map_point_to_screen(detection.centroid, frame_size, screen_size, calibration)
        except CalibrationError:
            mapped_point = None
        if mapped_point is not None:
            if mirror_enabled:
                mirrored_x = screen_size[0] - mapped_point[0]
                mirrored_x = max(0, min(screen_size[0] - 1, mirrored_x))
                mapped_point = (mirrored_x, mapped_point[1])
            pygame.draw.circle(overlay_target, (255, 0, 0), mapped_point, 14, 3)

    if toggles.controls_visible:
        prize_overlay.draw(overlay_target, state_machine.prize_manager)
        control_panel.draw(overlay_target)
    camera_overlay.draw(overlay_target, camera_status)

    if diagnostics_data is not None:
        if mirror_enabled:
            overlay_surface = pygame.Surface(target.get_size(), pygame.SRCALPHA)
            diagnostics_overlay.draw(overlay_surface, diagnostics_data)
            flipped_overlay = pygame.transform.flip(overlay_surface, True, False)
            overlay_target.blit(flipped_overlay, (0, 0))
        else:
            diagnostics_overlay.draw(overlay_target, diagnostics_data)

    if overlay_target is not target:
        def _blit_rotated_layer(layer: pygame.Surface) -> None:
            content_rect = layer.get_bounding_rect()
            if content_rect.width == 0 and content_rect.height == 0:
                return

            rotated_overlay = _rotate_ui_surface(layer, ui_rotation_deg)
            target_rect = target.get_rect()
            rotated_rect = rotated_overlay.get_rect(center=target_rect.center)

            rotated_content_rect = rotated_overlay.get_bounding_rect()
            if rotated_content_rect.width and rotated_content_rect.height:
                offset_x = content_rect.left - (
                    rotated_rect.left + rotated_content_rect.left
                )
                offset_y = content_rect.top - (
                    rotated_rect.top + rotated_content_rect.top
                )
                rotated_rect.move_ip(offset_x, offset_y)

            target.blit(rotated_overlay, rotated_rect)

        if prompt_layer_separate:
            _blit_rotated_layer(prompt_layer)
        _blit_rotated_layer(overlay_target)


def _draw_picture_in_picture(
    target: pygame.Surface,
    inset_surface: pygame.Surface,
    *,
    width_ratio: float = 0.28,
    margin: int = 24,
) -> None:
    """Render a secondary camera feed as a picture-in-picture overlay."""

    if width_ratio <= 0:
        return

    inset_width = max(1, int(target.get_width() * width_ratio))
    aspect = inset_surface.get_width() / max(1, inset_surface.get_height())
    inset_height = max(1, int(inset_width / aspect))

    scaled = pygame.transform.smoothscale(inset_surface, (inset_width, inset_height))
    background = pygame.Surface((inset_width + margin, inset_height + margin), pygame.SRCALPHA)
    background.fill((20, 20, 20, 200))

    rect = background.get_rect()
    rect.topright = (target.get_width() - margin, margin)
    target.blit(background, rect)
    target.blit(scaled, scaled.get_rect(center=rect.center))


def run_mirror(
    config: MirrorConfig,
    monitor_override: Optional[int] = None,
    *,
    config_path: Path = CONFIG_PATH,
) -> None:
    """Run the mirror display loop using the supplied configuration."""

    monitor_index = monitor_override if monitor_override is not None else config.monitor_index
    control_monitor_index = config.control_monitor_index
    dry_run = config.dry_run

    pygame.init()
    pygame.display.set_caption("Terp Mirror")
    pygame.mouse.set_visible(False)
    screen_size = _resolve_display_size(monitor_index)
    control_size = _resolve_display_size(control_monitor_index)
    screen = pygame.display.set_mode(screen_size, pygame.FULLSCREEN, display=monitor_index)
    control_display = ControlDisplay(control_size)
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
        ui_rotation_deg=config.ui_rotate_deg,
        prompt_offset_x=_clamp_prompt_offset(config.prompt_offset[0]),
        prompt_offset_y=_clamp_prompt_offset(config.prompt_offset[1]),
    )
    control_panel = ControlPanel(
        settings,
        state_machine,
        detector,
        config=config,
        config_path=config_path,
        prize_manager=prize_manager,
    )
    prize_manager.register_change_listener(control_panel.mark_dirty)
    spinner = SpinnerOverlay()
    result_overlay = ResultOverlay()
    prompt_overlay = PromptOverlay()
    diagnostics_overlay = DiagnosticsOverlay()
    camera_overlay = CameraStatusOverlay()
    prize_overlay = PrizeStatusOverlay()
    toggles = RuntimeToggles()
    tracking_camera_manager = CameraManager(config.tracking_camera_index, enabled=not dry_run)
    display_camera_manager = CameraManager(config.display_camera_index, enabled=not dry_run)
    composite_surface = pygame.Surface(screen_size)
    control_surface = pygame.Surface(control_size)

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

    last_tracking_frame: Optional[pygame.Surface] = None
    last_display_frame: Optional[pygame.Surface] = None
    last_detection: Optional[DetectionResult] = None
    last_tracking_frame_size: Optional[tuple[int, int]] = None
    last_display_frame_size: Optional[tuple[int, int]] = None
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

            tracking_frame, detection_result, tracking_size = _capture_phase(
                tracking_camera_manager, config, control_size, detector
            )
            if tracking_frame is not None:
                last_tracking_frame = tracking_frame
            last_detection = detection_result
            if tracking_size is not None:
                last_tracking_frame_size = tracking_size

            display_frame, _, display_size = _capture_phase(
                display_camera_manager, config, screen_size, None
            )
            if display_frame is not None:
                if settings.mirror_enabled:
                    display_frame = pygame.transform.flip(display_frame, True, False)
                last_display_frame = display_frame
            if display_size is not None:
                last_display_frame_size = display_size
                if calibration_mapping is not None and active_calibration is None:
                    if calibration_mapping.is_compatible(last_display_frame_size, screen_size):
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

            tracking_status = tracking_camera_manager.status_text()
            display_status = display_camera_manager.status_text()
            diagnostics_data: Optional[DiagnosticsData] = None
            if toggles.diagnostics_visible:
                diagnostics_data = DiagnosticsData(
                    detection=last_detection,
                    fps=clock.get_fps(),
                    state=state_machine.state,
                    detection_paused=detector.paused if detector is not None else False,
                    last_trigger_latency=last_trigger_latency,
                    camera_status=(
                        f"Display: {display_status} | Tracking: {tracking_status}"
                    ),
                )

            control_camera_messages = [
                f"Tracking camera: {tracking_status}",
                f"Display camera: {display_status}",
            ]

            _render_phase(
                control_surface,
                last_tracking_frame,
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
                last_tracking_frame_size,
                active_calibration,
                config.calibration,
                control_camera_messages,
                detector.paused if detector is not None else False,
                False,
                (settings.prompt_offset_x, settings.prompt_offset_y),
                0,
            )

            if toggles.dual_camera_preview and last_display_frame is not None:
                _draw_picture_in_picture(control_surface, last_display_frame)

            control_display.present(control_surface)

            target_surface = composite_surface

            projector_toggles = RuntimeToggles(
                diagnostics_visible=False,
                controls_visible=False,
                dual_camera_preview=False,
            )

            _render_phase(
                target_surface,
                last_display_frame,
                state_machine,
                spinner,
                result_overlay,
                prompt_overlay,
                diagnostics_overlay,
                diagnostics_data,
                prize_overlay,
                control_panel,
                camera_overlay,
                projector_toggles,
                None,
                last_display_frame_size,
                active_calibration,
                config.calibration,
                [],
                detector.paused if detector is not None else False,
                settings.mirror_enabled,
                (settings.prompt_offset_x, settings.prompt_offset_y),
                settings.ui_rotation_deg,
            )

            if settings.mirror_enabled:
                flipped = pygame.transform.flip(target_surface, True, False)
                screen.blit(flipped, (0, 0))
            else:
                screen.blit(target_surface, (0, 0))

            pygame.display.flip()
            clock.tick(settings.target_fps)
    finally:
        if control_panel.has_unsaved_changes:
            control_panel.save_config(auto=True)
        tracking_camera_manager.close()
        display_camera_manager.close()
        control_display.close()
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
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    config = load_config(config_path)
    if args.dry_run:
        config = replace(config, dry_run=True)
    run_mirror(config, monitor_override=args.monitor, config_path=config_path)


if __name__ == "__main__":
    main()
