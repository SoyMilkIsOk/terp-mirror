# terp-mirror

Projection mirror code for the Terp Wizard - Terpscoops Halloween booth.

## Configuration

Runtime settings are stored in [`config.yaml`](config.yaml) at the project root.
You can adjust the projector monitor, camera, image rotation, and mirroring
options without modifying the code.

```yaml
monitor_index: 0
rotate_deg: 90
mirror: true
camera:
  index: 0
target_fps: 60
timers:
  rolling: 3.0
  result: 5.0
  cooldown: 2.5
```

* `monitor_index` &mdash; Which projector/monitor the fullscreen output should use
  (Pygame display index).
* `rotate_deg` &mdash; Rotation applied to each frame. Supported values are
  `0`, `90`, `-90`, `180`, and `-180` degrees.
* `mirror` &mdash; When `true`, the feed is mirrored horizontally. You can also
  toggle mirroring at runtime from the controls panel while calibrating or
  testing to verify the orientation looks correct.
* `camera.index` &mdash; The OpenCV camera index to open.
* `target_fps` &mdash; Desired update rate for the projector display.
* `timers.rolling` &mdash; Seconds spent animating the spinner before revealing a result.
* `timers.result` &mdash; Seconds the placeholder prize card stays on screen.
* `timers.cooldown` &mdash; Seconds to wait before returning to the idle camera feed.

## Running the mirror

Install the dependencies (OpenCV, Pygame, NumPy, and PyYAML), then launch the
mirror from the repository root:

```bash
pip install -r requirements.txt
python -m terp_mirror.mirror_app
```

### Optional CLI arguments

* `--monitor` &mdash; Override the monitor index from the configuration file.
* `--config` &mdash; Load settings from a different YAML config file.

The mirror runs fullscreen. During operation you can use the following hotkeys:

* <kbd>R</kbd> &mdash; Trigger a prize roll and show the spinner animation.
* <kbd>F</kbd> &mdash; Skip directly to the prize/result card.
* <kbd>Esc</kbd> / <kbd>Q</kbd> &mdash; Exit the application.
* <kbd>⌥ Option</kbd> (or <kbd>Alt</kbd>) &mdash; Toggle the operator controls panel.

## Controls panel (operator cheat-sheet)

The controls panel is visible by default. Press <kbd>Up</kbd>/<kbd>Down</kbd> to
select a control and <kbd>Left</kbd>/<kbd>Right</kbd> to adjust its value while
the mirror is running. Hold <kbd>Shift</kbd> with <kbd>Left</kbd>/<kbd>Right</kbd>
to jump in five-step increments. The detection controls let you
fine-tune how strict the wave detector behaves without restarting the app:

* **Mirror output** &mdash; Flip the projector view horizontally without restarting
  the app. Handy while dialing in calibration so you can confirm text and
  prompts read correctly.
* **Min contour area** &mdash; Ignore blobs smaller than this number of pixels.
* **Min circularity** &mdash; Reject candidates whose contour circularity
  (`4πA / P²`) falls below the threshold. Rounder hands or props score closer to
  `1.0`, while elongated streaks land nearer to `0.0`. Raise the threshold to
  filter out sleeves or stray limbs; lower it if legitimate waves fail to
  register. The live debug overlay shows the measured circularity so you can
  observe how tweaks affect detection.
* **Processing interval (s)** &mdash; Minimum delay between full contour analyses.
  Increase it to lighten the per-frame workload (helpful when the camera feed
  flickers after enabling circularity checks). Keep it at or below ~0.05&nbsp;s so
  the detector still samples motion often enough to track deliberate waves.

Recent builds fix the controls so every ROI slider reliably updates the
rectangle while keeping it inside the camera frame. Dial in the bounds first,
then fine-tune the HSV thresholds to match the calibrated region.

Enable the detector debug overlay with <kbd>Tab</kbd> to confirm that the ROI
and HSV mask continue to frame the operator correctly after any adjustments.
While the overlay is visible you can use the number-row hotkeys to tweak the
HSV thresholds (<kbd>1</kbd>/<kbd>2</kbd> adjust the hue lower bound, etc.). Hold
<kbd>Shift</kbd> with those hotkeys to apply a ×5 adjustment when you need bigger
swings quickly.

> **Note:** The requirements pin NumPy to the `1.26` series so it remains
> compatible with the tested OpenCV (4.9.x) and SciPy (1.10.x) wheels. Adjust the
> dependencies together if you change one.
