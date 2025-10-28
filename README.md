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
* `mirror` &mdash; When `true`, the feed is mirrored horizontally.
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

> **Note:** The requirements pin NumPy to the `1.26` series so it remains
> compatible with the tested OpenCV (4.9.x) and SciPy (1.10.x) wheels. Adjust the
> dependencies together if you change one.
