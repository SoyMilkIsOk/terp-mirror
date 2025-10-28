# terp-mirror

Projection mirror code for the Terp Wizard - Terpscoops Halloween booth.

## Configuration

Runtime settings are stored in [`config.yaml`](config.yaml) at the project root.
You can adjust the projector monitor, camera, image rotation, and mirroring
options without modifying the code.

```yaml
target_fps: 30
monitor_index: 0
rotate_deg: 0
mirror: true
camera:
  index: 0
```

* `monitor_index` &mdash; Which projector/monitor the fullscreen output should use
  (Pygame display index).
* `rotate_deg` &mdash; Rotation applied to each frame. Supported values are
  `0`, `90`, `-90`, `180`, and `-180` degrees.
* `mirror` &mdash; When `true`, the feed is mirrored horizontally.
* `camera.index` &mdash; The OpenCV camera index to open.
* `target_fps` &mdash; Desired update rate for the projector display.

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

The mirror runs fullscreen; press <kbd>Esc</kbd> or <kbd>Q</kbd> to exit.

> **Note:** The requirements pin NumPy to the `1.26` series so it remains
> compatible with the tested OpenCV (4.9.x) and SciPy (1.10.x) wheels. Adjust the
> dependencies together if you change one.
