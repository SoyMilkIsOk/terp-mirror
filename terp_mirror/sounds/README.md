# Sound assets

Place the following audio files in this directory for runtime playback:

- `wand.wav`
- `roll.wav`
- `prize.wav`
- `grand.wav`

Each file should be a short, mono or stereo WAV clip compatible with `pygame.mixer`. The
application will automatically load them if present; missing files simply disable the
corresponding sound cue at runtime.
