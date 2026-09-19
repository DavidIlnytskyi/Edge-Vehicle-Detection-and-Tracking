# Tracking preview provenance

These previews were derived from the supplied annotated videos. No inference was rerun, and predictions, track IDs, trajectories, and playback timing were retained. Full-video previews are resized to 960 pixels wide and encoded as H.264 for smaller, browser-compatible files.

| Preview | Source in `docs/tracking/` | Frames | Playback rate |
| --- | --- | ---: | ---: |
| `tracking_basketball.mp4` | `tracking_basketball.mp4` | 464 | 15 FPS |
| `tracking_basketball_blur.mp4` | `tracking_basketball_blur.mp4` | 464 | 15 FPS |
| `tracking_people_walk.mp4` | `tracking_people_walk.mp4` | 341 | 25 FPS |

`tracking_basketball.gif` and `tracking_people_walk.gif` each cover seconds 3–8 of their corresponding source clips at 640 pixels wide and 8 animation frames per second, preserving elapsed playback time. Playback FPS is not inference throughput. The original videos remain local; the compact previews and matching MOT prediction files can be committed with the README.

To reproduce an MP4 preview from the repository root (repeat with the other source filenames):

```bash
ffmpeg -i docs/tracking/tracking_basketball.mp4 \
  -vf scale=960:-2 -c:v libx264 -preset fast -crf 26 \
  -pix_fmt yuv420p -an -movflags +faststart \
  docs/previews/tracking_basketball.mp4
```

To reproduce an animated excerpt (repeat with `tracking_people_walk` for the second preview):

```bash
ffmpeg -ss 3 -t 5 -i docs/tracking/tracking_basketball.mp4 \
  -filter_complex '[0:v]fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' \
  -loop 0 docs/previews/tracking_basketball.gif
```
