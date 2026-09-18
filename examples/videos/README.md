# Example videos

The release includes six videos, named `video_0.mp4` through `video_5.mp4`.
The project maintainer has confirmed that these six clips may be publicly
redistributed as examples. They are media assets; the source-code license does
not assign a new license to their underlying content.

| File | Width × height | Orientation |
| --- | --- | --- |
| video_0.mp4 | 480 × 832 | Portrait |
| video_1.mp4 | 832 × 480 | Landscape |
| video_2.mp4 | 832 × 480 | Landscape |
| video_3.mp4 | 832 × 480 | Landscape |
| video_4.mp4 | 480 × 832 | Portrait |
| video_5.mp4 | 832 × 480 | Landscape |

Inputs were resized with aspect-ratio-preserving center crops. The full pipeline
uses the requested frame prefix and repeats the last frame when a clip is short.
It writes 30 FPS output without audio. `configs/inference.json` explicitly selects
the videos and HDRIs to combine; edit it to select your own inputs.
