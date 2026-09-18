"""Create a five-panel video comparison for a relighting run.

Panels are written in this order: processed input, MoGe shading, predicted
shading, relit RGB output, and the environment map used for that frame.
"""

import argparse
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np


PANEL_NAMES = ('input', 'moge_shading', 'output_shading', 'relit', 'envmap')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Processed input video')
    parser.add_argument('--moge_shading', required=True, help='MoGe-rendered shading cue video')
    parser.add_argument('--output_shading', required=True, help='Shading predicted by JointLight')
    parser.add_argument('--relit', required=True, help='Relit RGB video')
    parser.add_argument('--envmap', required=True, help='HDR/EXR environment map')
    parser.add_argument('--output', required=True, help='Five-panel MP4 output')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--num_frames', type=int, required=True)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--envmap_angle', type=float, default=0.0)
    parser.add_argument('--rotate', choices=('NONE', 'CW', 'CCW'), default='NONE')
    parser.add_argument(
        '--envmap_display_transform', choices=('source', 'mitsuba'), default='source',
        help=('How to orient the panorama in the preview: source keeps the HDRI '
              'file orientation (default); mitsuba applies the renderer-side '
              'horizontal reflection for coordinate debugging.'),
    )
    args = parser.parse_args(argv)
    if args.height <= 0 or args.width <= 0 or args.num_frames < 1:
        parser.error('--height, --width and --num_frames must be positive')
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be a positive finite number')
    if Path(args.output).suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4')
    return args


def load_envmap(path):
    image = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot read environment map: {path}')
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f'Environment map must have at least three channels: {path}')
    image = image[..., :3].astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f'Environment map contains non-finite values: {path}')
    return image


def tone_map_envmap(linear_bgr):
    """Make a stable SDR preview while retaining HDR color relationships."""
    rgb = np.maximum(linear_bgr, 0.0)
    luminance = rgb @ np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
    visible = np.isfinite(luminance) & (luminance > 1e-8)
    if visible.any():
        median = float(np.quantile(luminance[visible], 0.5))
        peak = float(np.quantile(np.max(rgb[visible], axis=-1), 0.995))
        gain = min(0.35 / max(median, 1e-8), 0.95 / max(peak, 1e-8))
        ev = float(np.clip(np.log2(max(gain, 1e-8)), -8.0, 8.0))
        rgb = rgb * (2.0 ** ev)
    peak = np.maximum(rgb.max(axis=-1, keepdims=True), 0.0)
    mapped_peak = 0.8 + 0.2 * (-np.expm1(-np.maximum(peak - 0.8, 0.0) / 0.2))
    rgb *= np.where(peak > 0.8, mapped_peak / np.maximum(peak, 1e-8), 1.0)
    srgb = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(np.maximum(rgb, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def resize_panel(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def add_label(frame, text):
    output = frame.copy()
    bar_height = max(30, min(52, output.shape[0] // 14))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], bar_height), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.62, output, 0.38, 0)
    cv2.putText(output, text, (10, max(21, bar_height - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def open_video(path):
    cap = cv2.VideoCapture(str(Path(path).expanduser()))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    return cap


def render_comparison(args):
    caps = [open_video(getattr(args, key)) for key in ('input', 'moge_shading', 'output_shading', 'relit')]
    try:
        fps = args.fps
        if fps <= 0:
            fps = caps[-1].get(cv2.CAP_PROP_FPS) or 30.0
        envmap = tone_map_envmap(load_envmap(args.envmap))
        # Keep the source HDRI orientation in the comparison by default. The
        # Mitsuba emitter applies its coordinate reflection while rendering;
        # that transform is not re-applied to the flat panorama shown here.
        # The renderer-oriented view remains available explicitly for
        # coordinate debugging.
        if args.envmap_display_transform == 'mitsuba':
            envmap = cv2.flip(envmap, 1)
        env_height, env_width = envmap.shape[:2]
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_width = args.width * len(PANEL_NAMES)
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{output_width}x{args.height}', '-r', f'{fps:.8g}', '-i', '-',
            '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame_index in range(args.num_frames):
                panels = []
                for cap, name in zip(caps, PANEL_NAMES[:4]):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'{name}: expected {args.num_frames} frames, stopped at {frame_index}')
                    panels.append(add_label(resize_panel(frame, args.width, args.height), name))

                angle = args.envmap_angle
                if args.rotate == 'CW':
                    angle -= frame_index / args.num_frames * 360.0
                elif args.rotate == 'CCW':
                    angle += frame_index / args.num_frames * 360.0
                shift = int(round(angle / 360.0 * env_width)) % env_width
                env_frame = np.roll(envmap, shift, axis=1) if shift else envmap
                env_panel = resize_panel(env_frame, args.width, args.height)
                panels.append(add_label(env_panel, 'envmap'))
                process.stdin.write(np.concatenate(panels, axis=1).tobytes())
            process.stdin.close()
            error = process.stderr.read().decode('utf-8', errors='replace')
            return_code = process.wait()
        except Exception:
            if process.stdin:
                process.stdin.close()
            process.kill()
            process.wait()
            raise
        if return_code != 0:
            raise RuntimeError(f'FFmpeg failed while writing {output_path}: {error}')
        print(f'Saved five-panel comparison: {output_path}', flush=True)
    finally:
        for cap in caps:
            cap.release()


if __name__ == '__main__':
    render_comparison(parse_args())
