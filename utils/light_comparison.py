"""Compare relit videos with synchronized, exposure-stable chrome-ball previews."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

BALL_SIZE = 288


def rotation_angles(rotate, initial_angle, frames):
    direction = {'NONE': 0, 'CW': -1, 'CCW': 1}[rotate]
    return [initial_angle + direction * i / frames * 360 for i in range(frames)]


def chrome_frames(envmap, angles, cache_dir):
    """Use the same reflected emitter and yaw schedule as generate_shading.py."""
    import mitsuba as mi

    size = BALL_SIZE
    signature = {
        'envmap_sha256': hashlib.sha256(envmap.read_bytes()).hexdigest(),
        'angles': angles, 'size': size, 'spp': 256, 'version': 1,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'{key}.npz'
    if cache.is_file():
        with np.load(cache) as data:
            return data['rgba']
    scene = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'sphere', 'radius': 1.0,
            'bsdf': {'type': 'conductor', 'material': 'none',
                     'specular_reflectance': {'type': 'rgb', 'value': [1, 1, 1]}},
        },
        'envmap': {
            'type': 'envmap', 'filename': str(envmap),
            'to_world': mi.ScalarTransform4f.rotate([0, 1, 0], angles[0])
                        @ mi.ScalarTransform4f.scale([-1, 1, 1]),
        },
        'sensor': {
            'type': 'perspective', 'fov': 45.0,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 2.617], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': size, 'height': size,
                     'pixel_format': 'rgba', 'component_format': 'float32'},
            'sampler': {'type': 'independent', 'sample_count': 256},
        },
        'integrator': {'type': 'path', 'max_depth': 6, 'hide_emitters': True},
    })
    params = mi.traverse(scene.emitters()[0])
    frames = []
    render_angles = angles if len(set(angles)) > 1 else angles[:1]
    for index, angle in enumerate(render_angles):
        params['to_world'] = (mi.ScalarTransform4f.rotate([0, 1, 0], angle)
                              @ mi.ScalarTransform4f.scale([-1, 1, 1]))
        params.update()
        frames.append(np.array(mi.render(scene, seed=1, spp=256), dtype=np.float32))
        if index % 10 == 0:
            print(f'{envmap.stem}: chrome ball {index + 1}/{len(render_angles)}', flush=True)
    linear = np.stack(frames)
    y, x = np.ogrid[:size, :size]
    circle = (x - size // 2) ** 2 + (y - size // 2) ** 2 <= (size // 2) ** 2
    alpha = np.clip(linear[..., 3], 0, 1) * circle
    rgb = np.maximum(np.nan_to_num(linear[..., :3], nan=0, posinf=0, neginf=0), 0)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    valid = (alpha > 0.99) & (luminance > 1e-8)
    # One gain for the entire rotation avoids exposure flicker; preview only.
    exposure_ev = 0.0
    if valid.any():
        gain = min(0.35 / float(np.median(luminance[valid])),
                   0.95 / float(np.quantile(rgb.max(-1)[valid], 0.995)))
        exposure_ev = float(np.clip(np.log2(gain), -8, 8))
    rgb *= 2 ** exposure_ev
    peak = rgb.max(-1, keepdims=True)
    shoulder = 0.8 + 0.2 * (-np.expm1(-np.maximum(peak - 0.8, 0) / 0.2))
    rgb *= np.where(peak > 0.8, shoulder / np.maximum(peak, 1e-8), 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
    rgba = np.concatenate([
        np.clip(srgb * 255, 0, 255).astype(np.uint8),
        np.round(alpha[..., None] * 255).astype(np.uint8),
    ], axis=-1)
    np.savez_compressed(cache, rgba=rgba)
    cache.with_suffix('.json').write_text(json.dumps({
        **signature, 'preview_exposure_ev': exposure_ev,
    }, indent=2) + '\n')
    return rgba


def result_paths(options):
    """Read the exact result manifest; support older outputs without a manifest."""
    folder = Path(options['output_dir']) / options['output_subdir']
    manifest = folder / 'result.json'
    if manifest.is_file():
        record = json.loads(manifest.read_text())
        source, relit = Path(record['input']), Path(record['relit'])
    else:
        source = folder / 'assets/input.mp4'
        relit = folder / 'relit.mp4'
        if not relit.is_file():
            candidates = [p for p in (folder / 'relit').rglob('*.mp4')
                          if p.is_file() and not p.name.endswith('_shading.mp4')]
            if len(candidates) != 1:
                raise ValueError(f'{folder}: expected one relit video, found {len(candidates)}')
            relit = candidates[0]
        metadata = list((folder / 'moge_shading').glob('*.json'))
        if len(metadata) != 1:
            raise ValueError(f'{folder}: cannot determine the shading rotation metadata')
        record = json.loads(metadata[0].read_text())
        record['num_frames'] = len(record['frames'])
        record['envmap'] = record['env']
    for key in ('rotate', 'envmap_angle', 'num_frames'):
        if record[key] != options[key]:
            raise ValueError(f'{folder}: saved {key} differs from the config')
    saved_env = Path(record['envmap'])
    requested_env = Path(options['envmap_path'])
    if not saved_env.is_file() or saved_env.read_bytes() != requested_env.read_bytes():
        raise ValueError(f'{folder}: saved HDRI differs from the config or is missing')
    for path in (source, relit):
        if not path.is_file():
            raise FileNotFoundError(path)
    return source, relit


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot read video: {path}')
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def compose(paths, labels, balls, output, metadata):
    width, height, frames, fps = video_info(paths[0])
    size = min(BALL_SIZE, width - 16)
    if size < 1 or frames < 1 or fps <= 0:
        raise ValueError(f'Invalid video dimensions or timing: {paths[0]}')
    top, bottom = 36, size + 32
    canvas_width = width * len(paths)
    canvas_height = top + height + bottom
    # H.264 yuv420p requires even dimensions.
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.mp4')
    command = [
        'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{canvas_width}x{canvas_height}', '-r', str(fps), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary),
    ]
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    poster = None
    succeeded = False
    try:
        with output.with_suffix('.ffmpeg.log').open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for index in range(frames):
                    canvas = np.full((canvas_height, canvas_width, 3), 20, np.uint8)
                    for column, capture in enumerate(captures):
                        ok, frame = capture.read()
                        if not ok:
                            raise ValueError(f'Missing frame {index}: {paths[column]}')
                        canvas[top:top + height, column * width:(column + 1) * width] = frame
                        label = labels[column]
                        scale = min(0.8, (width - 16) / max(1, cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
                        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                        cv2.putText(canvas, label,
                                    (column * width + (width - text_width) // 2, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                                    (235, 235, 235), 1, cv2.LINE_AA)
                        if column:
                            sequence = balls[column - 1]
                            ball = sequence[index if len(sequence) > 1 else 0]
                            if size != BALL_SIZE:
                                ball = cv2.resize(ball, (size, size), interpolation=cv2.INTER_AREA)
                            alpha = ball[..., 3:] / 255.0
                            x = column * width + (width - size) // 2
                            y = top + height + 16
                            patch = canvas[y:y + size, x:x + size]
                            patch[:] = np.round(ball[..., :3][..., ::-1] * alpha
                                                + patch * (1 - alpha)).astype(np.uint8)
                    if index == 0:
                        poster = canvas.copy()
                    process.stdin.write(canvas.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f'FFmpeg failed: {output.with_suffix(".ffmpeg.log")}')
                succeeded = True
            except BaseException:
                process.kill()
                process.wait()
                raise
    finally:
        for capture in captures:
            capture.release()
        if temporary.exists() and not succeeded:
            temporary.unlink()
    temporary.replace(output)
    cv2.imwrite(str(output.with_suffix('.jpg')), poster)
    output.with_suffix('.json').write_text(json.dumps({
        **metadata, 'frames': frames, 'fps': fps,
        'resolution': [canvas_width, canvas_height],
        'source_videos': [str(path) for path in paths],
    }, indent=2) + '\n')
    print(f'Lighting comparison: {output}', flush=True)


def make_light_comparisons(jobs):
    import drjit as dr
    import mitsuba as mi
    from run_relit_pipeline import parse_args

    mi.set_variant('llvm_ad_rgb')
    dr.set_thread_count(8)
    groups = {}
    for _, arguments in jobs:
        options = vars(parse_args(arguments))
        root = Path(options['output_dir'])
        video = Path(options['output_subdir']).parts[0]
        groups.setdefault((root, video), []).append(options)
    for (root, video), group in groups.items():
        results = [result_paths(options) for options in group]
        source = results[0][0]
        reference = video_info(source)
        if reference[2] != group[0]['num_frames']:
            raise ValueError(f'{source}: frame count differs from config')
        source_digest = hashlib.sha256(source.read_bytes()).digest()
        for original, relit in results:
            if hashlib.sha256(original.read_bytes()).digest() != source_digest:
                raise ValueError(f'{video}: processed inputs differ between lighting runs')
            info = video_info(relit)
            if info[:3] != reference[:3] or abs(info[3] - reference[3]) > 0.01:
                raise ValueError(f'Video timing/dimensions mismatch: {relit}')
        labels, balls, schedules = ['Original'], [], {}
        for options in group:
            name = Path(options['output_subdir']).name
            rotate = options['rotate']
            labels.append(name + (f' / {rotate} 360 deg' if rotate != 'NONE' else ''))
            schedule = rotation_angles(rotate, options['envmap_angle'], reference[2])
            schedules[name] = schedule
            balls.append(chrome_frames(Path(options['envmap_path']), schedule,
                                       root / 'four_lights/chrome_balls'))
        suffix = 'four_lights' if len(group) == 4 else 'lights'
        compose([source] + [relit for _, relit in results], labels, balls,
                root / 'four_lights' / f'{video}_{suffix}.mp4',
                {'video': video, 'columns': labels, 'light_angles': schedules})
