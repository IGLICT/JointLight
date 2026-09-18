"""Render aligned shading using MoGe geometry and a target environment map.

Adapted from the project's MoGe/moge/scripts/moge_mitsuba.py integration.
MoGe copyright (c) Microsoft Corporation; see ../third_party/MoGe/LICENSE.
Changes: standalone video-to-MP4 CLI, portable envmap path, no debug geometry,
no fixed 85-frame cap, temporary render frames and fail-fast rendering errors.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', '-i', required=True, help='Prepared source video')
    parser.add_argument('--output', '-o', required=True, help='Output shading MP4, or PNG for a single-frame preview')
    parser.add_argument('--env', required=True, help='HDR/EXR environment map')
    parser.add_argument('--pretrained', default='Ruicheng/moge-2-vitl-normal', help='MoGe checkpoint path or model identifier')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--envmap_angle', type=float, default=0.0)
    parser.add_argument('--rotate', choices=['NONE', 'CW', 'CCW'], default='NONE')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--fov_x', type=float, default=None)
    parser.add_argument('--resolution_level', type=int, choices=range(10), default=9)
    parser.add_argument('--num_tokens', type=int, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--exposure_ev', default='auto',
                        help='auto: meter the whole clip and protect highlights; or a fixed numeric EV (e.g. -1)')
    parser.add_argument('--spp', type=int, default=64, help='Actual rendering samples per pixel')
    parser.add_argument('--save_exr', action='store_true', help='Also retain raw linear RGBA, before exposure')
    parser.add_argument('--frame_index', type=int, default=None, help='Zero-based source frame for PNG output (default: 0)')
    args = parser.parse_args(argv)
    if args.exposure_ev != 'auto':
        try:
            args.exposure_ev = float(args.exposure_ev)
        except ValueError:
            parser.error('--exposure_ev must be auto or a number')
        if not math.isfinite(args.exposure_ev) or not -20 <= args.exposure_ev <= 20:
            parser.error('--exposure_ev must be finite and between -20 and 20')
    if args.spp < 1:
        parser.error('--spp must be positive')
    if Path(args.output).suffix.lower() not in ('.mp4', '.png'):
        parser.error('--output must end in .mp4 or .png')
    if args.frame_index is not None and (args.frame_index < 0 or Path(args.output).suffix.lower() != '.png'):
        parser.error('--frame_index requires PNG output and a non-negative index')
    if args.device == 'cpu' and args.fp16:
        parser.error('--fp16 is only supported for GPU inference')
    return args


class AutoExposureMeter:
    """Bounded-memory, whole-clip metering of visible linear surface radiance."""

    def __init__(self):
        self.edges = np.linspace(-32.0, 32.0, 4097)
        self.luminance_hist = np.zeros(4096, dtype=np.int64)
        self.peak_hist = np.zeros(4096, dtype=np.int64)

    def update(self, linear):
        if not np.isfinite(linear).all():
            raise ValueError('Renderer produced non-finite radiance')
        rgb = np.maximum(linear[..., :3], 0)
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
        visible = linear[..., 3] > 0.99 if linear.shape[-1] == 4 else np.ones(linear.shape[:2], dtype=bool)
        visible &= luminance > 1e-8
        for values, hist in ((luminance[visible], self.luminance_hist),
                             (rgb.max(axis=-1)[visible], self.peak_hist)):
            log_values = np.clip(np.log2(values), self.edges[0], self.edges[-1])
            hist += np.histogram(log_values, bins=self.edges)[0]

    def resolve(self):
        if self.luminance_hist.sum() == 0:
            return 0.0, {'metered_pixels': 0, 'reason': 'No visible non-black surface pixels'}

        def quantile(hist, q):
            index = np.searchsorted(np.cumsum(hist), q * hist.sum())
            # Use the upper bin edge; metering resolution is 1/64 EV.
            return float(2.0 ** self.edges[min(index + 1, len(self.edges) - 1)])

        median = quantile(self.luminance_hist, 0.5)
        highlight = quantile(self.peak_hist, 0.995)
        gain = min(0.35 / median, 0.95 / highlight)
        ev = float(np.clip(np.log2(gain), -8.0, 8.0))
        return ev, {'metered_pixels': int(self.luminance_hist.sum()),
                    'median_linear_luminance': median, 'peak_channel_p99_5': highlight,
                    'target_median_luminance': 0.35, 'target_highlight': 0.95,
                    'exposure_ev': ev, 'ev_limits': [-8, 8]}


def compress_highlights(rgb):
    """Smooth shoulder above 0.8, scaling RGB together to preserve channel ratios."""
    peak = np.maximum(rgb.max(axis=-1, keepdims=True), 0)
    mapped = 0.8 + 0.2 * (-np.expm1(-np.maximum(peak - 0.8, 0) / 0.2))
    gain = np.where(peak > 0.8, mapped / np.maximum(peak, 1e-8), 1.0)
    return rgb * gain


def write_shading_frame(path, rendered, exposure_ev, exr_path=None, tone_map=False):
    """Apply one fixed linear RGB gain, preserving raw radiance and alpha in EXR."""
    linear = np.array(rendered, dtype=np.float32)
    if not np.isfinite(linear).all():
        raise ValueError('Renderer produced non-finite radiance')
    rgb = linear[..., :3] * (2.0 ** exposure_ev)
    visible = linear[..., 3] > 0.99 if linear.shape[-1] == 4 else np.ones(linear.shape[:2], dtype=bool)
    pre_tonemap_clipped = float((rgb[visible] > 1).any(axis=-1).mean()) if visible.any() else 0.0
    if tone_map:
        rgb = compress_highlights(rgb)
    samples = rgb[visible]
    stats = {
        'exposure_ev': exposure_ev,
        'tone_mapping': 'highlight_shoulder' if tone_map else 'none',
        'pre_tonemap_clipped_any_channel_fraction': pre_tonemap_clipped,
        'visible_pixels': int(visible.sum()),
        'clipped_any_channel_fraction': float((samples > 1).any(axis=-1).mean()) if samples.size else 0.0,
        'clipped_all_channels_fraction': float((samples > 1).all(axis=-1).mean()) if samples.size else 0.0,
        'exposed_rgb_p99': float(np.quantile(samples, 0.99)) if samples.size else 0.0,
        'exposed_rgb_max': float(rgb.max()),
    }
    if exr_path is not None:
        Path(exr_path).parent.mkdir(parents=True, exist_ok=True)
        mi.util.write_bitmap(str(exr_path), linear, write_async=False)
    # write_bitmap performs the single linear-to-sRGB conversion. No manual gamma.
    # Synchronous writes ensure FFmpeg never sees an incomplete final PNG.
    mi.util.write_bitmap(str(path), rgb, write_async=False)
    return stats


def images_to_video(image_folder, output_video, fps=30, image_format='png', start_number=0):
    """
    将符合 %03d 命名格式的图片序列合成为视频（如 001.png, 002.png, ...）。

    参数:
        image_folder (str): 图片所在文件夹路径。
        output_video (str): 输出视频的完整路径（如 '/path/to/video.mp4'）。
        fps (int): 视频帧率，默认 30。
        image_format (str): 图片扩展名，如 'png' 或 'jpg'，默认 'png'。
        start_number (int): 起始编号（对应 %03d 中的第一个数字），默认 1。
    """
    input_pattern = os.path.join(image_folder, f"render_%03d.{image_format}")

    cmd = [
        'ffmpeg',
        '-y',                          # 自动覆盖输出文件
        '-r', str(fps),                # 输入帧率
        '-start_number', str(start_number),
        '-i', input_pattern,           # 输入序列
        '-c:v', 'libx264',             # 视频编码器
        '-pix_fmt', 'yuv420p',         # 兼容性更好的像素格式（适用于大多数播放器）
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # 确保宽高为偶数（H.264 要求）
        output_video
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr.decode()}")

def update_envmap(envmap, to_world):
    """Keep Mitsuba's linear data and periodic seam column; update orientation only."""
    try:
        envmap_params = mi.traverse(envmap)
        envmap_params['to_world'] = to_world
        envmap_params.update()
        return True
    except Exception as e:
        print(f"更新 envmap 错误: {e}")
    
    return False

def update_mesh(mesh, mesh_trimesh):
    vertices = mesh_trimesh.vertices.astype(np.float32)  # shape (N, 3)
    faces = mesh_trimesh.faces.astype(np.uint32)         # shape (F, 3)
    uv = mesh_trimesh.visual.uv.astype(np.float32)
    uv[:,1] = 1 - uv[:,1]

    normals = mesh_trimesh.vertex_normals.astype(np.float32)
    vertex_positions = mi.Float(vertices.ravel())
    face_indices = mi.UInt32(faces.ravel())
    vertex_texcoords = mi.Float(uv.ravel())
    vertex_normals = mi.Float(normals.ravel())

    updated_vertex_positions = False
    updated_faces = False
    updated_vertex_normals = False
    updated_vertex_texcoords = False

    """更新 mesh"""
    try:
        params = mi.traverse(mesh)
        for key in params.keys():
            if 'vertex_positions' in key.lower():
                params[key] = vertex_positions
                params.set_dirty(key)
                updated_vertex_positions = True
            if 'faces' in key.lower():
                params[key] = face_indices
                params.set_dirty(key)
                updated_faces = True
            if 'vertex_normals' in key.lower():
                params[key] = vertex_normals
                params.set_dirty(key)
                updated_vertex_normals = True
            if 'vertex_texcoords' in key.lower():
                params[key] = vertex_texcoords
                params.set_dirty(key)
                updated_vertex_texcoords = True


        if updated_vertex_positions and updated_faces and updated_vertex_normals and updated_vertex_texcoords:
            params.update()
            return True
    except Exception as e:
        print(f"更新 mesh 错误: {e}")
    
    return False

def update_fov(fov_x, scene):
    try:
        params = mi.traverse(scene)
        for key in params.keys():
            if 'x_fov' in key.lower():
                params[key] = fov_x
                params.update()
                return True
    except Exception as e:
        print(f"更新 fov 错误: {e}")
    
    return False


def render_video(args):
    # CPU rendering enables single-frame validation on machines without a GPU.
    mi.set_variant('llvm_ad_rgb' if args.device == 'cpu' else 'cuda_ad_rgb')
    if args.device == 'cpu':
        dr.set_thread_count(8)
        torch.set_num_threads(8)
    model = import_model_class_by_version('v2').from_pretrained(args.pretrained).to(args.device).eval()
    if args.fp16:
        model.half()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(Path(args.input).expanduser()))
    if not cap.isOpened():
        raise ValueError(f'Cannot open input video: {args.input}')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        raise ValueError('Input video has no frames')
    preview = output_path.suffix.lower() == '.png'
    frame_index = args.frame_index if args.frame_index is not None else 0
    if preview and frame_index >= total_frames:
        cap.release()
        raise ValueError(f'Frame {frame_index} is outside video with {total_frames} frames')
    if preview:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    frame_stats = []
    auto_exposure = args.exposure_ev == 'auto'
    meter = AutoExposureMeter() if auto_exposure else None
    resolved_ev = args.exposure_ev
    metering_stats = None
    pending_frames = []
    try:
        with tempfile.TemporaryDirectory(prefix='shading-', dir=output_path.parent) as frame_dir:
            frame_indices = [frame_index] if preview else range(total_frames)
            for frame_count in tqdm(frame_indices, desc='Rendering shading'):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Failed to read input frame {frame_count}/{total_frames}')
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=args.device).permute(2, 0, 1)
                output = model.infer(image_tensor, fov_x=args.fov_x, resolution_level=args.resolution_level,
                                     num_tokens=args.num_tokens, use_fp16=args.fp16)
                points, depth, mask, intrinsics = (output[k].cpu().numpy() for k in ('points', 'depth', 'mask', 'intrinsics'))
                normal = output['normal'].cpu().numpy() if 'normal' in output else None
                fov_x, _ = utils3d.numpy.intrinsics_to_fov(intrinsics)
                fov_x = np.rad2deg(fov_x)
                threshold = args.threshold
                mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=threshold)
                if normal is None:
                    faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
                        points,
                        image.astype(np.float32) / 255,
                        utils3d.np.uv_map(height, width),
                        mask=mask_cleaned,
                        tri=True
                    )
                    vertex_normals = None
                else:
                    faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.np.build_mesh_from_map(
                        points,
                        image.astype(np.float32) / 255,
                        utils3d.np.uv_map(height, width),
                        normal,
                        mask=mask_cleaned,
                        tri=True
                    )
                vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
                if normal is not None:
                    vertex_normals = vertex_normals * [1, -1, -1]
                mesh_trimesh = trimesh.Trimesh(
                    vertices=vertices, vertex_normals=vertex_normals, faces=faces,
                    visual=trimesh.visual.texture.TextureVisuals(
                        uv=vertex_uvs,
                        material=trimesh.visual.material.PBRMaterial(
                            baseColorTexture=Image.fromarray(image), metallicFactor=0.5, roughnessFactor=1.0)),
                    process=False,
                )
                # Preserve half-resolution shading, original camera and path integrator.
                res_w, res_h = width // 2, height // 2
                spp, fov = args.spp, fov_x
                shading_material = {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': [0.5, 0.5, 0.5]
                    },
                    'id': 'shading_diffuse'
                }


                camera = {
                        "type": "perspective",
                        "fov_axis": "x",
                        "near_clip": 0.001,
                        "far_clip": 100.0,
                        "focus_distance": 1000.0,
                        "fov": fov,
                        "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 0],
                                                                target=[0, 0, -1],
                                                                up=[0, 1, 0]),
                        "film": {
                            "type": "hdrfilm",
                            "rfilter": {
                                "type": "tent"
                            },
                            "width": res_w,
                            "height": res_h,
                            "pixel_format": "rgba",
                            "component_format": "float32"
                        }, 
                        "sampler": {
                            "type": "independent",
                            "sample_count": spp,
                        },
                    }
                scene_mesh = mi.Mesh('scene_mesh', vertex_count=100, face_count=100)
                scene_mesh.set_bsdf(mi.load_dict(shading_material))
                scene = mi.load_dict({
                    'type': 'scene',
                    'integrator': {'type': 'path', 'max_depth': 3, 'hide_emitters': True},
                    'camera': camera,
                    'geometry': scene_mesh,
                    'environment': {
                        'type': 'envmap', 'filename': str(Path(args.env).expanduser().resolve()),
                        'scale': 1, 'id': 'hdri_light',
                        'to_world': mi.ScalarTransform4f.rotate([0, 1, 0], 0),
                    },
                })
                rotate_angle = args.envmap_angle
                if args.rotate == 'CW':
                    rotate_angle -= frame_count / total_frames * 360
                elif args.rotate == 'CCW':
                    rotate_angle += frame_count / total_frames * 360
                # Keep the reference MoGe/Mitsuba coordinate conversion. This
                # reflection belongs to the emitter transform used for
                # rendering; comparison.mp4 displays the source panorama by
                # default and does not apply this transform a second time.
                to_world = mi.ScalarTransform4f.rotate([0, 1, 0], rotate_angle) @ mi.ScalarTransform4f.scale([-1, 1, 1])
                if not update_envmap(scene.emitters()[0], to_world):
                    raise RuntimeError(f'Cannot update environment for frame {frame_count}')
                if not update_mesh(scene_mesh, mesh_trimesh):
                    raise RuntimeError(f'Cannot update mesh for frame {frame_count}')
                if not update_fov(fov_x, scene):
                    raise RuntimeError(f'Cannot update camera for frame {frame_count}')
                rendered = mi.render(scene, spp=args.spp)
                png_path = output_path if preview else Path(frame_dir) / f'render_{frame_count:03d}.png'
                exr_path = None
                if args.save_exr:
                    exr_path = (output_path.with_suffix('.exr') if preview else
                                output_path.with_suffix('') / 'linear' / f'render_{frame_count:03d}.exr')
                if auto_exposure:
                    linear = np.array(rendered, dtype=np.float32)
                    meter.update(linear)
                    raw_path = exr_path or Path(frame_dir) / f'linear_{frame_count:03d}.exr'
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    mi.util.write_bitmap(str(raw_path), linear, write_async=False)
                    pending_frames.append((frame_count, png_path, raw_path))
                else:
                    stats = write_shading_frame(png_path, rendered, resolved_ev, exr_path)
                    frame_stats.append({'frame_index': frame_count, **stats})
            if auto_exposure:
                resolved_ev, metering_stats = meter.resolve()
                print(f'Auto exposure: {resolved_ev:+.3f} EV for all {len(pending_frames)} frames', flush=True)
                for frame_count, png_path, raw_path in tqdm(pending_frames, desc='Encoding auto exposure'):
                    linear = np.array(mi.Bitmap(str(raw_path)), dtype=np.float32)
                    stats = write_shading_frame(png_path, linear, resolved_ev, tone_map=True)
                    frame_stats.append({'frame_index': frame_count, **stats})
            if not preview:
                images_to_video(frame_dir, str(output_path))
        output_path.with_suffix('.json').write_text(json.dumps({
            'input': str(Path(args.input).resolve()), 'env': str(Path(args.env).resolve()),
            'spp': args.spp, 'requested_exposure_ev': args.exposure_ev, 'exposure_ev': resolved_ev,
            'auto_exposure_metering': metering_stats,
            'envmap_angle': args.envmap_angle, 'rotate': args.rotate,
            'threshold': args.threshold, 'frames': frame_stats,
        }, indent=2) + '\n')
        clipped = max(s['clipped_any_channel_fraction'] for s in frame_stats)
        print(f'Maximum per-frame clipped-pixel fraction after exposure: {clipped:.2%}', flush=True)
    finally:
        cap.release()
    print(f'Saved shading cue: {output_path}', flush=True)


if __name__ == '__main__':
    args = parse_args()
    # Import optional GPU dependencies after parsing, so --help needs only Python.
    import drjit as dr
    import mitsuba as mi
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    from tqdm import tqdm
    import trimesh
    import trimesh.visual
    from moge.model import import_model_class_by_version
    import utils3d

    render_video(args)
