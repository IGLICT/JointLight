"""Save a MoGe single-image mesh and render a short textured camera orbit."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='RGB image used for MoGe reconstruction')
    parser.add_argument('--output_video', required=True, help='Rendered MP4 path')
    parser.add_argument('--mesh_output', help='Mesh prefix; writes .ply, .obj/.mtl and .glb when possible')
    parser.add_argument('--pretrained', default='Ruicheng/moge-2-vitl-normal')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=832)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--duration', type=float, default=1.0)
    parser.add_argument('--orbit_angle_deg', type=float, default=15.0,
                        help='Angular distance from the original view (degrees)')
    parser.add_argument('--orbit_degrees', type=float, default=360.0,
                        help='Azimuth travelled around the view direction (degrees)')
    parser.add_argument('--spp', type=int, default=16)
    parser.add_argument('--resolution_level', type=int, choices=range(10), default=9)
    parser.add_argument('--num_tokens', type=int, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--background', type=float, default=1.0,
                        help='Uniform environment radiance for textured preview')
    parser.add_argument('--texture', action='store_true',
                        help='Use the input image as a mesh texture; default is a uniform gray material')
    parser.add_argument('--exposure_ev', type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.width % 2 or args.height % 2:
        parser.error('--width and --height must be positive even values')
    if args.fps < 1 or args.duration <= 0:
        parser.error('--fps must be positive and --duration must be positive')
    if args.spp < 1:
        parser.error('--spp must be positive')
    if args.orbit_angle_deg < 0 or args.orbit_angle_deg >= 90:
        parser.error('--orbit_angle_deg must lie in [0, 90)')
    if not math.isfinite(args.exposure_ev) or not -20 <= args.exposure_ev <= 20:
        parser.error('--exposure_ev must be finite and between -20 and 20')
    if args.device == 'cpu' and args.fp16:
        parser.error('--fp16 is only supported on CUDA')
    return args


def build_mesh(model, image, utils3d, np, torch, trimesh, Image, args):
    image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255
    image_tensor = image_tensor.to(args.device)
    with torch.inference_mode():
        output = model.infer(image_tensor, resolution_level=args.resolution_level,
                             num_tokens=args.num_tokens, use_fp16=args.fp16)
    points, mask, intrinsics = (output[k].detach().cpu().numpy()
                                for k in ('points', 'mask', 'intrinsics'))
    normal = output['normal'].detach().cpu().numpy() if 'normal' in output else None
    points = np.asarray(points)
    mask = np.asarray(mask).astype(bool)
    height, width = image.shape[:2]
    mask_cleaned = mask & ~utils3d.np.depth_map_edge(points[..., 2], rtol=0.1)
    if normal is None:
        faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
            points, image.astype(np.float32) / 255, utils3d.np.uv_map(height, width),
            mask=mask_cleaned, tri=True)
        vertex_normals = None
    else:
        faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.np.build_mesh_from_map(
            points, image.astype(np.float32) / 255, utils3d.np.uv_map(height, width), normal,
            mask=mask_cleaned, tri=True)

    vertices = vertices.astype(np.float32) * [1, -1, -1]
    vertex_uvs = vertex_uvs.astype(np.float32) * [1, -1] + [0, 1]
    if vertex_normals is not None:
        vertex_normals = vertex_normals.astype(np.float32) * [1, -1, -1]
    # Limit the target estimate to the central image region, avoiding distant floor/background.
    yy, xx = np.mgrid[:height, :width]
    central = mask_cleaned & (xx >= width * 0.15) & (xx <= width * 0.85) & \
        (yy >= height * 0.15) & (yy <= height * 0.85)
    target_points = points[central]
    if target_points.size == 0:
        target_points = points[mask_cleaned]
    target = np.median(target_points.astype(np.float32) * [1, -1, -1], axis=0)
    fov_x = float(np.rad2deg(utils3d.numpy.intrinsics_to_fov(intrinsics)[0]))

    if args.texture:
        texture = Image.fromarray(image)
        visual = trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0))
    else:
        # Keep the geometry useful in standard viewers without shipping the source image.
        visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(np.array([[160, 160, 160, 255]], dtype=np.uint8), (len(vertices), 1)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals,
                           visual=visual, process=False)
    return mesh, target, fov_x, mask_cleaned, points


def save_mesh(mesh, prefix, image, metadata, with_texture=False):
    import numpy as np
    import trimesh
    prefix = Path(prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Save a self-contained mesh; texture export is opt-in.
    textured = mesh.copy()
    if with_texture:
        texture_path = prefix.with_name(prefix.name + '_texture.png')
        from PIL import Image
        Image.fromarray(image).save(texture_path)
        try:
            textured.visual.material.baseColorTexture = Image.fromarray(image)
        except Exception:
            pass
    color_visual = mesh.visual if hasattr(mesh.visual, 'vertex_colors') else mesh.visual.to_color()
    colors = np.clip(np.asarray(color_visual.vertex_colors), 0, 255).astype(np.uint8)
    color_mesh = mesh.copy()
    color_mesh.visual = trimesh.visual.ColorVisuals(color_mesh, vertex_colors=colors)
    textured.export(prefix.with_suffix('.glb'))
    if with_texture:
        textured.export(prefix.with_suffix('.obj'), include_texture=True)
    else:
        # Avoid a generic MTL file that could accidentally reference an old texture.
        from trimesh.exchange.obj import export_obj
        prefix.with_suffix('.obj').write_text(export_obj(
            color_mesh, include_normals=True, include_color=True, include_texture=False))
    color_mesh.export(prefix.with_suffix('.ply'))
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return prefix


def camera_orbit(target, distance, angle_deg, azimuth_deg, np, mi):
    """Return camera poses on a 15-degree cone around the original target ray."""
    v0 = -target / max(np.linalg.norm(target), 1e-6)  # target -> original camera
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    b1 = np.cross(v0, up)
    if np.linalg.norm(b1) < 1e-5:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        b1 = np.cross(v0, up)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(v0, b1)
    b2 /= np.linalg.norm(b2)
    delta = math.radians(angle_deg)
    phi = math.radians(azimuth_deg)
    ring = math.cos(phi) * b1 + math.sin(phi) * b2
    direction = math.cos(delta) * v0 + math.sin(delta) * ring
    camera = target + distance * direction
    return mi.ScalarTransform4f.look_at(origin=camera.tolist(), target=target.tolist(), up=up.tolist()), camera


def render(args):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import trimesh
    import utils3d
    import drjit as dr
    import mitsuba as mi
    from moge.model import import_model_class_by_version

    mi.set_variant('llvm_ad_rgb' if args.device == 'cpu' else 'cuda_ad_rgb')
    if args.device == 'cpu':
        dr.set_thread_count(min(16, os.cpu_count() or 8))
        torch.set_num_threads(min(16, os.cpu_count() or 8))
    input_path = Path(args.input).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    mesh_prefix = Path(args.mesh_output).expanduser() if args.mesh_output else output_video.with_suffix('') / 'moge_mesh'
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f'Cannot read image: {input_path}')
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    model = import_model_class_by_version('v2').from_pretrained(args.pretrained).to(args.device).eval()
    mesh, target, fov_x, mask, points = build_mesh(model, image, utils3d, np, torch, trimesh, Image, args)
    valid_points = points[mask]
    if valid_points.size == 0:
        raise ValueError('MoGe returned no valid mesh pixels')
    distance = float(np.linalg.norm(target))
    if distance < 1e-4:
        distance = float(np.median(np.linalg.norm(valid_points * [1, -1, -1], axis=-1)))
    metadata = {
        'input': str(input_path), 'image_width': int(image.shape[1]), 'image_height': int(image.shape[0]),
        'valid_pixels': int(mask.sum()), 'vertices': int(len(mesh.vertices)), 'faces': int(len(mesh.faces)),
        'fov_x_deg': fov_x, 'target_world': target.tolist(), 'camera_distance': distance,
        'orbit_angle_deg': args.orbit_angle_deg, 'orbit_degrees': args.orbit_degrees,
        'fps': args.fps, 'duration_sec': args.duration, 'render_width': args.width,
        'render_height': args.height, 'spp': args.spp, 'device': args.device,
    }
    metadata['material'] = 'image_texture' if args.texture else 'uniform_gray'
    save_mesh(mesh, mesh_prefix, image, metadata, with_texture=args.texture)
    print(f'Saved mesh: {mesh_prefix.with_suffix(".ply")}', flush=True)

    # Make Mitsuba's mesh from the same coordinates and UVs used by the saved mesh.
    mi_mesh = mi.Mesh('moge_mesh', vertex_count=len(mesh.vertices), face_count=len(mesh.faces),
                      has_vertex_normals=mesh.vertex_normals is not None,
                      has_vertex_texcoords=args.texture)
    params = mi.traverse(mi_mesh)
    params['vertex_positions'] = mi.Float(np.asarray(mesh.vertices, dtype=np.float32).ravel())
    params['faces'] = mi.UInt32(np.asarray(mesh.faces, dtype=np.uint32).ravel())
    if args.texture:
        params['vertex_texcoords'] = mi.Float(np.asarray(mesh.visual.uv, dtype=np.float32).ravel())
    if mesh.vertex_normals is not None:
        params['vertex_normals'] = mi.Float(np.asarray(mesh.vertex_normals, dtype=np.float32).ravel())
    params.update()
    if args.texture:
        bitmap = {'type': 'bitmap', 'filename': str(mesh_prefix.with_name(mesh_prefix.name + '_texture.png')),
                  'filter_type': 'bilinear', 'raw': False}
        reflectance = bitmap
    else:
        reflectance = {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}
    mi_mesh.set_bsdf(mi.load_dict({'type': 'diffuse', 'reflectance': reflectance}))
    orbit_frames = max(1, int(round(args.fps * args.duration)))
    film = {'type': 'hdrfilm', 'width': args.width, 'height': args.height,
            'pixel_format': 'rgba', 'component_format': 'float32',
            'rfilter': {'type': 'tent'}}
    scene = mi.load_dict({
        'type': 'scene', 'integrator': {'type': 'path', 'max_depth': 2}, 'moge_mesh': mi_mesh,
        'sensor': {'type': 'perspective', 'fov_axis': 'x', 'fov': fov_x,
                   'near_clip': 0.001, 'far_clip': max(100.0, distance * 20), 'film': film,
                   'sampler': {'type': 'independent', 'sample_count': args.spp}},
        'environment': {'type': 'constant', 'radiance': {'type': 'rgb',
                                                          'value': [args.background] * 3}},
    })
    with tempfile.TemporaryDirectory(prefix=output_video.stem + '_frames-', dir=output_video.parent) as temp:
        temp_root = Path(temp)
        for i in range(orbit_frames):
            transform, camera = camera_orbit(target, distance, args.orbit_angle_deg,
                                             args.orbit_degrees * i / orbit_frames, np, mi)
            params = mi.traverse(scene)
            params['sensor.to_world'] = transform
            params.update()
            rendered = mi.render(scene, spp=args.spp)
            rgb = np.asarray(rendered)[..., :3] * (2 ** args.exposure_ev)
            mi.util.write_bitmap(str(temp_root / f'frame_{i:03d}.png'), rgb, write_async=False)
            if i == 0:
                metadata['first_camera_world'] = camera.tolist()
        # Encode exactly orbit_frames / fps seconds.
        result = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                                 '-i', str(temp_root / 'frame_%03d.png'), '-c:v', 'libx264',
                                 '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_video)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
    metadata['output_video'] = str(output_video)
    mesh_prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved orbit video: {output_video}', flush=True)


if __name__ == '__main__':
    # Delay optional imports so --help works without CUDA/MoGe dependencies.
    args = parse_args()
    import numpy as np
    import trimesh
    render(args)
