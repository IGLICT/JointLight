"""Prepare an input video, render its shading cue, and run JointLight relighting."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--envmap_path', help='Target HDR/EXR environment map; required unless --skip_moge')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--checkpoint', default='models/JointLight/JointLight.safetensors')
    parser.add_argument('--model_config', default='model_configs.json')
    parser.add_argument('--tokenizer_path', default='models/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl')
    parser.add_argument('--device', default='cuda', help='JointLight inference device')
    parser.add_argument('--input_video_path', help='Optional separate RGB initialization video')
    parser.add_argument('--height', type=int, help='Set together with --width; default follows video orientation')
    parser.add_argument('--width', type=int)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--shading_timestep_init', type=float, default=0.6)
    parser.add_argument('--input_timestep_init', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--envmap_angle', type=float, default=0.0, help='Initial yaw in degrees')
    parser.add_argument('--rotate', choices=['NONE', 'CW', 'CCW'], default='NONE')
    parser.add_argument(
        '--envmap_display_transform', choices=['source', 'mitsuba'], default='source',
        help=('Orientation of the envmap panel in comparison.mp4: source keeps '
              'the HDRI file orientation (default), mitsuba applies the '
              'renderer-side horizontal reflection.'),
    )
    parser.add_argument('--threshold', type=float, default=0.1, help='Relative depth-edge threshold')
    parser.add_argument('--shading_exposure_ev', default='auto',
                        help='auto: whole-clip exposure with highlight protection; or a fixed numeric EV')
    parser.add_argument('--shading_spp', type=int, default=64, help='Mitsuba samples per pixel')
    parser.add_argument('--save_shading_exr', action='store_true', help='Retain raw linear shading frames before exposure')
    parser.add_argument('--moge_checkpoint', help='Local MoGe weights or model identifier; defaults to MoGe v2')
    parser.add_argument('--moge_repo', help='Optional MoGe source checkout; otherwise use the installed package')
    parser.add_argument('--moge_env', help='Conda environment for MoGe/Mitsuba; otherwise use current Python')
    parser.add_argument('--infer_env', help='Conda environment for JointLight; otherwise use current Python')
    parser.add_argument('--conda_exe', default=os.environ.get('CONDA_EXE', 'conda'))
    parser.add_argument('--output_dir', default='outputs/inference')
    parser.add_argument('--output_subdir',
                        help='Optional run-specific subdirectory under --output_dir; defaults to the video filename')
    parser.add_argument('--skip_moge', action='store_true', help='Reuse --shading_cue_path without loading MoGe/Mitsuba')
    parser.add_argument('--shading_cue_path', help='Existing shading video for --skip_moge')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Do not save the five-panel input/shading/relit/envmap comparison video')
    parser.add_argument('--dry_run', action='store_true', help='Print stages/commands without writing files or running models')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_moge and not args.shading_cue_path:
        parser.error('--skip_moge requires --shading_cue_path')
    if not args.skip_moge and not args.envmap_path:
        parser.error('--envmap_path is required unless --skip_moge is set')
    if args.num_frames < 1 or args.num_frames % 4 != 1:
        parser.error('--num_frames must be positive and equal to 4n+1')
    if (args.height is None) != (args.width is None):
        parser.error('provide both --height and --width')
    if args.output_subdir and (Path(args.output_subdir).is_absolute() or '..' in Path(args.output_subdir).parts):
        parser.error('--output_subdir must stay within --output_dir')
    if args.height is not None and any(v <= 0 or v % 32 for v in (args.height, args.width)):
        parser.error('--height and --width must be positive multiples of 32')
    if not all(0 <= v <= 1 for v in (args.shading_timestep_init, args.input_timestep_init)):
        parser.error('noise strengths must lie in [0, 1]')
    if not math.isfinite(args.envmap_angle) or not math.isfinite(args.cfg_scale):
        parser.error('--envmap_angle and --cfg_scale must be finite')
    if args.shading_exposure_ev != 'auto':
        try:
            args.shading_exposure_ev = float(args.shading_exposure_ev)
        except ValueError:
            parser.error('--shading_exposure_ev must be auto or a number')
        if not math.isfinite(args.shading_exposure_ev) or not -20 <= args.shading_exposure_ev <= 20:
            parser.error('--shading_exposure_ev must be finite and between -20 and 20')
    if args.shading_spp < 1:
        parser.error('--shading_spp must be positive')
    if args.skip_moge and args.save_shading_exr:
        parser.error('--save_shading_exr requires rendering; raw radiance cannot be recovered from a shading MP4')
    return args


def resolve_paths(args):
    for key in ('video_path', 'envmap_path', 'checkpoint', 'model_config', 'tokenizer_path',
                'input_video_path', 'output_dir', 'shading_cue_path', 'moge_repo'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, str(Path(value).expanduser().resolve()))
    if args.moge_checkpoint and Path(args.moge_checkpoint).expanduser().exists():
        args.moge_checkpoint = str(Path(args.moge_checkpoint).expanduser().resolve())
    # Preserve executable names so PATH lookup works; only resolve explicit paths.
    if '/' in args.conda_exe:
        args.conda_exe = str(Path(args.conda_exe).expanduser().resolve())
    return args


def build_python_command(script_path, env_name, conda_exe):
    if env_name:
        return [conda_exe, 'run', '--no-capture-output', '-n', env_name, 'python', str(script_path)]
    return [sys.executable, str(script_path)]


def run_command(command, dry_run=False, extra_env=None):
    print(shlex.join(command), flush=True)
    if not dry_run:
        env = os.environ.copy()
        env.update(extra_env or {})
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def expected_inference_outputs(base, processed, shading, args):
    """Reconstruct the deterministic output paths used by inference.py."""
    video_name = '_'.join(Path(processed).parts[-3:])
    shading_name = '_'.join(Path(shading).parts[-3:])
    save_dir = (base / 'relit' / Path(args.checkpoint).stem /
                f'frames_{args.num_frames}' / f'cfg_{args.cfg_scale}' /
                f'shading_timestep_init_{args.shading_timestep_init}' /
                f'input_timestep_init_{args.input_timestep_init}' /
                f'hw_{args.height}_{args.width}' / video_name)
    relit = save_dir / shading_name
    output_shading = save_dir / shading_name.replace('.mp4', '_shading.mp4')
    return relit, output_shading


def detect_target_hw(video_path):
    from diffsynth.data.video import VideoData
    width, height = VideoData(video_path)[0].size
    return (480, 832) if width >= height else (832, 480)


def save_processed_video(video_path, output_path, height, width, num_frames, dry_run=False):
    print(f'Prepare {video_path} -> {output_path} ({height}x{width}, {num_frames} frames)', flush=True)
    if dry_run:
        return
    from diffsynth.data.video import VideoData, save_video
    data = VideoData(str(video_path), height=height, width=width)
    available_frames = len(data)
    if available_frames < 1:
        raise ValueError(f'{video_path}: video contains no frames')
    data.set_length(min(available_frames, num_frames))
    frames = data.raw_data()
    if len(frames) < num_frames:
        padding = num_frames - len(frames)
        print(f'Pad {video_path}: {len(frames)} -> {num_frames} frames by repeating the last frame {padding} times', flush=True)
        last_frame = frames[-1]
        frames.extend(last_frame.copy() for _ in range(padding))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(output_path), fps=30, quality=5)


def main(argv=None):
    args = resolve_paths(parse_args(argv))
    if not args.dry_run:
        required = [args.video_path, args.checkpoint, args.model_config]
        required += [args.shading_cue_path] if args.skip_moge else [args.envmap_path]
        if args.input_video_path:
            required.append(args.input_video_path)
        for path in required:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    height, width = (args.height, args.width) if args.height else detect_target_hw(args.video_path)
    # The inference-output lookup must use the same resolved dimensions.
    args.height, args.width = height, width
    run_subdir = args.output_subdir or Path(args.video_path).stem
    base = Path(args.output_dir) / run_subdir
    processed = base / 'assets' / 'input.mp4'
    save_processed_video(args.video_path, processed, height, width, args.num_frames, args.dry_run)

    if args.skip_moge:
        shading = Path(args.shading_cue_path)
    else:
        suffix = f'_{args.rotate}' if args.rotate != 'NONE' else ''
        shading = base / 'moge_shading' / (
            f'{Path(args.envmap_path).stem}_{args.envmap_angle}{suffix}_threshold_{args.threshold}'
            f'_ev_{args.shading_exposure_ev}_spp_{args.shading_spp}.mp4')
        if not args.dry_run:
            shutil.copy2(args.envmap_path, base / 'assets' / ('envmap' + Path(args.envmap_path).suffix))
        command = build_python_command(REPO_ROOT / 'scripts/generate_shading.py', args.moge_env, args.conda_exe)
        command += ['--input', str(processed), '--output', str(shading), '--env', args.envmap_path,
                    '--envmap_angle', str(args.envmap_angle), '--rotate', args.rotate,
                    '--threshold', str(args.threshold), '--device', args.device,
                    '--exposure_ev', str(args.shading_exposure_ev), '--spp', str(args.shading_spp)]
        if args.save_shading_exr:
            command.append('--save_exr')
        if args.moge_checkpoint:
            command += ['--pretrained', args.moge_checkpoint]
        env = {'OPENCV_IO_ENABLE_OPENEXR': '1'}
        if args.moge_repo:
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [args.moge_repo, os.environ.get('PYTHONPATH')]))
        run_command(command, args.dry_run, env)
        if not args.dry_run and not shading.is_file():
            raise FileNotFoundError(f'Shading renderer did not produce {shading}')

    command = build_python_command(REPO_ROOT / 'inference.py', args.infer_env, args.conda_exe)
    values = {'model_config': args.model_config, 'tokenizer_path': args.tokenizer_path,
              'checkpoint': args.checkpoint, 'device': args.device, 'video_path': processed,
              'shading_cue_paths': shading, 'prompt': args.prompt, 'num_frames': args.num_frames,
              'height': height, 'width': width, 'cfg_scale': args.cfg_scale,
              'shading_timestep_init': args.shading_timestep_init,
              'input_timestep_init': args.input_timestep_init, 'seed': args.seed,
              'output_dir': base / 'relit'}
    if args.input_video_path:
        initial = base / 'assets' / 'initial.mp4'
        save_processed_video(args.input_video_path, initial, height, width, args.num_frames, args.dry_run)
        values['input_video_path'] = initial
    for key, value in values.items():
        command += ['--' + key, str(value)]
    run_command(command, args.dry_run)
    print(f'Relighting output directory: {base / "relit"}', flush=True)

    relit_video, output_shading = expected_inference_outputs(base, processed, shading, args)
    if not args.dry_run:
        if not relit_video.is_file():
            raise FileNotFoundError(relit_video)
        (base / 'result.json').write_text(json.dumps({
            'input': str(processed), 'relit': str(relit_video),
            'envmap': args.envmap_path, 'rotate': args.rotate,
            'envmap_angle': args.envmap_angle, 'num_frames': args.num_frames,
        }, indent=2) + '\n')

    if not args.no_comparison and args.envmap_path:
        comparison = base / 'comparison.mp4'
        if not args.dry_run:
            for path in (relit_video, output_shading):
                if not path.is_file():
                    raise FileNotFoundError(f'Relighting stage did not produce {path}')
        comparison_command = build_python_command(REPO_ROOT / 'scripts/make_comparison.py', args.infer_env, args.conda_exe)
        comparison_command += [
            '--input', str(processed),
            '--moge_shading', str(shading),
            '--output_shading', str(output_shading),
            '--relit', str(relit_video),
            '--envmap', str(args.envmap_path),
            '--output', str(comparison),
            '--height', str(height), '--width', str(width),
            '--num_frames', str(args.num_frames),
            '--envmap_angle', str(args.envmap_angle), '--rotate', args.rotate,
            '--envmap_display_transform', args.envmap_display_transform,
        ]
        run_command(comparison_command, args.dry_run, {'OPENCV_IO_ENABLE_OPENEXR': '1'})
        if not args.dry_run and not comparison.is_file():
            raise FileNotFoundError(f'Comparison renderer did not produce {comparison}')
        print(f'Comparison video: {comparison}', flush=True)


if __name__ == '__main__':
    main()
