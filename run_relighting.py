"""Run configurable video/HDRI combinations, with one worker per visible GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from run_relit_pipeline import REPO_ROOT, build_parser, parse_args

PATH_OPTIONS = {'checkpoint', 'model_config', 'tokenizer_path', 'input_video_path',
                'output_dir', 'shading_cue_path', 'moge_repo'}


def check_keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f'{label}: unknown keys: {sorted(unknown)}')


def absolute_path(value, root):
    if not isinstance(value, str) or not value:
        raise ValueError('Paths must be non-empty strings')
    path = Path(value).expanduser()
    return str((root / path).resolve())


def entries(values, allowed, label, root):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{label} must be a non-empty list')
    result, names = [], set()
    for value in values:
        check_keys(value, allowed | {'name', 'path'}, label)
        path = absolute_path(value.get('path'), root)
        if not Path(path).is_file():
            raise ValueError(f'{label}: input file not found: {path}')
        name = value.get('name', Path(path).stem)
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError(f'{label}: name must be a simple directory name: {name!r}')
        if name in names:
            raise ValueError(f'{label}: duplicate name {name!r}; assign unique names')
        names.add(name)
        result.append({**value, 'path': path, 'name': name})
    return result


def option_arguments(options):
    actions = {a.dest: a for a in build_parser()._actions}
    arguments = []
    for key, value in options.items():
        if value is None:
            continue
        if isinstance(actions[key], argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be true or false')
            if value:
                arguments.append('--' + key)
        else:
            if isinstance(value, (dict, list, bool)):
                raise ValueError(f'{key} must be a string or number')
            if actions[key].type is int and not isinstance(value, int):
                raise ValueError(f'{key} must be an integer')
            arguments.extend(['--' + key, str(value)])
    # Reuse the single-run validation before launching any jobs.
    parse_args(arguments)
    return arguments


def load_jobs(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open() as stream:
        config = json.load(stream)
    check_keys(config, {'path_root', 'defaults', 'videos', 'hdris'}, 'config')
    root = Path(absolute_path(config.get('path_root', '.'), config_path.parent))
    allowed = {a.dest for a in build_parser()._actions} - {
        'help', 'video_path', 'envmap_path', 'output_subdir', 'dry_run'}
    defaults = config.get('defaults', {})
    check_keys(defaults, allowed, 'defaults')
    videos = entries(config.get('videos'), {'prompt', 'height', 'width', 'input_video_path'}, 'videos', root)
    hdris = entries(config.get('hdris'), {'rotate', 'envmap_angle', 'shading_exposure_ev', 'shading_spp'}, 'hdris', root)
    # Resolve built-in path defaults as well, independently of the caller's cwd.
    base_defaults = {a.dest: a.default for a in build_parser()._actions
                     if a.dest in PATH_OPTIONS and a.default is not None}
    jobs = []
    for video in videos:
        for hdri in hdris:
            options = {**base_defaults, **defaults}
            options.update({k: v for k, v in video.items() if k not in {'name', 'path'}})
            options.update({k: v for k, v in hdri.items() if k not in {'name', 'path'}})
            options.update(video_path=video['path'], envmap_path=hdri['path'],
                           output_subdir=f"{video['name']}/{hdri['name']}")
            for key in PATH_OPTIONS:
                if options.get(key) is not None:
                    options[key] = absolute_path(options[key], root)
            checkpoint = options.get('moge_checkpoint')
            if checkpoint is not None and not isinstance(checkpoint, str):
                raise ValueError('moge_checkpoint must be a local path or model identifier')
            if checkpoint and (checkpoint.startswith(('.', '/', '~')) or checkpoint.endswith(('.pt', '.pth'))):
                options['moge_checkpoint'] = absolute_path(checkpoint, root)
            jobs.append((options, option_arguments(options)))
    return jobs


def visible_gpus(override):
    spec = override if override is not None else os.environ.get('CUDA_VISIBLE_DEVICES')
    if spec is None:
        try:
            spec = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                text=True, stderr=subprocess.DEVNULL).strip().replace('\n', ',')
        except (OSError, subprocess.CalledProcessError):
            spec = '0'
    gpus = [part.strip() for part in spec.split(',')]
    if not all(gpus) or '-1' in gpus or len(set(gpus)) != len(gpus):
        raise ValueError('GPU list must contain distinct non-empty GPU indices or UUIDs')
    return gpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/inference.json')
    parser.add_argument('--gpus', help='Comma-separated physical GPU indices/UUIDs; defaults to CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dry_run', action='store_true', help='Validate inputs and print jobs without loading models or writing outputs')
    args = parser.parse_args(argv)
    try:
        jobs = load_jobs(args.config)
        gpus = visible_gpus(args.gpus)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'{len(jobs)} jobs on {len(gpus)} GPU(s): {", ".join(gpus)}', flush=True)

    def worker(index):
        gpu = gpus[index]
        for options, arguments in jobs[index::len(gpus)]:
            command = [sys.executable, str(REPO_ROOT / 'run_relit_pipeline.py'), *arguments]
            print(f'[GPU {gpu}] {options["output_subdir"]}: {shlex.join(command)}', flush=True)
            if args.dry_run:
                continue
            run_dir = Path(options['output_dir']) / options['output_subdir']
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'inference_config.json').write_text(json.dumps(options, indent=2) + '\n')
            subprocess.run(command, cwd=REPO_ROOT,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}, check=True)

    with ThreadPoolExecutor(max_workers=min(len(gpus), len(jobs))) as pool:
        futures = [pool.submit(worker, i) for i in range(min(len(gpus), len(jobs)))]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
