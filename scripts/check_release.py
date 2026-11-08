"""Check source completeness and built distributions without GPU dependencies."""

import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ('diffsynth', 'utils', 'third_party/MoGe/moge')
REQUIRED = (
    'diffsynth/models/__init__.py',
    'diffsynth/data/video.py',
    'diffsynth/models/wan_video_modality_identifier.py',
    'third_party/MoGe/moge/model/dinov2/models/vision_transformer.py',
)


def source_files():
    return sorted(p for name in SOURCE_ROOTS for p in (ROOT / name).rglob('*.py'))


def check_source(check_git):
    for name in REQUIRED:
        assert (ROOT / name).is_file(), f'Missing release source: {name}'
    sources = source_files()
    for path in sources:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level or not node.module:
                continue
            parent = path.parent
            for _ in range(node.level - 1):
                parent = parent.parent
            target = parent.joinpath(*node.module.split('.'))
            assert target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), (
                f'{path.relative_to(ROOT)}:{node.lineno}: missing relative import {node.module}')
    vendor = ROOT / 'third_party/MoGe'
    for line in (vendor / 'SOURCE_FILES.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == digest, (
            f'MoGe source checksum mismatch: {name}')
    if check_git:
        tracked = set(subprocess.check_output(
            ['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
        missing = [str(p.relative_to(ROOT)) for p in sources if str(p.relative_to(ROOT)) not in tracked]
        assert not missing, f'Source files absent from Git index: {missing}'
    print(f'Source completeness, syntax and MoGe checksums OK ({len(sources)} Python files)')


def check_distribution(archive):
    if archive.suffix == '.whl':
        with zipfile.ZipFile(archive) as bundle:
            contents = {name: bundle.read(name) for name in bundle.namelist() if not name.endswith('/')}
        expected = [p for name in ('diffsynth', 'utils') for p in (ROOT / name).rglob('*.py')]
    else:
        with tarfile.open(archive) as bundle:
            contents = {member.name.split('/', 1)[1]: bundle.extractfile(member).read()
                        for member in bundle.getmembers() if member.isfile() and '/' in member.name}
        expected = source_files()
        for directory in ('configs', 'docs', 'scripts', 'examples'):
            expected.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                            and '__pycache__' not in p.parts and p.suffix != '.pyc')
        expected.extend(ROOT / name for name in (
            'README.md', 'CHANGELOG.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
            'constraints.txt', 'requirements.txt', 'requirements-shading.txt',
            'model_configs.json', 'CHECKPOINTS.sha256', 'train.py', 'inference.py',
            'run_relit_pipeline.py', 'run_relighting.py', 'third_party/MoGe/LICENSE',
            'third_party/MoGe/pyproject.toml', 'third_party/MoGe/SOURCE_FILES.sha256'))
    for path in expected:
        name = path.relative_to(ROOT).as_posix()
        assert name in contents, f'{archive.name}: missing {name}'
        assert contents[name] == path.read_bytes(), f'{archive.name}: stale {name}'
    for name in contents:
        parts = Path(name).parts
        assert parts[0] not in ('models', 'data', 'datasets', 'logs', 'wandb'), name
        assert not parts[0].startswith(('outputs', '.conda', '.venv')), name
        assert '__pycache__' not in parts, name
        assert Path(name).suffix not in ('.pyc', '.pth', '.pt', '.ckpt', '.safetensors'), name
    print(f'{archive.name}: complete, current, and free of runtime assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check_git', action='store_true', help='Require Python sources in the Git index')
    parser.add_argument('--dist', type=Path, nargs='*', default=[], help='Check built wheel/sdist contents')
    args = parser.parse_args()
    check_source(args.check_git)
    for archive in args.dist:
        check_distribution(archive)


if __name__ == '__main__':
    main()
