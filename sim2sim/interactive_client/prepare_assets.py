"""Stage the exported robot and real ONNX policies for a static browser build."""
from pathlib import Path
import argparse
import json
import shutil

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parents[1] / 'build' / 'demo'


def stage_policies(source, destination):
    manifest = json.loads((source / 'manifest.json').read_text())
    paths = {'manifest.json'}
    for policy in manifest['policies'].values():
        paths.add(policy['metadata'])
        paths.update(policy['files'].values())
    for name in paths:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    for name in sorted(paths):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    # These files are generated for the browser. Retire old checkpoints only
    # after the complete active policy set has been copied successfully.
    for path in destination.rglob('*'):
        if path.is_file() and path.relative_to(destination).as_posix() not in paths:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, default=DELIVERY, help='Directory containing exported policies/ and sceneassets/')
    parser.add_argument('--public', type=Path, default=ROOT/'public', help='Vite public assets directory')
    args = parser.parse_args()
    public = args.public
    public.mkdir(exist_ok=True)
    for directory in ['policies', 'sceneassets', 'runtime']:
        (public / directory).mkdir(exist_ok=True)
    stage_policies(args.delivery / 'policies', public / 'policies')
    scene_source = args.delivery / 'sceneassets'
    if not (scene_source / 'manifest.json').exists():
        raise FileNotFoundError('Run export_scene.py before staging browser assets.')
    shutil.copytree(scene_source, public / 'sceneassets', dirs_exist_ok=True)
    wasm = 'ort-wasm-simd-threaded.wasm'
    shutil.copy2(ROOT / 'node_modules' / 'onnxruntime-web' / 'dist' / wasm, public / 'runtime' / wasm)
    shutil.copy2(ROOT / 'node_modules' / '@mujoco' / 'mujoco' / 'mujoco.wasm', public / 'runtime' / 'mujoco.wasm')
    notices = []
    for package in ['@mujoco/mujoco', 'onnxruntime-web', 'onnxruntime-common', 'three']:
        folder = ROOT / 'node_modules' / package
        found = False
        for name in ['LICENSE', 'LICENSE.txt', 'LICENSE.md']:
            if (folder / name).exists():
                notices.append(f'{package}\n' + (folder / name).read_text())
                found = True
                break
        if not found and package == '@mujoco/mujoco':
            # The official npm distribution omits the engine's license file.
            # Preserve the exact license from the matching upstream release.
            notices.append(f'{package} 3.13.0\n' + (ROOT / 'licenses' / 'mujoco-3.13.0.txt').read_text())
    # Scene content credits travel with the assets that embed them: the YCB carton scan when it is in the staged set.
    manifest_path = public / 'sceneassets' / 'manifest.json'
    if manifest_path.exists() and 'meshes/cracker_box_visual.obj' in json.loads(manifest_path.read_text()).get('assets', {}):
        notices.append('YCB Object and Model Set - 003_cracker_box (Calli et al. 2015)\n' + (ROOT / 'licenses' / 'ycb-003_cracker_box.txt').read_text())
    (public / 'THIRD_PARTY_NOTICES.txt').write_text('\n\n'.join(notices))
    total = sum(p.stat().st_size for p in public.rglob('*') if p.is_file())
    print(f'Staged static assets: {total / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
