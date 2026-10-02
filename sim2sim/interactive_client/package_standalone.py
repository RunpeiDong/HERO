"""Embed a built demo and compressed assets in a double-clickable HTML file."""
from pathlib import Path
import argparse
import base64
import gzip
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parent
DELIVERY = ROOT.parents[1] / 'build' / 'demo'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DELIVERY)
    args = parser.parse_args()
    delivery = args.output
    dist = ROOT / 'dist'
    html = (dist / 'index.html').read_text()
    scripts = re.findall(r'<script\b[^>]*src="([^"]+)"[^>]*></script>', html)
    if len(scripts) != 1:
        raise ValueError('The standalone build must have one bundled entry script.')
    entry = dist / scripts[0].removeprefix('./')
    script = entry.read_text()
    # A classic worker is embedded by Vite's ?worker&inline import. All other
    # runtime assets are requested through the explicit in-memory resolver.
    if re.search(r'\bfrom\s*["\']\./', script):
        raise ValueError('External JavaScript chunks must be bundled before packaging.')
    css_paths = re.findall(r'<link\b[^>]*href="([^"]+\.css)"[^>]*>', html)
    for path in css_paths:
        css = (dist / path.removeprefix('./')).read_text()
        html = re.sub(r'<link\b[^>]*href="' + re.escape(path) + r'"[^>]*>', lambda _: '<style>' + css + '</style>', html)
    assets = {}
    for folder in ['policies', 'sceneassets', 'runtime']:
        for path in sorted((dist / folder).rglob('*')):
            if path.is_file():
                assets[path.relative_to(dist).as_posix()] = base64.b64encode(gzip.compress(path.read_bytes(), mtime=0)).decode('ascii')
    embedded = '<script>globalThis.__TABLETOP_ASSETS__=' + json.dumps(assets, separators=(',', ':')) + ';</script>\n'
    notices = json.dumps((dist / 'THIRD_PARTY_NOTICES.txt').read_text()).replace('<', '\\u003c')
    embedded += '<script type="application/json" id="third-party-notices">' + notices + '</script>\n'
    embedded += '<script type="module">' + script.replace('</script', '<\\/script') + '</script>'
    html = re.sub(r'<script\b[^>]*src="[^"]+"[^>]*></script>', lambda _: embedded, html)
    html = re.sub(r'<link\b[^>]*rel="modulepreload"[^>]*>', '', html)
    delivery.mkdir(parents=True, exist_ok=True)
    target = delivery / 'Tabletop_Lab.html'
    target.write_text(html)
    # This directory is generated exclusively by this packager. Clear previous
    # hashed bundles so an archive never accumulates stale policy/runtime code.
    static = delivery / 'static'
    if static.exists():
        shutil.rmtree(static)
    shutil.copytree(dist, static)
    shutil.copy2(ROOT / 'README.md', delivery / 'README.md')
    shutil.copy2(dist / 'THIRD_PARTY_NOTICES.txt', delivery / 'THIRD_PARTY_NOTICES.txt')
    # The README's licence pointers must resolve in the delivery folder too.
    shutil.copytree(ROOT / 'licenses', delivery / 'licenses', dirs_exist_ok=True)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        'schema': 'tabletop_browser_build_v1',
        'sourceSha256': {p.name: digest(p) for p in sorted(ROOT.iterdir())
                         if p.is_file() and p.suffix in {'.js', '.mjs', '.html', '.css', '.py', '.json'}},
        'policyManifest': json.loads((dist / 'policies' / 'manifest.json').read_text()),
        'standalone': {'file': target.name, 'bytes': target.stat().st_size, 'sha256': digest(target)},
    }
    (delivery / 'BUILD.json').write_text(json.dumps(provenance, indent=2) + '\n')
    archive = shutil.make_archive(str(delivery / 'Tabletop_Lab_static'), 'zip', root_dir=static)
    print(f'{target} ({target.stat().st_size / 1024**2:.1f} MiB)')
    print(archive)


if __name__ == '__main__':
    main()
