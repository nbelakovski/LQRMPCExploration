'''
Assemble site/ for publishing.

The page fetches the .py files over HTTP and hands them to Pyodide, and it also shows
them on screen, so the code a reader sees is the code that just ran. Copying rather than
symlinking keeps that true on GitHub Pages, which does not follow symlinks.

    uv run build_site.py && python -m http.server -d site
'''

import pathlib
import shutil

HERE = pathlib.Path(__file__).parent
SITE = HERE / 'site'

# Everything Pyodide needs to import websim, in no particular order.
PYTHON_SOURCES = [
    'constants.py', 'dynamics.py', 'simpid.py', 'lqr.py', 'mpc_control.py',
    'main.py', 'websim.py',
]
ASSETS = ['final.glb']


def main():
    py_dir = SITE / 'py'
    py_dir.mkdir(parents=True, exist_ok=True)
    for name in PYTHON_SOURCES:
        shutil.copy2(HERE / name, py_dir / name)
    for name in ASSETS:
        shutil.copy2(HERE / name, SITE / name)

    total_mb = sum(p.stat().st_size for p in SITE.rglob('*') if p.is_file()) / 1e6
    print(f'site/ assembled: {len(PYTHON_SOURCES)} python files, '
          f'{len(ASSETS)} asset(s), {total_mb:.1f} MB total')


if __name__ == '__main__':
    main()
