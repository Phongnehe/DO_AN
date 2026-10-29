"""Run project commands with the isolated local dependencies.

Example: python3 scripts/local_runtime.py scripts/train_cached.py
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
default_deps = ROOT / '.runtime/site-packages'
if not default_deps.exists():
    default_deps = ROOT / '.mabsa-test-venv/lib/python3.9/site-packages'
deps = Path(os.environ.get('MABSA_LOCAL_DEPS', str(default_deps)))
if sys.version_info[:2] == (3, 9) and deps.exists():
    sys.path.insert(0, str(deps))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
sys.dont_write_bytecode = True
# Training can disable an optional broken system Arrow installation.
# The packaged local runtime includes a clean Arrow build for Streamlit.
if os.environ.get('MABSA_DISABLE_ARROW') == '1':
    sys.modules['pyarrow'] = None
if os.environ.get('MABSA_DEBUG') == '1':
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)

if __name__ == '__main__':
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    if target == '--pytest':
        sys.argv[0] = 'pytest'
        runpy.run_module('pytest', run_name='__main__', alter_sys=True)
    elif target == '--streamlit':
        sys.argv = ['streamlit', 'run', str(ROOT / 'app/app.py')] + sys.argv[1:]
        runpy.run_module('streamlit', run_name='__main__', alter_sys=True)
    else:
        runpy.run_path(str(ROOT / target), run_name='__main__')
