import sys
import os
import runpy

# Ensure local fab04 folder is first on sys.path
root = os.path.dirname(os.path.abspath(__file__))
if root in sys.path:
    sys.path.remove(root)
sys.path.insert(0, root)
# Also ensure the copied bench directory is importable (after fab04)
bench_dir = os.path.join(root, 'bench-p04-pcam')
if bench_dir in sys.path:
    sys.path.remove(bench_dir)
sys.path.insert(1, bench_dir)

script = os.path.join(bench_dir, 'self_check.py')
# Forward CLI args to the bench script
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name='__main__')
