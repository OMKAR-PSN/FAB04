import sys, os, runpy
root = os.path.dirname(os.path.abspath(__file__))
bench_dir = os.path.join(root, 'out', 'bench-p04-pcam')
# ensure paths
if root in sys.path:
    sys.path.remove(root)
sys.path.insert(0, root)
if bench_dir in sys.path:
    sys.path.remove(bench_dir)
sys.path.insert(1, bench_dir)

script = os.path.join(bench_dir, 'run.py')
# forward args
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name='__main__')
