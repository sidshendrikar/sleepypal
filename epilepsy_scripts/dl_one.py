"""dl_one.py <patient> — download one CHB-MIT patient's seizure files + a couple
interictal files, in its own process (run several in parallel to beat the
per-connection throttle). Manifest is rebuilt separately afterwards."""
import sys, os
from fetch_data import fetch, parse_summary, BASE, DATA

p = sys.argv[1]
n_inter = int(sys.argv[2]) if len(sys.argv) > 2 else 2
os.makedirs(DATA, exist_ok=True)
sp = os.path.join(DATA, f"{p}-summary.txt")
if not (os.path.exists(sp) and os.path.getsize(sp) > 0):
    fetch(f"{BASE}/{p}/{p}-summary.txt", sp)
ann = parse_summary(open(sp).read())
seizure_files = [fn for fn, iv in ann.items() if iv]
interictal = [fn for fn, iv in ann.items() if not iv][:n_inter]
want = seizure_files + interictal
print(f"[{p}] {len(seizure_files)} seizure + {len(interictal)} interictal files", flush=True)
tot = 0
for fn in want:
    ok, nb = fetch(f"{BASE}/{p}/{fn}", os.path.join(DATA, fn))
    tot += nb
    print(f"  [{p}] {fn} {'OK' if ok else 'FAIL'} {nb/1e6:.0f}MB (cum {tot/1e6:.0f})", flush=True)
print(f"[{p}] DONE {tot/1e6:.0f}MB", flush=True)
