"""
fetch_data.py — Targeted CHB-MIT downloader.

Downloads only what the framework needs: every EDF file that contains a
seizure, plus a handful of matched interictal files per patient, from the
PhysioNet CHB-MIT Scalp EEG Database (https://physionet.org/content/chbmit).

Also parses the per-patient *-summary.txt files into a single seizures.json
annotation manifest (file -> list of (start_sec, end_sec)) used downstream by
preprocessing.py for labeling.
"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = "https://physionet.org/files/chbmit/1.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "epilepsy")

# Patients to pull. Each is a distinct pediatric subject with >=1 seizure.
PATIENTS = ["chb01", "chb03", "chb05", "chb08"]
# Cap interictal (seizure-free) files per patient to keep the download lean.
MAX_INTERICTAL = 4


def fetch(url, dest, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True, os.path.getsize(dest)
    for attempt in range(retries):
        try:
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.rename(tmp, dest)
            return True, os.path.getsize(dest)
        except Exception as e:
            print(f"    retry {attempt+1}/{retries} for {os.path.basename(dest)}: {e}", flush=True)
            time.sleep(3)
    return False, 0


def parse_summary(text):
    """Return dict: filename -> list of (start,end) seizure intervals (sec)."""
    ann = {}
    cur = None
    starts = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            ann.setdefault(cur, [])
            continue
        m = re.search(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", line)
        if m and cur:
            starts.append(int(m.group(1)))
            continue
        m = re.search(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", line)
        if m and cur and starts:
            ann[cur].append((starts.pop(), int(m.group(1))))
    return ann


def main():
    os.makedirs(DATA, exist_ok=True)
    manifest = {}
    total_bytes = 0
    for p in PATIENTS:
        sum_path = os.path.join(DATA, f"{p}-summary.txt")
        if not os.path.exists(sum_path):
            ok, _ = fetch(f"{BASE}/{p}/{p}-summary.txt", sum_path)
            if not ok:
                print(f"[{p}] could not fetch summary, skipping", flush=True)
                continue
        with open(sum_path) as f:
            ann = parse_summary(f.read())

        seizure_files = [fn for fn, iv in ann.items() if iv]
        interictal_files = [fn for fn, iv in ann.items() if not iv][:MAX_INTERICTAL]
        want = seizure_files + interictal_files
        print(f"[{p}] {len(seizure_files)} seizure files, "
              f"{len(interictal_files)} interictal files", flush=True)

        for fn in want:
            dest = os.path.join(DATA, fn)
            ok, nbytes = fetch(f"{BASE}/{p}/{fn}", dest)
            if ok:
                total_bytes += nbytes
                manifest[fn] = {"patient": p, "seizures": ann.get(fn, [])}
                print(f"    {fn}: {nbytes/1e6:.1f} MB  (cum {total_bytes/1e6:.0f} MB)", flush=True)
            else:
                print(f"    {fn}: FAILED", flush=True)

    with open(os.path.join(DATA, "seizures.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    n_sz = sum(len(v["seizures"]) for v in manifest.values())
    print(f"\nDONE. {len(manifest)} files, {total_bytes/1e6:.0f} MB, "
          f"{n_sz} annotated seizures. Manifest -> seizures.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
