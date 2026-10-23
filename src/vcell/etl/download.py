"""Resumable, checksum-verified downloads into data/sources/ (was eda/02, 05, 10 and 11, each with its own copy of the same
fetch loop).

    python -m vcell etl download small [--list]                 # Replogle pseudobulks, X-Atlas guide calls, Zhu tables (0.7 GB)
    python -m vcell etl download h1 [--list] [--small] [--only train]   # Arc H1 hESC from the public GCS bucket (34 GB)
    python -m vcell etl download xatlas --line HCT116 [--list] [--no-verify]   # X-Atlas/Orion Parquet (46.6 / 79.7 GB)
    python -m vcell etl download more [--list] [--only nadig_jurkat]  # Nadig Jurkat/HepG2, Replogle RPE1/K562 single cells

Downloads resume with HTTP Range from a partial file and are checked against what the server publishes: MD5 (Figshare
API), MD5 or crc32c (GCS), sha256 (Hugging Face LFS), else the size (and, for h5ad files, that h5py can open them).
KOLF2.1J is not downloaded: etl.stage1.kolf streams the columns it needs. HKU, HPA, GENCODE, GSE225775 and ELEM-1200RS were
fetched by hand; their locations and checksums are in docs/DATASET_INVENTORY.md.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import time
from pathlib import Path

import requests

from vcell import config


# ------------------------------------------------------------------------------------------------------------- shared
def keep_awake() -> None:
    """Keep Windows from idle-sleeping while this process runs (a manual sleep still stops it)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def file_hash(path, algo: str = "md5", b64: bool = False) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode() if b64 else h.hexdigest()


def crc32c_b64(path) -> str:
    import google_crc32c
    c = google_crc32c.Checksum()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            c.update(chunk)
    return base64.b64encode(c.digest()).decode()


def fetch(url: str, dest: Path, size: int | None = None, progress: bool = False, timeout=(30, 60)) -> None:
    """Download url to dest, resuming from a partial file. With a known size a complete file is skipped and a larger one
    (stale or corrupt) restarts."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size is not None:
        if have == size:
            return
        if have > size:
            dest.unlink()
            have = 0
    with requests.get(url, stream=True, timeout=timeout, headers={"Range": f"bytes={have}-"} if have else {}) as r:
        if r.status_code == 416:                          # already complete
            return
        r.raise_for_status()
        mode = "ab" if (have and r.status_code == 206) else "wb"
        done, last = (have if mode == "ab" else 0), -1
        with open(dest, mode) as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
                done += len(chunk)
                if progress and size:
                    pct = int(100 * done / size)
                    if pct != last and pct % 5 == 0:
                        print(f"   {dest.name}: {pct}%  ({done / 1e9:.2f} / {size / 1e9:.2f} GB)", flush=True)
                        last = pct


def fetch_retry(url: str, dest: Path, size: int | None = None, tries: int = 5, **kw) -> None:
    """fetch with retries: a dropped connection (network change, sleep) resumes from the partial file."""
    for attempt in range(1, tries + 1):
        try:
            fetch(url, dest, size, **kw)
            return
        except (requests.RequestException, OSError) as e:
            print(f"   attempt {attempt} failed ({type(e).__name__}); retrying in 15 s", flush=True)
            time.sleep(15)


def remote_size(url: str) -> int:
    r = requests.get(url, stream=True, timeout=(30, 60), headers={"Range": "bytes=0-0"})
    r.raise_for_status()
    cr = r.headers.get("Content-Range")                   # 'bytes 0-0/12345'
    n = int(cr.split("/")[1]) if cr else int(r.headers["Content-Length"])
    r.close()
    return n


# ------------------------------------------------------------------------------------------- small sources (eda/02)
FIG = "https://ndownloader.figshare.com/files/{}"
RAW = "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/{}"
SMALL = {   # name -> (url, "figshare" if Figshare publishes an MD5 else None, note)
    "replogle/K562_gwps_raw_bulk_01.h5ad": (FIG.format(35774443), "figshare", "K562 genome-wide pseudobulk, mean counts per cell (0.37 GB)"),
    "replogle/K562_essential_raw_bulk_01.h5ad": (FIG.format(35773070), "figshare", "K562 essential pseudobulk (0.08 GB), no panel targets; only for comparison"),
    "replogle/rpe1_raw_bulk_01.h5ad": (FIG.format(35775581), "figshare", "RPE1 pseudobulk (0.10 GB), no panel targets; only for comparison"),
    "xatlas/guide_library.csv": (FIG.format(57368587), "figshare", "X-Atlas/Orion guide pairs (4 MB)"),
    "xatlas/HCT116_filtered_guide_calls_per_cell.csv.gz": (FIG.format(59490731), "figshare", "HCT116 guide calls, one row per cell (70 MB)"),
    "xatlas/HEK293T_filtered_guide_calls_per_cell.csv.gz": (FIG.format(59490734), "figshare", "HEK293T guide calls (91 MB)"),
    "zhu/guide_kd_efficiency.suppl_table.csv": (RAW.format("guide_kd_efficiency.suppl_table.csv"), None, "CD4+ T knockdown and cells per guide (13 MB)"),
    "zhu/sgrna_library_metadata.suppl_table.csv": (RAW.format("sgrna_library_metadata.suppl_table.csv"), None, "CD4+ T guide library (10 MB)"),
}
_FIGSHARE_MD5: dict = {}


def figshare_md5(file_id: str) -> str | None:
    if not _FIGSHARE_MD5:
        for art in (20029387, 29190726):
            r = requests.get(f"https://api.figshare.com/v2/articles/{art}", timeout=60).json()
            for f in r["files"]:
                _FIGSHARE_MD5[str(f["id"])] = f["computed_md5"]
    return _FIGSHARE_MD5.get(file_id)


def small(list_only: bool = False) -> int:
    bad = 0
    for rel, (url, md5, note) in SMALL.items():
        dest = config.SOURCES / rel
        print(f"{rel:60s} {note}")
        if list_only:
            continue
        want = figshare_md5(url.rsplit("/", 1)[1]) if md5 == "figshare" else None
        if dest.exists() and want and file_hash(dest) == want:
            print("   present, md5 ok")
            continue
        fetch(url, dest, timeout=120)
        if want:
            ok = file_hash(dest) == want
            print("   downloaded, md5", "ok" if ok else "MISMATCH")
            bad += not ok
        else:
            print(f"   downloaded, {dest.stat().st_size / 1e6:.1f} MB (no checksum published)")
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------- Arc H1 (eda/05)
H1_BUCKET = "arc-institute-virtual-cell-atlas"
H1_PREFIX = "virtual-cell-challenge/2025/"
H1_API = f"https://storage.googleapis.com/storage/v1/b/{H1_BUCKET}/o"
H1_DL = f"https://storage.googleapis.com/{H1_BUCKET}/"
H1_SPLITS = ("train", "validation", "test")


def h1_listing() -> list[dict]:
    """Objects under the 2025 prefix, minus the FASTQ folder."""
    items, token = [], None
    while True:
        params = {"prefix": H1_PREFIX, "maxResults": 500}
        if token:
            params["pageToken"] = token
        r = requests.get(H1_API, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        items += [o for o in j.get("items", []) if not o["name"].startswith(H1_PREFIX + "FASTQ/") and not o["name"].endswith("/")]
        token = j.get("nextPageToken")
        if not token:
            return items


def gcs_verified(dest: Path, o: dict) -> str:
    """'md5', 'crc32c' or 'size' if the file matches the strongest checksum the bucket publishes, else ''.
    Composite objects (the big h5ad files) have no md5Hash, only crc32c."""
    if not dest.exists() or dest.stat().st_size != int(o["size"]):
        return ""
    if "md5Hash" in o:
        return "md5" if file_hash(dest, b64=True) == o["md5Hash"] else ""
    if "crc32c" in o:
        return "crc32c" if crc32c_b64(dest) == o["crc32c"] else ""
    return "size"


def h1(list_only: bool = False, small_only: bool = False, only: str | None = None) -> int:
    items = h1_listing()
    if only:
        items = [o for o in items if o["name"].startswith(H1_PREFIX + only + "/") or o["name"] == H1_PREFIX + "gene_names.csv"]
    if small_only:
        items = [o for o in items if int(o["size"]) < 50e6]
    total = sum(int(o["size"]) for o in items)
    print(f"{len(items)} files, {total / 1e9:.2f} GB, into {config.SOURCES / 'h1'}")
    bad = 0
    for o in sorted(items, key=lambda o: o["name"]):
        size = int(o["size"])
        dest = config.SOURCES / "h1" / o["name"][len(H1_PREFIX):]
        print(f"{o['name'][len(H1_PREFIX):]:48s} {size / 1e9:7.3f} GB")
        if list_only:
            continue
        how = gcs_verified(dest, o)
        if how:
            print(f"   present, {how} ok")
            continue
        fetch(H1_DL + o["name"], dest, size, progress=True, timeout=120)
        how = gcs_verified(dest, o)
        print(f"   downloaded, {how} ok" if how else "   downloaded, CHECKSUM MISMATCH")
        bad += not how
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------- X-Atlas/Orion (eda/10)
XATLAS_REPO = "Xaira-Therapeutics/X-Atlas-Orion"
XATLAS_API = f"https://huggingface.co/api/datasets/{XATLAS_REPO}/tree/main/"
XATLAS_DL = f"https://huggingface.co/datasets/{XATLAS_REPO}/resolve/main/"


def hf_listing(prefix: str) -> list[dict]:
    r = requests.get(XATLAS_API + prefix, timeout=60)
    r.raise_for_status()
    return [o for o in r.json() if o["type"] == "file"]


def xatlas(line: str, list_only: bool = False, verify: bool = True) -> int:
    """One X-Atlas/Orion line (HCT116 46.6 GB, HEK293T 79.7 GB) plus metadata/gene_metadata.parquet. Licence CC BY-NC-SA 4.0."""
    keep_awake()
    files = [o for o in hf_listing("data") if o["path"].startswith(f"data/{line}")]
    files.append(next(o for o in hf_listing("metadata") if o["path"].endswith("gene_metadata.parquet")))
    total = sum(o["size"] for o in files)
    print(f"{line}: {len(files)} files, {total / 1e9:.1f} GB -> {config.SOURCES / 'xatlas' / line}")
    if list_only:
        for o in files[:5]:
            print(f"   {o['path']:50s} {o['size'] / 1e6:8.1f} MB")
        return 0
    bad = 0
    for k, o in enumerate(sorted(files, key=lambda o: o["path"]), 1):
        name = o["path"].split("/", 1)[1]
        dest = config.SOURCES / "xatlas" / (line if o["path"].startswith("data/") else "") / name
        want = o["lfs"]["oid"] if "lfs" in o else None
        fetch_retry(XATLAS_DL + o["path"], dest, o["size"])
        ok = dest.stat().st_size == o["size"] and (not verify or want is None or file_hash(dest, "sha256") == want)
        print(f"[{k}/{len(files)}] {name:40s} {o['size'] / 1e6:8.1f} MB  {'ok' if ok else 'MISMATCH'}", flush=True)
        bad += not ok
    return 1 if bad else 0


# -------------------------------------------------------------------------- held-out lines and K562 cells (eda/11)
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/{}"
MORE = {
    "nadig_jurkat": (GEO.format("GSE264667_jurkat_raw_singlecell_01.h5ad"), "nadig/GSE264667_jurkat_raw_singlecell_01.h5ad"),
    "nadig_hepg2": (GEO.format("GSE264667_hepg2_raw_singlecell_01.h5ad"), "nadig/GSE264667_hepg2_raw_singlecell_01.h5ad"),
    "rpe1_sc": (FIG.format(35775606), "replogle/rpe1_raw_singlecell_01.h5ad"),
    "k562_gwps_sc": (FIG.format(35775507), "replogle/K562_gwps_raw_singlecell_01.h5ad"),
}


def more(list_only: bool = False, only: str | None = None) -> int:
    """Size equal to the server's Content-Length (no MD5 published for these), then a check that h5py can open the file."""
    import h5py
    keep_awake()
    bad = 0
    for name, (url, rel) in MORE.items():
        if only and name != only:
            continue
        dest = config.SOURCES / rel
        size = remote_size(url)
        print(f"{name:14s} {size / 1e9:6.2f} GB  -> {dest}", flush=True)
        if list_only:
            continue
        fetch_retry(url, dest, size)
        ok = dest.exists() and dest.stat().st_size == size
        if ok:
            try:
                with h5py.File(dest, "r") as f:
                    ok = "X" in f and "obs" in f
            except OSError:
                ok = False
        print("   ", "ok" if ok else "FAILED", flush=True)
        bad += not ok
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vcell etl download", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["small", "h1", "xatlas", "more"])
    ap.add_argument("--list", action="store_true", help="list only, download nothing")
    ap.add_argument("--small", action="store_true", help="h1: only files under 50 MB")
    ap.add_argument("--only", default=None, help="h1: one split (train | validation | test); more: one item")
    ap.add_argument("--line", choices=["HCT116", "HEK293T"], help="xatlas: the line")
    ap.add_argument("--no-verify", action="store_true", help="xatlas: skip the sha256 pass (slow on tens of GB)")
    a = ap.parse_args(argv)
    if a.what == "small":
        return small(a.list)
    if a.what == "h1":
        return h1(a.list, a.small, a.only)
    if a.what == "xatlas":
        return xatlas(a.line, a.list, not a.no_verify)
    return more(a.list, a.only)
