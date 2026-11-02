"""Anonymous Kaggle dataset downloader (no API token needed).

Usage:  python download_kaggle.py <owner/slug> <out_dir>
"""
import sys, os, ssl, time, urllib.request

def download(ref, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/download/" + ref
    out = os.path.join(out_dir, ref.split("/")[-1] + ".zip")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        total = int(r.headers.get("Content-Length") or 0)
        print(f"[start] {ref} -> {out}  total={total/1e6:.1f}MB", flush=True)
        got = 0
        last = 0
        with open(out, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got - last > 50 * 1024 * 1024:
                    last = got
                    sp = got / max(time.time() - t0, 1e-6) / 1e6
                    pct = (got / total * 100) if total else 0
                    print(f"  {got/1e6:8.1f}MB / {total/1e6:.1f}MB ({pct:5.1f}%)  {sp:.1f} MB/s", flush=True)
    print(f"[done] {out}  {os.path.getsize(out)/1e6:.1f}MB in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
