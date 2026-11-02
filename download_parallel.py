"""Parallel segmented downloader (uses HTTP Range). Speeds up large wheels.

Usage: python download_parallel.py <url> <out_path> [n_threads]
"""
import sys, os, ssl, time, threading, urllib.request

def head_size(url, ctx):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"})
    r = urllib.request.urlopen(req, timeout=30, context=ctx)
    cr = r.headers.get("Content-Range")
    r.close()
    if cr and "/" in cr:
        return int(cr.split("/")[-1])
    return None

def worker(url, path, start, end, ctx, stats, idx):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            r = urllib.request.urlopen(req, timeout=60, context=ctx)
            with open(path, "r+b") as f:
                f.seek(start)
                got = 0
                while True:
                    c = r.read(1024 * 512)
                    if not c:
                        break
                    f.write(c)
                    got += len(c)
                    stats["done"] += len(c)
            r.close()
            if got >= (end - start + 1):
                stats["parts"][idx] = True
                return
        except Exception as e:
            time.sleep(1)
    print(f"  [thread {idx}] FAILED range {start}-{end}", flush=True)

def download(url, out, n=8):
    ctx = ssl.create_default_context()
    total = head_size(url, ctx)
    assert total, "server does not report size"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.truncate(total)
    seg = (total + n - 1) // n
    stats = {"done": 0, "parts": [False] * n}
    t0 = time.time()

    def printer():
        while not all(stats["parts"]):
            time.sleep(3)
            done = stats["done"]
            sp = done / max(time.time() - t0, 1e-6) / 1e6
            eta = (total - done) / max(sp * 1e6, 1)
            print(f"  {done/1e6:8.1f}/{total/1e6:.1f}MB ({done/total*100:5.1f}%) "
                  f"{sp:5.2f} MB/s  ETA {eta/60:4.1f}min", flush=True)

    th = [threading.Thread(target=worker, args=(url, out, i * seg, min((i + 1) * seg - 1, total - 1), ctx, stats, i))
          for i in range(n)]
    pt = threading.Thread(target=printer, daemon=True)
    pt.start()
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(f"[done] {out}  {os.path.getsize(out)/1e6:.1f}MB in {(time.time()-t0)/60:.1f}min  "
          f"ok={all(stats['parts'])}", flush=True)

if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
