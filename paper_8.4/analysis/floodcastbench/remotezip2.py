#!/usr/bin/env python3
"""List or extract members of a remote zip using HTTP range requests.
   remotezip.py URL list
   remotezip.py URL get PREFIX OUTDIR      (extract members whose name starts with PREFIX)
   remotezip.py URL hourly PREFIX OUTDIR   (only frames named <seconds>.tif with seconds % 3600 == 0)"""
import io, os, sys, urllib.request, zipfile

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read(); break
            except Exception:
                if attempt == 4: raise
        self.pos += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

url, cmd = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=4 << 20))
if cmd == "list":
    for i in z.infolist():
        print(f"{i.file_size:14d} {i.filename}")
else:
    prefix, out = sys.argv[3], sys.argv[4]
    def wanted(name):
        if cmd != "hourly": return True
        stem = name.rsplit("/", 1)[-1][:-4]
        return name.endswith(".tif") and stem.isdigit() and int(stem) % 3600 == 0
    for i in z.infolist():
        if i.filename.startswith(prefix) and not i.is_dir() and wanted(i.filename):
            if os.path.exists(os.path.join(out, i.filename)) and os.path.getsize(os.path.join(out, i.filename)) == i.file_size:
                continue
            z.extract(i, out); print("got", i.filename, i.file_size, flush=True)
