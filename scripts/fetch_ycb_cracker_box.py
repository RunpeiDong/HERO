#!/usr/bin/env python3
"""Fetch and prepare the YCB 003_cracker_box (Cheez-It) model for the Tabletop Lab demo.

Downloads the Google 16k textured scan from the YCB benchmark bucket, verifies its SHA-256, recentres the mesh so
its axis-aligned bounding box is centred at the origin (z up, thin axis along x, long axis along y), keeps the
vertex/texture-coordinate/face records only, and downsamples the 4096 px atlas to 1024 px. Output:
assets/objects/ycb/003_cracker_box/{cracker_box_visual.obj,cracker_box_texture.png,SOURCE.json}.

Source: Calli, Singh, Walsman, Srinivasa, Abbeel, Dollar, "The YCB Object and Model Set", ICAR 2015
(http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/). Check the dataset's licence terms before redistribution.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path
import argparse

import _bootstrap  # noqa: F401
from hero_isaacsim.paths import ASSETS_ROOT

import numpy as np
from PIL import Image

URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/003_cracker_box_google_16k.tgz"
SHA256 = "0f084b0212274700d416384b9bc290521257d2b15ab66f68768a6a22806226b6"
OUT = ASSETS_ROOT / "objects/ycb/003_cracker_box"
TEXTURE_PX = 1024


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "003_cracker_box_google_16k.tgz"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        print("downloading", URL)
        archive.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"Unexpected archive SHA-256: {digest}")
    with tarfile.open(archive) as tar:
        obj = tar.extractfile("003_cracker_box/google_16k/textured.obj").read().decode()
        png = tar.extractfile("003_cracker_box/google_16k/texture_map.png").read()
    vertices, texcoords, faces = [], [], []
    for line in obj.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(v) for v in parts[1:4]])
        elif parts[0] == "vt":
            texcoords.append([float(v) for v in parts[1:3]])
        elif parts[0] == "f":
            faces.append([tuple(token.split("/")[:2]) for token in parts[1:4]])
    v = np.asarray(vertices)
    lo, hi = v.min(0), v.max(0)
    v = v - (lo + hi) / 2
    with (OUT / "cracker_box_visual.obj").open("w") as f:
        f.write("# YCB 003_cracker_box (google_16k), bounding box recentred at the origin; texture cracker_box_texture.png\n")
        for p in v:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in texcoords:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for tri in faces:
            f.write("f " + " ".join(f"{a}/{b}" for a, b in tri) + "\n")
    Image.open(io.BytesIO(png)).convert("RGB").resize((TEXTURE_PX, TEXTURE_PX), Image.LANCZOS).save(OUT / "cracker_box_texture.png", optimize=True)
    extent = (hi - lo).round(4).tolist()
    (OUT / "SOURCE.json").write_text(json.dumps(dict(url=URL, sha256=SHA256, extent_m=extent, texture_px=TEXTURE_PX,
        citation="Calli et al., The YCB Object and Model Set: Towards Common Benchmarks for Manipulation Research, ICAR 2015",
        vertices=len(vertices), faces=len(faces)), indent=2) + "\n")
    print(json.dumps(dict(out=str(OUT), extent_m=extent, vertices=len(vertices), faces=len(faces)), indent=1))


if __name__ == "__main__":
    main()
