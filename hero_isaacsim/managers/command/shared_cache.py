"""Share a validated motion corpus between training processes.

The cache uses file locking, source fingerprints and atomic completion
markers to avoid duplicate conversion and partially visible payloads."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from loguru import logger

FORMAT_VERSION = 1
#: Fixed ``.npy`` header length written by :class:`NpyStreamWriter` (magic + version + len + padded dict); every array file
#: is exactly ``NPY_HEADER_BYTES + nbytes`` long, which is what the reader validates.
NPY_HEADER_BYTES = 128
READY_NAME = "READY"
META_NAME = "meta.json"
ARRAYS_DIR = "arrays"
PROGRESS_NAME = "PROGRESS"

#: Names ``--list`` / ``--clean`` recognise under a cache base dir: the 32-hex key (entry dir) and its side files.
KEY_RE = re.compile(r"^(?P<key>[0-9a-f]{32})(?P<suffix>\.lock|\.FAILED|\.building\.\d+)?$")
ENV_CACHE_DIR = "HERO_SHARED_CACHE_DIR"
ENV_TIMEOUT_S = "HERO_SHARED_CACHE_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 3.0 * 3600.0
OFF_VALUES = ("", "off", "none", "0", "false", "no")
AUTO_VALUE = "auto"
DEV_SHM = Path("/dev/shm")
#: Free space the auto mode wants on ``/dev/shm`` relative to the corpus size (header + slack).
AUTO_SHM_HEADROOM = 1.15


# ----------------------------------------------------------------------------------------------------------------------
# cache key
# ----------------------------------------------------------------------------------------------------------------------
def file_records(files: Iterable[str | os.PathLike]) -> list[tuple[str, int, int]]:
    """``(real absolute path, size bytes, mtime_ns)`` per file in the given order (the loader's discovery order).

    ``os.path.realpath`` (symlinks resolved), not ``abspath``: a relative spelling, a symlinked corpus dir or macOS ``/tmp`` vs
    ``/private/tmp`` must map to ONE key, otherwise a ``--build`` from another cwd than the launcher's silently rebuilds."""
    out: list[tuple[str, int, int]] = []
    for f in files:
        p = Path(os.path.realpath(os.path.expanduser(str(f))))
        st = p.stat()
        out.append((str(p), int(st.st_size), int(st.st_mtime_ns)))
    return out


def corpus_cache_key(files: Sequence[str | os.PathLike], *, params: Mapping[str, Any] | None = None) -> str:
    """sha256 (hex, 32 chars) of the file list (real path / size / mtime_ns, in order) + the loader parameters + the format version."""
    h = hashlib.sha256()
    h.update(f"hero_shared_cache/v{FORMAT_VERSION}\n".encode())
    h.update(json.dumps(_jsonable(dict(params or {})), sort_keys=True, separators=(",", ":")).encode())
    h.update(b"\n")
    for path, size, mtime in file_records(files):
        h.update(f"{path}\t{size}\t{mtime}\n".encode())
    return h.hexdigest()[:32]


def corpus_bytes(files: Iterable[str | os.PathLike]) -> int:
    """Sum of the file sizes (an ``np.savez`` npz is stored uncompressed: ~= the resident size of the timelines)."""
    return sum(size for _, size, _ in file_records(files))


def _jsonable(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return repr(v)


# ----------------------------------------------------------------------------------------------------------------------
# streaming .npy writer (fixed-length header rewritten at close)
# ----------------------------------------------------------------------------------------------------------------------
def npy_header_bytes(shape: Sequence[int], dtype: Any = np.float32, total: int = NPY_HEADER_BYTES) -> bytes:
    """A version-1.0 ``.npy`` header padded with spaces to exactly ``total`` bytes (``np.load`` / ``mmap_mode`` read it as is)."""
    descr = np.lib.format.dtype_to_descr(np.dtype(dtype))
    body = ("{'descr': %r, 'fortran_order': False, 'shape': %r, }" % (descr, tuple(int(s) for s in shape))).encode("latin1")
    hlen = total - 10  # magic (6) + version (2) + uint16 header length (2)
    if len(body) + 1 > hlen:
        raise ValueError(f"npy header for shape {tuple(shape)} does not fit in {total} bytes")
    body = body + b" " * (hlen - len(body) - 1) + b"\n"
    return b"\x93NUMPY" + bytes([1, 0]) + int(hlen).to_bytes(2, "little") + body


class NpyStreamWriter:
    """Append rows to a ``.npy`` file without knowing the final row count (header rewritten at ``close``)."""

    def __init__(self, path: str | os.PathLike, row_shape: Sequence[int], dtype: Any = np.float32):
        self.path = Path(path)
        self.row_shape = tuple(int(s) for s in row_shape)
        self.dtype = np.dtype(dtype)
        self.rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "wb", buffering=8 * 1024 * 1024)
        self._fp.write(npy_header_bytes((0, *self.row_shape), self.dtype))

    def append(self, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype=self.dtype)
        if a.shape[1:] != self.row_shape:
            raise ValueError(f"{self.path.name}: row shape {a.shape[1:]} != {self.row_shape}")
        self._fp.write(a.tobytes())
        self.rows += int(a.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.rows * int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize)

    def close(self) -> dict[str, Any]:
        """Finalize the header; returns the array descriptor recorded in ``meta.json``."""
        self._fp.flush()
        self._fp.seek(0)
        self._fp.write(npy_header_bytes((self.rows, *self.row_shape), self.dtype))
        self._fp.flush()
        os.fsync(self._fp.fileno())
        self._fp.close()
        return {
            "shape": [self.rows, *self.row_shape],
            "dtype": np.lib.format.dtype_to_descr(self.dtype),
            "nbytes": self.nbytes,
            "file": f"{ARRAYS_DIR}/{self.path.name}",
        }

    def abort(self) -> None:
        try:
            self._fp.close()
        finally:
            self.path.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------------------------------
# cache entry: paths + validation
# ----------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CacheEntry:
    base_dir: Path
    key: str
    #: Arrays the reader will index.  A READY entry that lacks one (written by a loader with a different timeline key set)
    #: is reported as stale by :meth:`problems` and rebuilt under the lock instead of crashing the reader with ``KeyError``.
    #: ``object_arrays`` are required only when ``meta["has_object"]`` (object-free corpora do not write them).
    required_arrays: tuple[str, ...] = ()
    object_arrays: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_dir", Path(os.path.abspath(os.path.expanduser(str(self.base_dir)))))
        object.__setattr__(self, "required_arrays", tuple(str(a) for a in self.required_arrays))
        object.__setattr__(self, "object_arrays", tuple(str(a) for a in self.object_arrays))

    @property
    def dir(self) -> Path:
        return self.base_dir / self.key

    @property
    def lock_path(self) -> Path:
        return self.base_dir / f"{self.key}.lock"

    @property
    def failed_path(self) -> Path:
        return self.base_dir / f"{self.key}.FAILED"

    @property
    def ready_path(self) -> Path:
        return self.dir / READY_NAME

    @property
    def meta_path(self) -> Path:
        return self.dir / META_NAME

    def building_dir(self, pid: int | None = None) -> Path:
        return self.base_dir / f"{self.key}.building.{os.getpid() if pid is None else pid}"

    def building_dirs(self) -> list[Path]:
        return sorted(p for p in self.base_dir.glob(f"{self.key}.building.*") if p.is_dir())

    # ---- validation
    def problems(self) -> list[str]:
        """Empty list == a complete, matching cache; otherwise why it is unusable (stale / partial / foreign)."""
        if not self.dir.is_dir():
            return ["no cache dir"]
        if not self.ready_path.is_file():
            return ["no READY marker"]
        try:
            ready = json.loads(self.ready_path.read_text() or "{}")
        except (OSError, ValueError) as exc:
            return [f"READY unreadable: {exc}"]
        if ready.get("key") != self.key:
            return [f"READY key {ready.get('key')!r} != {self.key}"]
        try:
            meta = json.loads(self.meta_path.read_text())
        except (OSError, ValueError) as exc:
            return [f"meta.json unreadable: {exc}"]
        return validate_meta(self.dir, meta, self.key, required_arrays=self.required_arrays, object_arrays=self.object_arrays)

    def is_ready(self) -> bool:
        return not self.problems()

    def read_meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text())


def validate_meta(
    cache_dir: Path,
    meta: Mapping[str, Any],
    key: str | None = None,
    *,
    required_arrays: Iterable[str] = (),
    object_arrays: Iterable[str] = (),
) -> list[str]:
    """Why ``meta`` (of the entry in ``cache_dir``) is unusable; ``[]`` == complete.

    Structural checks: format version, key, every listed array present with exactly ``NPY_HEADER_BYTES + nbytes`` on disk,
    and -- when the caller names them -- every ``required_arrays`` entry listed (plus ``object_arrays`` when
    ``meta["has_object"]``), so a timeline-key-set change that was shipped without a ``FORMAT_VERSION`` bump reads as
    ``array X: not in meta`` (stale -> rebuilt) rather than as a ``KeyError`` in the reader."""
    problems: list[str] = []
    if meta.get("format_version") != FORMAT_VERSION:
        problems.append(f"format_version {meta.get('format_version')} != {FORMAT_VERSION}")
    if key is not None and meta.get("key") != key:
        problems.append(f"meta key {meta.get('key')!r} != {key}")
    arrays = meta.get("arrays")
    if not isinstance(arrays, Mapping) or not arrays:
        problems.append("meta has no arrays")
        return problems
    object_arrays = tuple(object_arrays)
    need = list(required_arrays)
    if object_arrays:
        if "has_object" not in meta:
            problems.append("meta lacks has_object")
        elif meta.get("has_object"):
            need.extend(object_arrays)
    for name in need:
        if name not in arrays:
            problems.append(f"array {name}: not in meta (written by a loader with another timeline key set)")
    for name, desc in arrays.items():
        p = cache_dir / str(desc.get("file", f"{ARRAYS_DIR}/{name}.npy"))
        if not p.is_file():
            problems.append(f"array {name}: file missing")
            continue
        want = NPY_HEADER_BYTES + int(desc["nbytes"])
        have = p.stat().st_size
        if have != want:
            problems.append(f"array {name}: {have} bytes on disk != {want} expected (partial / truncated)")
    return problems


def load_arrays(cache_dir: Path, meta: Mapping[str, Any], *, mmap_mode: str = "c") -> dict[str, np.memmap]:
    """``{name: np.memmap}`` of every array in ``meta`` (copy-on-write mappings)."""
    out: dict[str, np.memmap] = {}
    for name, desc in meta["arrays"].items():
        arr = np.load(cache_dir / desc["file"], mmap_mode=mmap_mode)
        want_shape = tuple(int(s) for s in desc["shape"])
        if tuple(arr.shape) != want_shape or np.dtype(arr.dtype) != np.dtype(desc["dtype"]):
            raise RuntimeError(f"shared cache array {name}: header {arr.shape}/{arr.dtype} != meta {want_shape}/{desc['dtype']}")
        out[name] = arr
    return out


def entry_size_bytes(cache_dir: Path) -> int:
    total = 0
    for p in cache_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ----------------------------------------------------------------------------------------------------------------------
# lock / wait / build protocol
# ----------------------------------------------------------------------------------------------------------------------
class SharedCacheError(RuntimeError):
    """The writer failed (its error text is carried over to the waiting ranks)."""


class SharedCacheTimeout(TimeoutError):
    """No READY marker appeared within the timeout."""


def _try_lock(path: Path) -> int | None:
    """Non-blocking exclusive flock on ``path``; returns the fd (keep it open to hold the lock) or None when held elsewhere.

    Only ``EAGAIN`` / ``EWOULDBLOCK`` (:class:`BlockingIOError`) means "another process holds it".  Any other ``OSError``
    (``ENOLCK`` / ``ENOSYS`` / ``EACCES``: NFS without lockd, FUSE / S3 mounts, a read-only lock file) means the filesystem
    cannot lock at all -- raised as :class:`SharedCacheError` so the ranks fail in seconds instead of all of them concluding
    "held elsewhere", nobody building, and everyone waiting out ``HERO_SHARED_CACHE_TIMEOUT_S``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except OSError as exc:
        os.close(fd)
        raise SharedCacheError(
            f"flock() unsupported/denied on {path.parent} ({exc.strerror or exc}, errno {exc.errno}); point {ENV_CACHE_DIR} at a "
            f"local disk or /dev/shm path, or set it to off"
        ) from exc
    return fd


def _unlock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_progress(entry: CacheEntry) -> str:
    for d in entry.building_dirs():
        try:
            return (d / PROGRESS_NAME).read_text().strip()
        except OSError:
            continue
    return "?"


def _fresh_failure(entry: CacheEntry, since: float) -> str | None:
    """Error text of a ``.FAILED`` marker written after ``since`` (older markers are leftovers of an earlier run)."""
    try:
        st = entry.failed_path.stat()
    except OSError:
        return None
    if st.st_mtime < since - 2.0:
        return None
    try:
        return json.loads(entry.failed_path.read_text()).get("error", "?")
    except (OSError, ValueError):
        return "?"


def timeout_from_env(default: float = DEFAULT_TIMEOUT_S) -> float:
    raw = os.environ.get(ENV_TIMEOUT_S, "")
    try:
        return float(raw) if raw.strip() else float(default)
    except ValueError:
        logger.warning(f"{ENV_TIMEOUT_S}={raw!r} is not a number; using {default} s")
        return float(default)


def writer_activity_signature(entry: CacheEntry) -> tuple:
    """What a waiter can observe of a live writer: the scratch dirs' mtimes, their ``PROGRESS`` text and the bytes of every array
    written so far.  Any change means the writer is making progress (used by :func:`acquire` with ``writer_activity_extends``)."""
    sig: list = []
    for d in entry.building_dirs():
        try:
            sig.append((d.name, int(d.stat().st_mtime_ns)))
        except OSError:
            continue
        try:
            sig.append((d / PROGRESS_NAME).read_text().strip())
        except OSError:
            pass
        arrays = d / ARRAYS_DIR
        if arrays.is_dir():
            try:
                sig.append(sum(p.stat().st_size for p in arrays.iterdir() if p.is_file()))
            except OSError:
                pass
    return tuple(sig)


def acquire(
    entry: CacheEntry,
    build: Callable[[Path, Callable[[int, int], None]], Mapping[str, Any]],
    *,
    timeout_s: float | None = None,
    poll_s: float = 1.0,
    log_every_s: float = 30.0,
    timeout_env_name: str = ENV_TIMEOUT_S,
    writer_activity_extends: bool = False,
) -> tuple[Path, str]:
    """Return ``(ready cache dir, role)`` with ``role in {"hit", "built"}``.

    ``build(scratch_dir, progress)`` (run by the writer only) streams the arrays into ``scratch_dir/arrays/`` and returns the
    ``meta`` mapping (must contain ``arrays``; ``key`` / ``format_version`` / writer info are added here).  ``progress(done,
    total)`` may be called freely (rate-limited PROGRESS file for the waiters).

    ``timeout_env_name`` is the knob named in the :class:`SharedCacheTimeout` text (a cache layered on this protocol -- the stock
    loader cache -- has its own variable; naming this module's would send the operator to a knob without effect)."""
    timeout_s = timeout_from_env() if timeout_s is None else float(timeout_s)
    entry.base_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    wall0 = time.time()
    last_log = 0.0
    last_activity = t0
    last_sig: tuple | None = None
    while True:
        if entry.is_ready():
            return entry.dir, "hit"
        failure = _fresh_failure(entry, wall0)
        if failure is not None:
            raise SharedCacheError(f"shared corpus cache {entry.dir}: the writer failed: {failure}")
        fd = _try_lock(entry.lock_path)
        if fd is not None:
            try:
                if entry.is_ready():  # finished between our two checks
                    return entry.dir, "hit"
                _build_locked(entry, build)
                return entry.dir, "built"
            finally:
                _unlock(fd)
        now = time.monotonic()
        elapsed = now - t0
        if writer_activity_extends:
            sig = writer_activity_signature(entry)
            if sig != last_sig:
                last_sig = sig
                last_activity = now
            idle = now - last_activity
        else:
            idle = elapsed
        if idle > timeout_s:
            how = f"the lock holder showed no progress for {idle:.0f} s (waited {elapsed:.0f} s)" if writer_activity_extends else f"after {elapsed:.0f} s"
            raise SharedCacheTimeout(
                f"shared corpus cache {entry.dir}: no READY marker {how} (writer progress {_read_progress(entry)}); "
                f"raise {timeout_env_name} or inspect {entry.base_dir}"
            )
        if elapsed - last_log >= log_every_s:
            logger.info(
                f"shared corpus cache: waiting for the writer of {entry.dir} ({elapsed:.0f} s, progress {_read_progress(entry)} clips)"
            )
            last_log = elapsed
        time.sleep(poll_s)


def _build_locked(entry: CacheEntry, build: Callable[[Path, Callable[[int, int], None]], Mapping[str, Any]]) -> None:
    """Writer side (lock held): wipe partial state, build into a scratch dir, finalize atomically."""
    entry.failed_path.unlink(missing_ok=True)
    for d in entry.building_dirs():
        shutil.rmtree(d, ignore_errors=True)
    if entry.dir.exists():
        logger.warning(f"shared corpus cache: removing stale / partial {entry.dir} ({'; '.join(entry.problems())})")
        shutil.rmtree(entry.dir, ignore_errors=True)
    scratch = entry.building_dir()
    scratch.mkdir(parents=True)
    (scratch / ARRAYS_DIR).mkdir()
    state = {"last": 0.0}

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - state["last"] < 5.0 and done < total:
            return
        state["last"] = now
        try:
            (scratch / PROGRESS_NAME).write_text(f"{done}/{total}")
        except OSError:
            pass

    t0 = time.monotonic()
    logger.info(f"shared corpus cache: this process builds {entry.dir} (pid {os.getpid()}, scratch {scratch.name})")
    try:
        meta = dict(build(scratch, progress))
        meta.update(
            key=entry.key,
            format_version=FORMAT_VERSION,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            writer={"pid": os.getpid(), "host": socket.gethostname(), "build_s": round(time.monotonic() - t0, 1)},
        )
        problems = validate_meta(scratch, meta, entry.key, required_arrays=entry.required_arrays, object_arrays=entry.object_arrays)
        if problems:
            raise SharedCacheError(f"freshly written cache is inconsistent: {problems}")
        _write_json(scratch / META_NAME, meta)
        (scratch / PROGRESS_NAME).unlink(missing_ok=True)
        _write_json(scratch / READY_NAME, {"key": entry.key, "created_utc": meta["created_utc"], "writer": meta["writer"]})
        _fsync_dir(scratch)
        os.rename(scratch, entry.dir)
        _fsync_dir(entry.base_dir)
        size_gb = entry_size_bytes(entry.dir) / 1e9
        logger.info(
            f"shared corpus cache: READY {entry.dir} ({size_gb:.2f} GB, {meta.get('time_step_total', '?')} frames, "
            f"{len(meta.get('clips', []))} clips) built in {time.monotonic() - t0:.1f} s"
        )
    except BaseException as exc:  # noqa: BLE001 -- the waiters must learn about ANY writer failure
        shutil.rmtree(scratch, ignore_errors=True)
        try:
            _write_json(entry.failed_path, {"error": f"{type(exc).__name__}: {exc}", "pid": os.getpid(), "time": time.time()})
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------------------------------------------------
# placement
# ----------------------------------------------------------------------------------------------------------------------
def free_bytes(path: Path) -> int | None:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        st = os.statvfs(p)
    except OSError:
        return None
    return int(st.f_bavail) * int(st.f_frsize)


def default_disk_cache_dir() -> Path:
    root = os.environ.get("HERO_ROOT", "")
    if root:
        return Path(root) / "data" / "_shared_cache"
    return Path(tempfile.gettempdir()) / "hero_shared_cache"


def resolve_cache_dir(
    spec: str | os.PathLike | None,
    *,
    motion_files: Sequence[str] | Callable[[], Sequence[str]] | None = None,
    use_env: bool = True,
) -> tuple[Path | None, str]:
    """``(base dir or None, source)`` for a ``shared_cache_dir`` config value.

    ``motion_files`` sizes the corpus for the ``auto`` mode only; pass a callable (e.g. the loader's discovery) and it is
    invoked ONLY in that branch, so the default / off / explicit-path modes never touch the corpus directory.

    * ``None``  -> ``$HERO_SHARED_CACHE_DIR`` when ``use_env`` and it is set (source ``env``), else off (``config:none``)
    * ``""`` / ``off`` / ``none`` / ``0`` -> off even when the env var is set (source ``config:off``)
    * ``auto`` -> ``/dev/shm/hero_corpus`` when ``/dev/shm`` can hold ``AUTO_SHM_HEADROOM x`` the corpus bytes, else
      :func:`default_disk_cache_dir` (``$HERO_ROOT/data/_shared_cache``; page-cache backed)
    * anything else -> that path (``~`` expanded)"""
    source = "config"
    value: str | None = None if spec is None else str(spec)
    if value is None and use_env:
        env = os.environ.get(ENV_CACHE_DIR)
        if env is not None and env.strip():
            value, source = env, "env"
    if value is None:
        return None, "config:none"
    stripped = value.strip()
    if stripped.lower() in OFF_VALUES:
        return None, f"{source}:off"
    if stripped.lower() == AUTO_VALUE:
        need = None
        files = motion_files() if callable(motion_files) else motion_files
        if files:
            try:
                need = int(AUTO_SHM_HEADROOM * corpus_bytes(files))
            except OSError:
                need = None
        shm_free = free_bytes(DEV_SHM) if DEV_SHM.is_dir() else None
        if shm_free is not None and (need is None or shm_free >= need):
            logger.info(f"shared corpus cache: auto -> /dev/shm ({shm_free / 1e9:.0f} GB free{'' if need is None else f', need {need / 1e9:.0f} GB'})")
            return DEV_SHM / "hero_corpus", f"{source}:auto:/dev/shm"
        disk = default_disk_cache_dir()
        logger.info(
            f"shared corpus cache: auto -> {disk} (page cache; /dev/shm "
            f"{'absent' if shm_free is None else f'{shm_free / 1e9:.0f} GB free'}{'' if need is None else f' < {need / 1e9:.0f} GB needed'})"
        )
        return disk, f"{source}:auto:disk"
    return Path(os.path.expanduser(stripped)), source


# ----------------------------------------------------------------------------------------------------------------------
# maintenance: list / clean
# ----------------------------------------------------------------------------------------------------------------------
def is_entry_dir(path: Path) -> bool:
    """A directory a writer produced (or was producing): ``meta.json`` / ``READY`` / ``arrays/`` inside, or a ``.building.<pid>``
    scratch dir.  ``clean`` never removes a directory that fails this test, even when its name is a 32-hex key."""
    if not path.is_dir():
        return False
    m = KEY_RE.match(path.name)
    if m is None:
        return False
    if m.group("suffix"):
        return m.group("suffix").startswith(".building.")
    return any((path / n).exists() for n in (META_NAME, READY_NAME, ARRAYS_DIR))


def _probe_lock(entry: CacheEntry) -> tuple[bool | None, str | None]:
    """``(locked, error)``: ``locked`` None when the filesystem refused to lock (error text carried)."""
    try:
        fd = _try_lock(entry.lock_path)
    except (SharedCacheError, OSError) as exc:
        return None, str(exc)
    if fd is None:
        return True, None
    _unlock(fd)
    return False, None


def list_entries(base_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """One record per entry dir / scratch dir under ``base_dir`` (names of the cache's own shape only; other children are ignored)."""
    base = Path(os.path.expanduser(str(base_dir)))
    out: list[dict[str, Any]] = []
    if not base.is_dir():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        m = KEY_RE.match(d.name)
        if m is None or (m.group("suffix") and not m.group("suffix").startswith(".building.")):
            continue
        key = m.group("key")
        entry = CacheEntry(base, key)
        size = entry_size_bytes(d)
        rec: dict[str, Any] = {"path": str(d), "key": key, "building": bool(m.group("suffix")), "size_bytes": size, "size_gb": round(size / 1e9, 3)}
        if not rec["building"]:
            rec["ready"] = entry.is_ready()
            try:
                meta = entry.read_meta()
                rec.update(
                    motion_dir=meta.get("motion_dir"),
                    clips=len(meta.get("clips", [])),
                    frames=meta.get("time_step_total"),
                    created_utc=meta.get("created_utc"),
                    robot_bodies=len(meta.get("params", {}).get("robot_body_names", []) or []),
                )
            except (OSError, ValueError):
                pass
        rec["locked"], lock_error = _probe_lock(entry)
        if lock_error:
            rec["lock_error"] = lock_error
        out.append(rec)
    return out


def foreign_children(base_dir: str | os.PathLike) -> list[Path]:
    """Children of ``base_dir`` that are NOT of the cache's shape (corpora, other caches' per-corpus dirs, READMEs ...)."""
    base = Path(os.path.expanduser(str(base_dir)))
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if KEY_RE.match(p.name) is None)


def clean(base_dir: str | os.PathLike, *, key: str | None = None, force: bool = False, dry_run: bool = False) -> list[Path]:
    """Remove cache entries (all keys, or one) with their scratch dirs / markers / lock files.

    Only names of the cache's own shape are considered (:data:`KEY_RE`), and an entry directory is removed only when it looks
    like one (:func:`is_entry_dir`): ``--clean $HERO_ROOT/data`` or ``--clean <the parent of the per-corpus cache dirs>``
    leaves everything there alone (the skipped children are logged).  ``key`` must be a bare 32-hex key (``ValueError``).
    An entry whose lock is held (a build in flight) is skipped unless ``force``.  Deleting an entry that other processes
    still have mmapped is safe on Linux (the pages live until the last mapping goes away)."""
    base = Path(os.path.expanduser(str(base_dir)))
    removed: list[Path] = []
    if key is not None:
        m = KEY_RE.match(str(key))
        if m is None or m.group("suffix"):
            raise ValueError(f"--key {key!r} is not a cache key (32 hex chars; see --list)")
    if not base.is_dir():
        return removed
    if key is not None:
        keys = {str(key)}
    else:
        keys = set()
        for child in base.iterdir():
            m = KEY_RE.match(child.name)
            if m is not None:
                keys.add(m.group("key"))
        foreign = foreign_children(base)
        if foreign:
            logger.info(
                f"shared corpus cache: {len(foreign)} child(ren) of {base} are not cache entries and are left alone: "
                f"{', '.join(p.name for p in foreign[:8])}{' ...' if len(foreign) > 8 else ''}"
            )
    for k in sorted(keys):
        entry = CacheEntry(base, k)
        fd = _try_lock(entry.lock_path)
        if fd is None and not force:
            logger.warning(f"shared corpus cache: {entry.dir} is locked (build in flight); skipped (use --force)")
            continue
        try:
            targets: list[Path] = []
            if entry.dir.exists():
                if is_entry_dir(entry.dir):
                    targets.append(entry.dir)
                else:
                    logger.warning(f"shared corpus cache: {entry.dir} has a key-shaped name but no {META_NAME} / {READY_NAME} / {ARRAYS_DIR}/; left alone")
            targets.append(entry.failed_path)
            targets.extend(entry.building_dirs())
            for t in targets:
                if not t.exists():
                    continue
                removed.append(t)
                if not dry_run:
                    if t.is_dir():
                        shutil.rmtree(t, ignore_errors=True)
                    else:
                        t.unlink(missing_ok=True)
            if entry.lock_path.exists():
                removed.append(entry.lock_path)
                if not dry_run:
                    entry.lock_path.unlink(missing_ok=True)
        finally:
            if fd is not None:
                _unlock(fd)
    return removed


# ----------------------------------------------------------------------------------------------------------------------
# micro-benchmark: torch gather on an mmap-backed tensor vs RAM vs np.take
# ----------------------------------------------------------------------------------------------------------------------
def bench_gather(
    size_gb: float, *, bench_dir: str | os.PathLike | None = None, rows: int = 4096, repeats: int = 20, keep: bool = False
) -> dict[str, Any]:
    """Synthetic ``(T, 34, 13)`` float32 timeline of ``size_gb`` GB written with :class:`NpyStreamWriter`; measures per-gather
    wall time (ms) of ``rows`` random frames: torch on the mmap tensor (cold = first touch, warm = page-cached), ``np.take``
    on the memmap + ``torch.as_tensor``, and torch on a fully resident copy when RAM allows.  Prints and returns the table."""
    import torch

    row = (34, 13)
    row_bytes = int(np.prod(row)) * 4
    n_rows = int(size_gb * 1e9 // row_bytes)
    d = Path(bench_dir or tempfile.gettempdir()) / f"hero_shared_cache_bench_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "body.npy"
    res: dict[str, Any] = {"rows_total": n_rows, "row_bytes": row_bytes, "size_gb": round(n_rows * row_bytes / 1e9, 2), "gather_rows": rows, "file": str(path)}
    try:
        t0 = time.perf_counter()
        w = NpyStreamWriter(path, row)
        chunk = 1 << 16
        rng = np.random.default_rng(0)
        block = rng.standard_normal((chunk, *row), dtype=np.float32)
        done = 0
        while done < n_rows:
            n = min(chunk, n_rows - done)
            w.append(block[:n])
            done += n
        w.close()
        res["write_s"] = round(time.perf_counter() - t0, 2)
        try:  # drop the just-written pages from the page cache so the first gather is a real cold read (Linux only)
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
            finally:
                os.close(fd)
        except (AttributeError, OSError):
            pass

        gen = torch.Generator().manual_seed(0)
        idx_sets = [torch.randint(0, n_rows, (rows,), generator=gen) for _ in range(repeats)]

        mm = np.load(path, mmap_mode="c")
        tm = torch.from_numpy(mm)
        t0 = time.perf_counter()
        _ = tm[idx_sets[0]]
        res["torch_mmap_cold_ms"] = round(1e3 * (time.perf_counter() - t0), 2)
        times = []
        for idx in idx_sets[1:]:
            t0 = time.perf_counter()
            _ = tm[idx]
            times.append(time.perf_counter() - t0)
        res["torch_mmap_random_pages_ms"] = round(1e3 * float(np.median(times)), 2)  # every set touches new pages (cold-ish)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = tm[idx_sets[1]]
            times.append(time.perf_counter() - t0)
        res["torch_mmap_warm_ms"] = round(1e3 * float(np.median(times)), 2)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = torch.as_tensor(np.take(mm, idx_sets[1].numpy(), axis=0))
            times.append(time.perf_counter() - t0)
        res["np_take_warm_ms"] = round(1e3 * float(np.median(times)), 2)
        # whole-array pass (what _prepare_clip_kinematic_stats does once at setup)
        t0 = time.perf_counter()
        _ = torch.linalg.norm(tm[:, 0, :2], dim=-1).cumsum(0)
        res["torch_mmap_full_pass_s"] = round(time.perf_counter() - t0, 2)

        avail = None
        try:
            avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")  # type: ignore[arg-type]
        except (ValueError, OSError, AttributeError):
            pass
        if avail is None or avail > 1.5 * n_rows * row_bytes:
            ram = torch.from_numpy(np.ascontiguousarray(np.load(path)))
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = ram[idx_sets[1]]
                times.append(time.perf_counter() - t0)
            res["torch_ram_ms"] = round(1e3 * float(np.median(times)), 2)
            del ram
        else:
            res["torch_ram_ms"] = None
        print(json.dumps(res, indent=1))
        return res
    finally:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def _resolve_build_params(exp_name: str | None, strict: bool | None) -> tuple[list[str], list[str], bool, str]:
    """``(robot body names, joint names, strict_extension_keys, note)`` the training ranks of ``exp_name`` use .  ``ValueError`` for an
    unregistered preset."""
    from hero_isaacsim.constants import DOF_NAMES, HERO_BODY_NAMES_34

    body_names = list(HERO_BODY_NAMES_34)
    strict_keys = True if strict is None else strict
    note = "default HERO 34-body Dex3 plant"
    if exp_name:
        from hero_isaacsim.config_values import command as hero_command
        from hero_isaacsim.config_values import experiment as E

        table = {**getattr(E, "DEFAULTS", {}), **getattr(E, "ABLATION_PRESETS", {})}
        exp = table.get(exp_name)
        if exp is None:
            raise ValueError(f"preset {exp_name!r} is not registered")
        body_names = list(exp.robot.body_names)
        mc = hero_command.get_motion_config(exp.command)
        if strict is None:
            strict_keys = bool(getattr(mc, "strict_extension_keys", True))
        note = f"preset {exp_name} ({getattr(exp.robot.asset, 'urdf_file', '?')})"
    return body_names, list(DOF_NAMES), strict_keys, note


def probe(
    motion_dir: str, cache_dir: str | os.PathLike, *, body_names: Sequence[str], joint_names: Sequence[str], strict: bool, fps: float
) -> dict[str, Any]:
    """What the ranks of a run will find under ``cache_dir`` for ``motion_dir`` -- WITHOUT loading the corpus.

    ``key`` (same ``cache_params`` as the loader), ``ready`` (THIS key complete), ``problems`` (why not), the corpus / free /
    needed GB, and ``other_entries`` (READY or partial entries of OTHER keys in the dir -- stale after a corpus re-sync or a
    preset change -- with the exact ``--clean`` command for each).  The launcher skips its free-space guard only on
    ``ready``."""
    from hero_isaacsim.managers.command.loader import (
        BASE_TIMELINE_KEYS,
        EXTENSION_TIMELINE_KEYS,
        HERO_BODY_NAME_ALIASES,
        OBJECT_TIMELINE_KEYS,
        HeroMultiMotionLoader,
    )

    base = Path(os.path.abspath(os.path.expanduser(str(cache_dir))))
    files = HeroMultiMotionLoader._discover_motion_files(motion_dir)
    params = HeroMultiMotionLoader.cache_params(
        robot_body_names=body_names, robot_joint_names=joint_names, body_name_aliases=HERO_BODY_NAME_ALIASES,
        strict_extension_keys=strict, expected_fps=fps,
    )
    key = corpus_cache_key(files, params=params) if files else None
    entry = CacheEntry(base, key, required_arrays=(*BASE_TIMELINE_KEYS, *EXTENSION_TIMELINE_KEYS), object_arrays=OBJECT_TIMELINE_KEYS) if key else None
    problems = entry.problems() if entry else ["no .npz files found"]


    num_clips: int | None = None
    skipped_files: list | None = None
    if entry is not None and not problems:
        try:
            meta = entry.read_meta()
            num_clips = len(meta.get("clips") or [])
            skipped_files = [list(x) for x in (meta.get("skipped_files") or [])]
        except (OSError, ValueError, TypeError) as exc:  # a READY entry validate_meta accepted; report, do not fail the probe
            problems = [f"meta.json clips / skipped_files unreadable: {exc}"]
    cbytes = corpus_bytes(files) if files else 0
    need_bytes = 0 if not problems else int(1.1 * cbytes)
    free = free_bytes(base)
    others = []
    for e in list_entries(base):
        if e["key"] == key:
            continue
        others.append({**e, "clean_cmd": f"python -m hero_isaacsim.managers.command.shared_cache --clean {base} --key {e['key']}"})
    return {
        "motion_dir": motion_dir,
        "num_files": len(files),
        "cache_dir": str(base),
        "key": key,
        "entry_dir": None if entry is None else str(entry.dir),
        "ready": not problems,
        "problems": problems,
        "num_clips": num_clips,
        "skipped_files": skipped_files,
        "entry_size_bytes": entry_size_bytes(entry.dir) if entry is not None and entry.dir.is_dir() else 0,
        "entry_size_gb": round(entry_size_bytes(entry.dir) / 1e9, 3) if entry is not None and entry.dir.is_dir() else 0.0,
        "corpus_bytes": int(cbytes),
        "need_bytes": need_bytes,
        "free_bytes": free,
        "corpus_gb": round(cbytes / 1e9, 6),
        "need_gb": round(need_bytes / 1e9, 6),
        "free_gb": None if free is None else round(free / 1e9, 6),
        "other_entries": others,
        "other_bytes": int(sum(int(e.get("size_bytes") or 0) for e in others)),
        "other_gb": round(sum(int(e.get("size_bytes") or 0) for e in others) / 1e9, 3),
        "params": params,
    }


def _cli_probe(motion_dir: str, cache_spec: str, exp_name: str | None, fps: float, strict: bool | None) -> int:
    from hero_isaacsim.managers.command.loader import HeroMultiMotionLoader

    try:
        body_names, joint_names, strict_keys, note = _resolve_build_params(exp_name, strict)
    except ValueError as exc:
        print(f"[shared_cache] {exc}", file=sys.stderr)
        return 2
    base, source = resolve_cache_dir(cache_spec, motion_files=lambda: HeroMultiMotionLoader._discover_motion_files(motion_dir), use_env=False)
    if base is None:
        print(f"[shared_cache] --cache-dir resolves to OFF ({source}); nothing to probe", file=sys.stderr)
        return 2
    rec = probe(motion_dir, base, body_names=body_names, joint_names=joint_names, strict=strict_keys, fps=fps)
    rec.update(cache_dir_source=source, note=note, exp=exp_name, fps=float(fps), strict_extension_keys=strict_keys)
    print(json.dumps(_jsonable(rec), sort_keys=True))
    return 0


def _cli_build(motion_dir: str, cache_dir: str, exp_name: str | None, fps: float, strict: bool | None) -> int:
    """Pre-build (or verify) the cache the training ranks will mmap; the preset supplies the robot body list.

    Checks the free space (``1.1 x`` the corpus bytes) BEFORE streaming when the key is not READY yet, so a full disk fails in
    seconds, not after the whole corpus has been read (rc 3)."""
    from hero_isaacsim.managers.command.loader import HERO_BODY_NAME_ALIASES, HeroMultiMotionLoader

    try:
        body_names, joint_names, strict_keys, note = _resolve_build_params(exp_name, strict)
    except ValueError as exc:
        print(f"[shared_cache] {exc}", file=sys.stderr)
        return 2
    rec = probe(motion_dir, cache_dir, body_names=body_names, joint_names=joint_names, strict=strict_keys, fps=fps)
    if rec["key"] is None:
        print(f"[shared_cache] no .npz files under {motion_dir}", file=sys.stderr)
        return 2
    print(
        f"[shared_cache] build {motion_dir} ({rec['num_files']} files, {rec['corpus_gb']:.2f} GB) -> {cache_dir} key {rec['key']} with {note}: "
        f"{len(body_names)} bodies, strict_extension_keys={strict_keys}, fps {fps}; READY={rec['ready']}"
        f"{'' if rec['ready'] else ' (' + '; '.join(rec['problems']) + ')'}"
    )
    if not rec["ready"] and rec["free_bytes"] is not None and rec["free_bytes"] < rec["need_bytes"]:
        stale = "".join(f"\n[shared_cache]   {e['clean_cmd']}  ({e.get('size_gb', 0):.2f} GB, {'READY' if e.get('ready') else 'partial'})" for e in rec["other_entries"])
        print(
            f"[shared_cache] {rec['free_gb']:.1f} GB free under {cache_dir} < {rec['need_gb']:.1f} GB needed for a fresh cache "
            f"(corpus {rec['corpus_gb']:.1f} GB x 1.1); nothing written."
            f"{' Stale entries of other keys occupy ' + format(rec['other_gb'], '.1f') + ' GB:' + stale if rec['other_entries'] else ''}",
            file=sys.stderr,
        )
        return 3
    t0 = time.perf_counter()
    ml = HeroMultiMotionLoader(
        motion_dir,
        body_names,
        joint_names,
        device="cpu",
        expected_fps=float(fps),
        storage_device="cpu",
        strict_extension_keys=strict_keys,
        body_name_aliases=HERO_BODY_NAME_ALIASES,
        shared_cache_dir=cache_dir,
    )
    entry = ml.shared_cache_entry
    assert entry is not None
    print(
        f"[shared_cache] {ml.shared_cache_role}: {entry.dir} ({entry_size_bytes(entry.dir) / 1e9:.2f} GB, {ml.time_step_total} frames, "
        f"{ml.num_motions} clips, {ml.num_skipped} skipped) in {time.perf_counter() - t0:.1f} s"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m hero_isaacsim.managers.command.shared_cache", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", metavar="DIR", help="list the cache entries under DIR")
    g.add_argument("--clean", metavar="DIR", help="remove cache entries under DIR (all keys, or --key); other children of DIR are left alone")
    g.add_argument("--probe", metavar="MOTION_DIR", help="print JSON: the key the ranks compute for MOTION_DIR, whether it is READY under --cache-dir, free / needed GB, stale keys")
    g.add_argument("--build", metavar="MOTION_DIR", help="pre-build the cache of MOTION_DIR into --cache-dir")
    g.add_argument("--bench-gb", type=float, metavar="GB", help="micro-benchmark: torch gather on an mmap-backed synthetic timeline of GB gigabytes")
    ap.add_argument("--key", default=None, help="--clean: only this key (32 hex chars, see --list)")
    ap.add_argument("--force", action="store_true", help="--clean: also remove entries whose lock is held")
    ap.add_argument("--dry-run", action="store_true", help="--clean: print what would be removed")
    ap.add_argument("--cache-dir", default=None, help="--build/--probe: shared cache base dir (default: $HERO_SHARED_CACHE_DIR or 'auto')")
    ap.add_argument("--exp", default=None, help="--build/--probe: preset whose robot body list / strictness to use")
    ap.add_argument("--fps", type=float, default=50.0, help="--build/--probe: expected clip frame rate")
    ap.add_argument("--strict", dest="strict", action="store_true", default=None, help="--build/--probe: strict extension keys")
    ap.add_argument("--no-strict", dest="strict", action="store_false", help="--build/--probe: zero-fill missing extension keys")
    ap.add_argument("--bench-dir", default=None, help="--bench-gb: directory for the synthetic file (default: tempdir)")
    ap.add_argument("--rows", type=int, default=4096, help="--bench-gb: frames per gather (env count per rank)")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--keep", action="store_true", help="--bench-gb: keep the synthetic file")
    args = ap.parse_args(argv)

    if args.list is not None:
        entries = list_entries(args.list)
        if not entries:
            print(f"[shared_cache] no entries under {args.list}")
        for e in entries:
            print(json.dumps(e, sort_keys=True))
        return 0
    if args.clean is not None:
        try:
            removed = clean(args.clean, key=args.key, force=args.force, dry_run=args.dry_run)
        except ValueError as exc:
            print(f"[shared_cache] {exc}", file=sys.stderr)
            return 2
        except SharedCacheError as exc:
            print(f"[shared_cache] {exc}", file=sys.stderr)
            return 1
        for p in removed:
            print(f"[shared_cache] {'would remove' if args.dry_run else 'removed'} {p}")
        foreign = foreign_children(args.clean) if args.key is None else []
        if foreign:
            print(f"[shared_cache] left alone ({len(foreign)} non-cache child(ren) of {args.clean}): {' '.join(p.name for p in foreign[:8])}{' ...' if len(foreign) > 8 else ''}")
        print(f"[shared_cache] {len(removed)} path(s) {'listed' if args.dry_run else 'removed'} under {args.clean}")
        return 0
    spec = args.cache_dir if args.cache_dir is not None else (os.environ.get(ENV_CACHE_DIR) or AUTO_VALUE)
    if args.probe is not None:
        return _cli_probe(args.probe, spec, args.exp, args.fps, args.strict)
    if args.build is not None:
        from hero_isaacsim.managers.command.loader import HeroMultiMotionLoader

        base, source = resolve_cache_dir(spec, motion_files=lambda: HeroMultiMotionLoader._discover_motion_files(args.build), use_env=False)
        if base is None:
            print(f"[shared_cache] --cache-dir resolves to OFF ({source}); nothing to build", file=sys.stderr)
            return 2
        try:
            return _cli_build(args.build, str(base), args.exp, args.fps, args.strict)
        except (SharedCacheError, SharedCacheTimeout) as exc:
            print(f"[shared_cache] {exc}", file=sys.stderr)
            return 1
    bench_gather(args.bench_gb, bench_dir=args.bench_dir, rows=args.rows, repeats=args.repeats, keep=args.keep)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
