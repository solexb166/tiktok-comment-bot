"""Storage for the agent registry.

Two kinds of data:
  * blobs   - large binary payloads (checkpoints, decision logs, training history), stored
              content-addressed by SHA-256. A blob is written once and never changes; storing
              the same bytes twice keeps one copy (no repository growth for duplicates).
  * records - small JSON documents (identity, versions, evaluations), written once; and
              append-only JSON-lines event logs (lifecycle).

`BlobStore` is the only thing that knows where blobs live. `LocalBlobStore` keeps them in a
directory tracked by git (temporary choice); a dedicated artifact store can replace it by
implementing the same four methods. Records refer to blobs only by SHA-256, so moving blobs
to another store changes no record.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class RegistryError(Exception):
    pass


class WriteOnceError(RegistryError):
    """Raised when something tries to overwrite an immutable record or blob."""


class IntegrityError(RegistryError):
    """Raised when stored bytes no longer match their recorded checksum."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class BlobStore:
    """Interface: put(bytes, ext) -> sha, get(sha) -> bytes, exists(sha), locate(sha) -> str."""

    def put(self, data: bytes, ext: str = "bin") -> str:
        raise NotImplementedError

    def get(self, sha: str) -> bytes:
        raise NotImplementedError

    def exists(self, sha: str) -> bool:
        raise NotImplementedError

    def locate(self, sha: str) -> str:
        raise NotImplementedError


class LocalBlobStore(BlobStore):
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, sha: str, ext: str | None = None) -> Path:
        d = self.root / sha[:2] / sha[2:4]
        if ext is not None:
            return d / f"{sha}.{ext}"
        hits = list(d.glob(f"{sha}.*")) if d.exists() else []
        if not hits:
            raise FileNotFoundError(f"blob {sha} not found in {self.root}")
        return hits[0]

    def put(self, data: bytes, ext: str = "bin") -> str:
        sha = sha256_bytes(data)
        if self.exists(sha):
            if sha256_file(self._path(sha)) != sha:
                raise IntegrityError(f"stored blob {sha} is corrupted")
            return sha                                  # identical content: one copy
        _atomic_write(self._path(sha, ext), data)
        return sha

    def put_file(self, path: Path, ext: str | None = None) -> str:
        return self.put(Path(path).read_bytes(), ext or Path(path).suffix.lstrip(".") or "bin")

    def get(self, sha: str) -> bytes:
        data = self._path(sha).read_bytes()
        if sha256_bytes(data) != sha:
            raise IntegrityError(f"blob {sha} does not match its checksum")
        return data

    def exists(self, sha: str) -> bool:
        d = self.root / sha[:2] / sha[2:4]
        return d.exists() and any(d.glob(f"{sha}.*"))

    def locate(self, sha: str) -> str:
        return str(self._path(sha))

    def verify_all(self) -> list[str]:
        """Checksums of every stored blob that no longer matches its name (should be empty)."""
        bad = []
        for p in self.root.rglob("*.*"):
            if p.name.startswith(".tmp-"):
                continue
            sha = p.name.split(".")[0]
            if sha256_file(p) != sha:
                bad.append(sha)
        return bad


def canonical_json(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=1, default=_json_default) + "\n").encode()


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_once_json(path: Path, obj) -> str:
    """Write an immutable JSON record. Refuses to overwrite; returns the record's SHA-256."""
    path = Path(path)
    data = canonical_json(obj)
    if path.exists():
        raise WriteOnceError(f"{path} already exists and is immutable")
    _atomic_write(path, data)
    return sha256_bytes(data)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, obj) -> None:
    """Append one event to an append-only log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True, default=_json_default)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
