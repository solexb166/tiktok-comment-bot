import hashlib


def event_log_hash(events) -> str:
    h = hashlib.sha256()
    for e in events:
        h.update(e.canonical().encode())
        h.update(b"\n")
    return h.hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
