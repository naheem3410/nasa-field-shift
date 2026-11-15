import logging
import os
import threading
import time

import requests

log = logging.getLogger("nasa.agent")

_lock = threading.Lock()

_meta: dict = {}
_last_fetched: float = 0.0
REFRESH_SECONDS = 86400  # 24 hours


def _fetch(base_url: str) -> dict:
    session = requests.Session()
    questions = session.get(f"{base_url}/meta/questions", timeout=15).json()
    states_raw = session.get(f"{base_url}/places/states", timeout=15).json()

    crop_ids = [c["id"] for c in questions.get("crops", [])]
    priority_keys = [
        o["key"]
        for step in questions.get("steps", [])
        if step.get("field") == "priorities"
        for o in step.get("options", [])
    ]
    state_names = [s["state"] for s in states_raw]

    return {
        "crop_ids": crop_ids,
        "priority_keys": priority_keys,
        "state_names": state_names,
    }


def load(base_url: str | None = None) -> None:
    global _meta, _last_fetched
    url = base_url or os.getenv("NASA_API_BASE_URL", "https://nasa-api-miva-pulsars.duckdns.org/api/v1")
    try:
        data = _fetch(url)
        with _lock:
            _meta = data
            _last_fetched = time.monotonic()
        log.info("meta cache loaded: %d crops, %d priorities, %d states",
                 len(data["crop_ids"]), len(data["priority_keys"]), len(data["state_names"]))
    except Exception as exc:
        log.warning("meta cache fetch failed: %s", exc)


def _refresh_loop(base_url: str) -> None:
    while True:
        time.sleep(REFRESH_SECONDS)
        load(base_url)


def start_background_refresh(base_url: str | None = None) -> None:
    url = base_url or os.getenv("NASA_API_BASE_URL", "https://nasa-api-miva-pulsars.duckdns.org/api/v1")
    t = threading.Thread(target=_refresh_loop, args=(url,), daemon=True)
    t.start()


def get() -> dict:
    with _lock:
        return dict(_meta)


def is_loaded() -> bool:
    with _lock:
        return bool(_meta)
