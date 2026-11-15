import logging
import os

import requests

from .models import ExtractionResponse

log = logging.getLogger("nasa.agent")


def _build_body(result: ExtractionResponse) -> dict:
    body: dict = {
        "state": result.state,
        "lga": result.lga,
        "seasons": result.seasons or [],
        "priorities": result.priorities or [],
    }
    if result.irrigated is not None:
        body["irrigated"] = result.irrigated
    if result.striga is not None:
        body["striga"] = result.striga
    if result.waterlogged is not None:
        body["waterlogged"] = result.waterlogged
    if result.enough_workers is not None:
        body["enough_workers"] = result.enough_workers
    if result.easy_seed is not None:
        body["easy_seed"] = result.easy_seed
    if result.tenure is not None:
        body["tenure"] = result.tenure
    return body


def fetch_advice(result: ExtractionResponse, base_url: str | None = None) -> ExtractionResponse:
    """Call POST /advice on the main API and attach the report to the result."""
    url = (base_url or os.getenv("NASA_API_BASE_URL", "https://nasa-api-miva-pulsars.duckdns.org/api/v1")) + "/advice"
    body = _build_body(result)

    try:
        resp = requests.post(url, json=body, timeout=60)
    except Exception as exc:
        log.error("advice API request failed: %s", exc)
        result.advice_error = f"Could not reach advice service: {exc}"
        return result

    if resp.status_code != 200:
        err = resp.json().get("error", {})
        msg = err.get("message") or resp.text[:200]
        log.warning("advice API returned %d: %s", resp.status_code, msg)
        result.advice_error = msg
        return result

    data = resp.json()
    result.advice_id = data.get("advice_id")
    result.advice_text = data.get("text")
    result.advice_page = data.get("page")
    return result
