import json
import logging
import os

import openai

from .models import ExtractionResponse

log = logging.getLogger("nasa.agent")

REQUIRED = {"state", "lga", "seasons", "priorities"}
VALID_TENURE = {"own", "rent", "borrowed"}

SYSTEM_TEMPLATE = """\
You extract structured farm information from a farmer's spoken message.
The farmer is in Nigeria. Extract only what is clearly stated; do not guess.

Valid Nigerian states (use exactly as written):
{state_list}

Valid crop IDs (use these exact IDs when the farmer names a known crop):
{crop_id_list}

Valid priority keys:
nitrogen, pests, water, organic_matter, income, erosion, inputs

Valid tenure values: own, rent, borrowed

Rules:
- seasons: a list of seasons, newest first. Each season is a list of crop IDs or typed names.
  An empty list [] means an explicitly mentioned fallow (land was rested). Up to 4 seasons, up to 4 crops per season.
  Example: "last year maize, year before cowpea, year before that nothing" → [["maize"], ["cowpea"], []]
  IMPORTANT: if the message does not mention seasons at all, return null (not []).
- priorities: list of 1–3 priority keys in the order the farmer mentions importance.
  IMPORTANT: if the message does not mention priorities at all, return null (not []).
- irrigated, striga, waterlogged, enough_workers, easy_seed: true, false, or null (null = not mentioned / not sure).
- tenure: "own", "rent", "borrowed", or null if not mentioned.
- If a state name is a common variant (FCT, Abuja → "Federal Capital Territory"), resolve it.
- If the farmer names a city or town instead of an LGA, use the name as-is.
- If a crop is not in the valid crop IDs list, include the farmer's name as a typed string.

Return a JSON object with these exact keys:
state, lga, seasons, priorities, irrigated, striga, waterlogged, enough_workers, easy_seed, tenure,
missing (list of required fields that could not be found, chosen from: state, lga, seasons, priorities),
clarifying_question (one plain English question to ask the farmer if a required field is missing
or genuinely ambiguous, otherwise null)."""


def _build_system(meta: dict) -> str:
    state_list = ", ".join(meta.get("state_names", []))
    crop_id_list = ", ".join(meta.get("crop_ids", []))
    return SYSTEM_TEMPLATE.format(state_list=state_list, crop_id_list=crop_id_list)


def _confidence(data: dict) -> str:
    missing = data.get("missing") or []
    if not missing:
        return "complete"
    if set(missing) <= {"state", "lga"}:
        return "partial"
    return "unclear"


FARM_FIELDS = ["state", "lga", "seasons", "priorities", "irrigated", "striga",
               "waterlogged", "enough_workers", "easy_seed", "tenure"]


def _is_empty(value) -> bool:
    """True when a value means 'not provided' — None or an empty list."""
    return value is None or value == []


def merge(new: ExtractionResponse, previous: dict) -> ExtractionResponse:
    """Fill empty/None fields in `new` with values from a previous extraction."""
    for field in FARM_FIELDS:
        current = getattr(new, field)
        prev_val = previous.get(field)
        if _is_empty(current) and not _is_empty(prev_val):
            setattr(new, field, prev_val)
    missing = [f for f in REQUIRED if getattr(new, f) is None]
    new.missing = missing
    new.confidence = "complete" if not missing else ("partial" if set(missing) <= {"state", "lga"} else "unclear")
    if not missing:
        new.clarifying_question = None
    return new


def extract(transcript: str, meta: dict, hint_state: str | None = None, hint_lga: str | None = None, previous: dict | None = None) -> ExtractionResponse:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    message = transcript
    if hint_state or hint_lga:
        hints = []
        if hint_state:
            hints.append(f"state: {hint_state}")
        if hint_lga:
            hints.append(f"LGA: {hint_lga}")
        message = f"[Location from GPS: {', '.join(hints)}]\n\n{transcript}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _build_system(meta)},
                {"role": "user", "content": message},
            ],
        )
        data = json.loads(response.choices[0].message.content)
    except Exception as exc:
        log.warning("GPT-4o extraction failed: %s", exc)
        result = ExtractionResponse(
            transcript=transcript,
            missing=list(REQUIRED),
            clarifying_question="I could not understand your message. Could you describe your farm location, what you grew last season, and what matters most to you?",
            confidence="unclear",
        )
        if previous:
            result = merge(result, previous)
        return result

    tenure = data.get("tenure")
    if tenure not in VALID_TENURE:
        tenure = None

    missing = [f for f in (data.get("missing") or []) if f in REQUIRED]
    confidence = _confidence(data)

    result = ExtractionResponse(
        state=data.get("state"),
        lga=data.get("lga"),
        seasons=data.get("seasons"),
        priorities=data.get("priorities"),
        irrigated=data.get("irrigated"),
        striga=data.get("striga"),
        waterlogged=data.get("waterlogged"),
        enough_workers=data.get("enough_workers"),
        easy_seed=data.get("easy_seed"),
        tenure=tenure,
        missing=missing,
        clarifying_question=data.get("clarifying_question"),
        transcript=transcript,
        confidence=confidence,
    )

    if previous:
        result = merge(result, previous)
    return result
