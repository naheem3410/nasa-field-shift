from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=3000)
    context: dict | None = None  # previous ExtractionResponse to merge with


class ExtractionResponse(BaseModel):
    # Fields ready to POST to /api/v1/advice on the live API
    state: str | None = None
    lga: str | None = None
    seasons: list[list[str]] | None = None
    priorities: list[str] | None = None
    irrigated: bool | None = None
    striga: bool | None = None
    waterlogged: bool | None = None
    enough_workers: bool | None = None
    easy_seed: bool | None = None
    tenure: str | None = None

    # Metadata for the caller
    missing: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    transcript: str = ""
    confidence: str = "unclear"  # "complete" | "partial" | "unclear"

    # Populated automatically when confidence is "complete"
    advice_id: str | None = None
    advice_text: str | None = None   # one-page plain-text advice
    advice_page: dict | None = None  # full structured page object
    advice_error: str | None = None  # set if the advice API call failed


class HealthResponse(BaseModel):
    status: str = "ok"
    meta_loaded: bool = False
