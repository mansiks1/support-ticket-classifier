from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr | list[StrictStr]

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        texts = [value] if isinstance(value, str) else value
        if not 1 <= len(texts) <= 64:
            raise ValueError("Batch must contain 1..64 texts")
        if any(not text.strip() or len(text) > 4000 for text in texts):
            raise ValueError("Each text must contain 1..4000 characters and not be blank")
        return value


class Prediction(BaseModel):
    category: str | None
    suggested_category: str
    confidence: float = Field(ge=0, le=1)
    priority: None = None
    requires_human: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]
