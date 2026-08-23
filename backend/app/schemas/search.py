from pydantic import BaseModel, ConfigDict, Field


class SearchResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: int
    document_id: int
    chunk_index: int
    filename: str
    content: str
    distance: float = Field(ge=0)
