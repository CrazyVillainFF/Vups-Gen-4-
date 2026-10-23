from typing import Literal
from pydantic import BaseModel, Field, field_validator
Skill = Literal["auto", "coding", "reasoning", "research", "teacher", "design", "vision", "empathetic"]
ImageMode = Literal["describe", "ocr", "ui_review", "code"]
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    conversation_id: str = Field(default="default", min_length=1, max_length=128)
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)
    skill: Skill = "auto"
    image_data_url: str | None = None
    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip(): raise ValueError("Message cannot be blank")
        return value.strip()
class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    skill: str
    model: str

class WorkspaceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class WorkspaceInfo(BaseModel):
    name: str
    file_count: int
    size_bytes: int
    updated_at: str


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceInfo]


class WorkspaceFileEntry(BaseModel):
    path: str
    size: int
    language: str


class WorkspaceFileContent(BaseModel):
    path: str
    content: str
    size: int
    language: str


class WorkspaceFileSaveRequest(BaseModel):
    path: str = Field(min_length=1, max_length=240)
    content: str = Field(max_length=1_000_000)


class WorkspaceFileSaveResponse(BaseModel):
    path: str
    size: int
    language: str
    saved: bool


class WorkspaceProposeRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    context_files: list[str] = Field(default_factory=list, max_length=50)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class ProposedFileInfo(BaseModel):
    path: str
    content: str
    action: str
    size: int
    language: str


class WorkspaceProposalResponse(BaseModel):
    proposal_id: str
    workspace: str
    summary: str
    files: list[ProposedFileInfo]
    model: str
    requires_overwrite_approval: bool
    warnings: list[str]
    created_count: int
    updated_count: int


class WorkspaceApplyRequest(BaseModel):
    approved: bool = False
    approve_overwrites: bool = False


class WorkspaceApplyResponse(BaseModel):
    created: list[str]
    modified: list[str]
    summary: str


class WorkspaceAskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    context_files: list[str] = Field(default_factory=list, max_length=50)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class WorkspaceAskResponse(BaseModel):
    workspace: str
    answer: str
    model: str
    context_files: list[str]
    warnings: list[str]


class ImageInfo(BaseModel):
    media_type: str
    format: str
    width: int
    height: int
    size_bytes: int


class ImageAnalyzeRequest(BaseModel):
    image_data_url: str = Field(min_length=20, max_length=20_000_000)
    prompt: str = Field(
        default="Describe this image in detail.",
        min_length=1,
        max_length=4_000,
    )
    mode: ImageMode = "describe"
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be blank")
        return value.strip()


class ImageAnalyzeResponse(BaseModel):
    answer: str
    model: str
    mode: str
    image: ImageInfo


class ImageGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    negative_prompt: str = Field(default="", max_length=2_000)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    steps: int = Field(default=20, ge=1, le=60)
    cfg: float = Field(default=7.0, ge=0.0, le=30.0)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)

    @field_validator("prompt")
    @classmethod
    def reject_blank_generation_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be blank")
        return value.strip()

    @field_validator("negative_prompt")
    @classmethod
    def strip_negative_prompt(cls, value: str) -> str:
        return value.strip()

    @field_validator("width", "height")
    @classmethod
    def require_multiple_of_eight(cls, value: int) -> int:
        if value % 8:
            raise ValueError("Image width and height must be multiples of 8")
        return value


class ImageGenerateResponse(BaseModel):
    image_id: str
    media_type: str
    format: str
    width: int
    height: int
    size_bytes: int
    seed: int
    checkpoint: str
    created_at: str
    prompt: str
    negative_prompt: str

