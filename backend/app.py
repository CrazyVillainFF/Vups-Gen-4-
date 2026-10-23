from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from .config import ENV_FILE, Settings, settings
from .image_generation import (
    ComfyGenerationRequest,
    ImageGenerationError,
    ImageGenerationNotConfigured,
    ImageGenerator,
)
from .image_understanding import (
    ImageAnalyzer,
    VisionNotConfigured,
    VisionUnavailable,
)
from .image_validation import ImageValidationError
from .provider import ChatProvider, ModelNotConfigured, ModelUnavailable
from .schemas import (
    ChatRequest,
    ChatResponse,
    ImageAnalyzeRequest,
    ImageAnalyzeResponse,
    ImageGenerateRequest,
    ImageGenerateResponse,
    ImageInfo,
    ProposedFileInfo,
    WorkspaceApplyRequest,
    WorkspaceApplyResponse,
    WorkspaceAskRequest,
    WorkspaceAskResponse,
    WorkspaceCreateRequest,
    WorkspaceFileContent,
    WorkspaceFileEntry,
    WorkspaceFileSaveRequest,
    WorkspaceFileSaveResponse,
    WorkspaceInfo,
    WorkspaceListResponse,
    WorkspaceProposalResponse,
    WorkspaceProposeRequest,
)
from .skills import build_system_prompt
from .store import Store, create_store
from .workspace import (
    ApprovalRequired,
    BinaryFile,
    FileTooLarge,
    InvalidPath,
    InvalidProposal,
    InvalidWorkspaceName,
    LimitExceeded,
    OverwriteApprovalRequired,
    ProposalExpired,
    ProposalNotFound,
    ProposalStore,
    SensitivePath,
    WorkspaceError,
    WorkspaceExists,
    WorkspaceNotFound,
    WorkspaceService,
)
from .workspace_generation import ModelOutputError, ProjectGenerator

LOG = logging.getLogger("vups_gen4")

_WORKSPACE_STATUS = {
    WorkspaceNotFound: 404,
    WorkspaceExists: 409,
    InvalidWorkspaceName: 400,
    InvalidPath: 400,
    SensitivePath: 403,
    FileTooLarge: 413,
    BinaryFile: 415,
    LimitExceeded: 422,
    InvalidProposal: 422,
    ApprovalRequired: 409,
    OverwriteApprovalRequired: 409,
    ProposalNotFound: 404,
    ProposalExpired: 410,
}


def _raise_workspace_error(exc: WorkspaceError) -> None:
    status = _WORKSPACE_STATUS.get(type(exc), 400)
    raise HTTPException(
        status,
        detail={"code": exc.code, "message": str(exc)},
    ) from exc


def _raise_vision_error(
    exc: VisionNotConfigured | VisionUnavailable, config: Settings
) -> None:
    if isinstance(exc, VisionNotConfigured):
        raise HTTPException(
            503,
            detail={
                "code": exc.code,
                "message": str(exc),
                "missing": config.missing_vision_settings,
            },
        ) from exc
    LOG.warning("Vision request failed: %s", exc)
    raise HTTPException(
        502,
        detail={"code": exc.code, "message": str(exc)},
    ) from exc


def _raise_image_error(exc: ImageValidationError | ImageGenerationError) -> None:
    if isinstance(exc, ImageValidationError):
        raise HTTPException(
            exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if not isinstance(exc, ImageGenerationNotConfigured):
        LOG.warning("Image generation failed: %s", exc)
    raise HTTPException(
        exc.status_code,
        detail={"code": exc.code, "message": str(exc)},
    ) from exc


def _raise_model_error(exc: ModelNotConfigured | ModelUnavailable, config: Settings) -> None:
    if isinstance(exc, ModelNotConfigured):
        raise HTTPException(
            503,
            detail={
                "code": exc.code,
                "message": str(exc),
                "missing": config.missing_model_settings,
            },
        ) from exc
    LOG.warning("Model request failed: %s", exc)
    raise HTTPException(
        502,
        detail={"code": exc.code, "message": str(exc)},
    ) from exc


def create_app(
    config: Settings | None = None,
    provider: ChatProvider | None = None,
    store: Store | None = None,
    workspaces: WorkspaceService | None = None,
    proposals: ProposalStore | None = None,
    generator: ProjectGenerator | None = None,
    vision: ImageAnalyzer | None = None,
    image_generator: ImageGenerator | None = None,
) -> FastAPI:
    config = config or settings
    provider = provider or ChatProvider(config)
    store = store or create_store(config)
    workspaces = workspaces or WorkspaceService.from_settings(config)
    proposals = proposals or ProposalStore()
    generator = generator or ProjectGenerator(
        provider,
        workspaces,
        max_tokens=config.workspace_model_max_tokens,
    )
    vision = vision or ImageAnalyzer(config)
    image_generator = image_generator or ImageGenerator(config)

    app = FastAPI(title=config.app_name, version="0.4.0")
    app.state.workspaces = workspaces
    app.state.proposals = proposals
    app.state.generator = generator
    app.state.vision = vision
    app.state.image_generator = image_generator

    if provider.configured:
        LOG.info(
            "Model configured: %s at %s",
            config.model_name,
            config.model_base_url,
        )
    else:
        LOG.warning(
            "Model is not configured. Missing: %s. Set these in %s and restart the API.",
            ", ".join(config.missing_model_settings),
            ENV_FILE,
        )

    def proposal_response(proposal: Any) -> WorkspaceProposalResponse:
        created = sum(1 for file in proposal.files if file.action == "create")
        updated = len(proposal.files) - created
        return WorkspaceProposalResponse(
            proposal_id=proposal.id,
            workspace=proposal.workspace,
            summary=proposal.summary,
            files=[
                ProposedFileInfo(
                    path=file.path,
                    content=file.content,
                    action=file.action,
                    size=file.size,
                    language=file.language,
                )
                for file in proposal.files
            ],
            model=proposal.model or config.model_name or "unconfigured",
            requires_overwrite_approval=proposal.requires_overwrite_approval,
            warnings=list(proposal.warnings),
            created_count=created,
            updated_count=updated,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": config.app_name}

    @app.get("/ready")
    def ready() -> dict[str, object]:
        try:
            db_ok = store.ping()
        except Exception:
            db_ok = False
        return {
            "ready": db_ok,
            "database": "ok" if db_ok else "unavailable",
            "model_configured": provider.configured,
            "model_status": "configured" if provider.configured else "not_configured",
        }

    @app.get("/model-info")
    def model_info() -> dict[str, object]:
        return {
            "app": config.app_name,
            "provider_configured": provider.configured,
            "model": config.model_name or None,
            "base_url": config.model_base_url or None,
            "endpoint": config.chat_completions_url or None,
            "vision_supported": provider.vision_supported,
            "vision_model": config.vision_model_name or None,
            "vision_endpoint": config.vision_chat_url or None,
            "vision_configured": config.vision_configured,
            "image_generation_provider": config.image_generation_provider,
            "image_generation_configured": config.image_generation_configured,
            "missing_settings": config.missing_model_settings,
            "skills": [
                "coding",
                "reasoning",
                "research",
                "teacher",
                "design",
                "vision",
                "empathetic",
            ],
        }

    @app.get("/model-status")
    async def model_status(live: bool = False) -> dict[str, object]:
        status = await provider.probe(live=live)
        status.update(
            {
                "vision_model": config.vision_model_name or None,
                "vision_configured": config.vision_configured,
            }
        )
        return status

    @app.get("/image-status")
    async def image_status() -> dict[str, object]:
        return {
            "vision": await vision.probe(),
            "generation": await image_generator.probe(),
            "limits": {
                "max_upload_bytes": config.image_max_upload_bytes,
                "max_pixels": config.image_max_pixels,
                "max_dimension": config.image_max_dimension,
            },
        }

    @app.post("/images/analyze", response_model=ImageAnalyzeResponse)
    async def analyze_image(request: ImageAnalyzeRequest) -> ImageAnalyzeResponse:
        try:
            result = await vision.analyze(
                image_data_url=request.image_data_url,
                prompt=request.prompt,
                mode=request.mode,
                history=[message.model_dump() for message in request.history],
            )
        except ImageValidationError as exc:
            _raise_image_error(exc)
        except (VisionNotConfigured, VisionUnavailable) as exc:
            _raise_vision_error(exc, config)
        return ImageAnalyzeResponse(
            answer=result.answer,
            model=result.model,
            mode=result.mode,
            image=ImageInfo(**result.image.as_dict()),
        )

    @app.post("/images/generate", response_model=ImageGenerateResponse)
    async def generate_image(
        request: ImageGenerateRequest,
    ) -> ImageGenerateResponse:
        try:
            result = await image_generator.generate(
                ComfyGenerationRequest(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    steps=request.steps,
                    cfg=request.cfg,
                    seed=request.seed,
                )
            )
        except ImageGenerationError as exc:
            _raise_image_error(exc)
        return ImageGenerateResponse(
            image_id=result.image_id,
            media_type=result.media_type,
            format=result.image_format,
            width=result.width,
            height=result.height,
            size_bytes=result.size_bytes,
            seed=result.seed,
            checkpoint=result.checkpoint,
            created_at=result.created_at,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
        )

    @app.get("/images/generated/{image_id}")
    def generated_image(image_id: str) -> Response:
        stored = image_generator.outputs.get(image_id)
        if stored is None:
            raise HTTPException(
                404,
                detail={
                    "code": "image_not_found",
                    "message": "That generated image does not exist.",
                },
            )
        path, media_type = stored
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise HTTPException(
                404,
                detail={
                    "code": "image_not_found",
                    "message": "That generated image is not readable.",
                },
            ) from exc
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{image_id}"',
                "Cache-Control": "private, max-age=3600",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/workspaces", response_model=WorkspaceListResponse)
    def list_workspaces() -> WorkspaceListResponse:
        try:
            items = workspaces.list_workspaces()
        except (WorkspaceError, OSError) as exc:
            if isinstance(exc, WorkspaceError):
                _raise_workspace_error(exc)
            raise HTTPException(
                500, detail={"code": "workspace_error", "message": "Could not list workspaces."}
            ) from exc
        return WorkspaceListResponse(workspaces=[WorkspaceInfo(**item) for item in items])

    @app.post("/workspaces", response_model=WorkspaceInfo, status_code=201)
    def create_workspace(request: WorkspaceCreateRequest) -> WorkspaceInfo:
        try:
            item = workspaces.create_workspace(request.name)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return WorkspaceInfo(**item)

    @app.get("/workspaces/{workspace}", response_model=WorkspaceInfo)
    def get_workspace(workspace: str) -> WorkspaceInfo:
        try:
            item = workspaces.get_workspace(workspace)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return WorkspaceInfo(**item)

    @app.get("/workspaces/{workspace}/files", response_model=list[WorkspaceFileEntry])
    def list_files(workspace: str) -> list[WorkspaceFileEntry]:
        try:
            items = workspaces.list_files(workspace)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return [WorkspaceFileEntry(**item) for item in items]

    @app.get("/workspaces/{workspace}/file", response_model=WorkspaceFileContent)
    def read_file(
        workspace: str,
        path: str = Query(min_length=1, max_length=240),
    ) -> WorkspaceFileContent:
        try:
            item = workspaces.read_file(workspace, path)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return WorkspaceFileContent(**item)

    @app.put("/workspaces/{workspace}/file", response_model=WorkspaceFileSaveResponse)
    def save_file(
        workspace: str, request: WorkspaceFileSaveRequest
    ) -> WorkspaceFileSaveResponse:
        try:
            item = workspaces.save_file(workspace, request.path, request.content)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return WorkspaceFileSaveResponse(**item)

    @app.post(
        "/workspaces/{workspace}/proposals",
        response_model=WorkspaceProposalResponse,
    )
    async def propose_changes(
        workspace: str, request: WorkspaceProposeRequest
    ) -> WorkspaceProposalResponse:
        if len(request.message) > config.max_message_chars:
            raise HTTPException(413, "Message is too long.")
        try:
            proposal = await generator.propose(
                workspace,
                request.message,
                context_files=request.context_files,
                history=[message.model_dump() for message in request.history],
            )
        except ModelNotConfigured as exc:
            _raise_model_error(exc, config)
        except ModelUnavailable as exc:
            _raise_model_error(exc, config)
        except ModelOutputError as exc:
            raise HTTPException(
                502,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        proposals.add(proposal)
        return proposal_response(proposal)

    @app.post(
        "/workspaces/{workspace}/proposals/{proposal_id}/apply",
        response_model=WorkspaceApplyResponse,
    )
    def apply_changes(
        workspace: str, proposal_id: str, request: WorkspaceApplyRequest
    ) -> WorkspaceApplyResponse:
        try:
            proposal = proposals.get(proposal_id, workspace)
            result = workspaces.apply_proposal(
                workspace,
                proposal,
                approved=request.approved,
                approve_overwrites=request.approve_overwrites,
            )
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        proposals.pop(proposal_id)
        return WorkspaceApplyResponse(**result)

    @app.post("/workspaces/{workspace}/ask", response_model=WorkspaceAskResponse)
    async def ask_workspace(
        workspace: str, request: WorkspaceAskRequest
    ) -> WorkspaceAskResponse:
        if len(request.message) > config.max_message_chars:
            raise HTTPException(413, "Message is too long.")
        try:
            answer, context_files, warnings = await generator.ask(
                workspace,
                request.message,
                context_files=request.context_files,
                history=[message.model_dump() for message in request.history],
            )
        except ModelNotConfigured as exc:
            _raise_model_error(exc, config)
        except ModelUnavailable as exc:
            _raise_model_error(exc, config)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return WorkspaceAskResponse(
            workspace=workspace,
            answer=answer,
            model=config.model_name or "unconfigured",
            context_files=context_files,
            warnings=warnings,
        )

    @app.get("/workspaces/{workspace}/export")
    def export_workspace(workspace: str) -> Response:
        try:
            data, filename = workspaces.export_zip(workspace)
        except WorkspaceError as exc:
            _raise_workspace_error(exc)
        return Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Vups-Warning": "Review generated code before execution.",
            },
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        if len(request.message) > config.max_message_chars:
            raise HTTPException(413, "Message is too long.")
        history = [m.model_dump() for m in request.history]
        if request.image_data_url:
            mode = (
                "ui_review"
                if request.skill == "design"
                else "code"
                if request.skill == "coding"
                else "describe"
            )
            try:
                analysis = await vision.analyze(
                    image_data_url=request.image_data_url,
                    prompt=request.message,
                    mode=mode,
                    history=history,
                )
            except ImageValidationError as exc:
                _raise_image_error(exc)
            except (VisionNotConfigured, VisionUnavailable) as exc:
                _raise_vision_error(exc, config)
            answer = analysis.answer
            response_model = analysis.model
        else:
            try:
                answer = await provider.generate(
                    system_prompt=build_system_prompt(request.skill, False),
                    message=request.message,
                    history=history,
                )
            except ModelNotConfigured as exc:
                _raise_model_error(exc, config)
            except ModelUnavailable as exc:
                _raise_model_error(exc, config)
            response_model = config.model_name or "unconfigured"
        try:
            store.save_turn(
                request.conversation_id,
                request.message,
                answer,
                request.skill,
            )
        except Exception as exc:
            raise HTTPException(
                503,
                "Could not persist conversation; check database configuration.",
            ) from exc
        return ChatResponse(
            conversation_id=request.conversation_id,
            answer=answer,
            skill=request.skill,
            model=response_model,
        )

    @app.get("/conversations/{conversation_id}")
    def conversation(conversation_id: str) -> dict[str, object]:
        try:
            return {
                "conversation_id": conversation_id,
                "messages": store.get_messages(conversation_id),
            }
        except Exception as exc:
            raise HTTPException(503, "Conversation storage is unavailable.") from exc

    return app


app = create_app()
