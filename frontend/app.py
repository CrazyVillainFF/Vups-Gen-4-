from __future__ import annotations

import base64
import hashlib
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import streamlit as st
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.api_client import (
    chat_timeout_seconds,
    configured_timeout,
    error_detail,
    image_analyze_timeout_seconds,
    request_json,
    vision_timeout_seconds,
)

ENV_VALUES = dotenv_values(ROOT / ".env")

st.set_page_config(
    page_title="Vups Gen 4 · AI Workspace & Image Studio",
    page_icon="✦",
    layout="wide",
)
API = (
    os.getenv("API_BASE_URL")
    or ENV_VALUES.get("API_BASE_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")
VISION_TIMEOUT_SECONDS = vision_timeout_seconds(os.environ, ENV_VALUES)
CHAT_TIMEOUT_SECONDS = chat_timeout_seconds(os.environ, ENV_VALUES)
IMAGE_ANALYZE_TIMEOUT_SECONDS = image_analyze_timeout_seconds(
    os.environ, ENV_VALUES
)
IMAGE_GENERATION_TIMEOUT_SECONDS = configured_timeout(
    os.environ, ENV_VALUES, "IMAGE_GENERATION_TIMEOUT_SECONDS", 360.0
)
SETUP_SNIPPET = (
    "MODEL_BASE_URL=http://127.0.0.1:11434/v1\n"
    "MODEL_NAME=qwen2.5-coder:7b\n"
    "MODEL_API_KEY=ollama"
)
START_SNIPPET = (
    "if (-not (Test-Path .env)) { Copy-Item .env.example .env }\n"
    "ollama serve\n"
    "ollama pull qwen2.5-coder:7b"
)
VISION_SETUP_SNIPPET = (
    "VISION_MODEL_NAME=qwen2.5vl:7b\n"
    "VISION_BASE_URL=http://127.0.0.1:11434\n"
    "VISION_TIMEOUT_SECONDS=900"
)
VISION_START_SNIPPET = "ollama pull qwen2.5vl:7b"
COMFY_SETUP_SNIPPET = (
    "IMAGE_GENERATION_PROVIDER=comfyui\n"
    "COMFYUI_BASE_URL=http://127.0.0.1:8188\n"
    "COMFYUI_CHECKPOINT=v1-5-pruned-emaonly.safetensors\n"
    "IMAGE_OUTPUT_DIR=generated_images"
)
IMAGE_MODES = {
    "Describe": "describe",
    "Read text (OCR)": "ocr",
    "Review interface": "ui_review",
    "Extract code": "code",
}
IMAGE_MODE_PROMPTS = {
    "describe": "Describe this image in detail, including any important text.",
    "ocr": "Transcribe all visible text and preserve its structure.",
    "ui_review": "Review this interface and suggest concrete design improvements.",
    "code": "Extract the code shown in this image and note any uncertain characters.",
}
WORKSPACE_ACTIONS = ["Propose file changes", "Ask about the project"]


@st.cache_data(ttl=8, show_spinner=False)
def backend_status(api: str) -> dict:
    status: dict = {
        "backend": False,
        "error": "",
        "model": None,
        "model_error": "",
    }
    try:
        health = httpx.get(f"{api}/health", timeout=3.0)
        health.raise_for_status()
        status["backend"] = True
    except httpx.HTTPError:
        status["error"] = f"Backend is unreachable at {api}."
        return status
    try:
        model = httpx.get(f"{api}/model-status", timeout=5.0)
        model.raise_for_status()
        status["model"] = model.json()
    except (httpx.HTTPError, ValueError):
        status["model_error"] = "The backend did not return model status."
    return status


def api_request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    timeout: float = 30.0,
    request_label: str = "Request",
    timeout_setting: str | None = None,
) -> tuple[object | None, str, str | None]:
    return request_json(
        API,
        method,
        path,
        params=params,
        json_body=json_body,
        timeout=timeout,
        request_label=request_label,
        timeout_setting=timeout_setting,
    )


def fetch_zip(workspace: str) -> tuple[bytes | None, str, str]:
    try:
        response = httpx.get(
            f"{API}/workspaces/{quote(workspace, safe='')}/export",
            timeout=120.0,
        )
    except httpx.HTTPError:
        return None, "", f"Could not reach the backend at {API}."
    if response.status_code >= 400:
        message, _ = error_detail(response)
        return None, "", message
    filename = f"{workspace}.zip"
    disposition = response.headers.get("content-disposition", "")
    if 'filename="' in disposition:
        filename = disposition.split('filename="', 1)[1].split('"', 1)[0]
    return response.content, filename, ""


@st.cache_data(ttl=8, show_spinner=False)
def image_capabilities(api: str) -> tuple[dict | None, str]:
    try:
        response = httpx.get(f"{api}/image-status", timeout=20.0)
    except httpx.TimeoutException:
        return None, "Image status timed out."
    except httpx.HTTPError:
        return None, f"Could not reach the backend at {api}."
    if response.status_code >= 400:
        message, _ = error_detail(response)
        return None, message
    try:
        payload = response.json()
    except ValueError:
        return None, "The backend returned invalid image status."
    if not isinstance(payload, dict):
        return None, "The backend returned invalid image status."
    return payload, ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_generated_image(api: str, image_id: str) -> tuple[bytes | None, str]:
    try:
        response = httpx.get(
            f"{api}/images/generated/{quote(image_id, safe='')}",
            timeout=60.0,
        )
    except httpx.HTTPError:
        return None, f"Could not reach the backend at {api}."
    if response.status_code >= 400:
        message, _ = error_detail(response)
        return None, message
    return response.content, ""


def image_setup_help() -> None:
    st.code(VISION_SETUP_SNIPPET, language="dotenv")
    st.caption("Pull the vision model, then restart the API:")
    st.code(VISION_START_SNIPPET, language="powershell")


def comfy_setup_help() -> None:
    st.code(COMFY_SETUP_SNIPPET, language="dotenv")
    st.caption(
        "Start ComfyUI, install a checkpoint in its `models/checkpoints` folder, "
        "match `COMFYUI_CHECKPOINT` to that filename, and restart the API."
    )
    st.code("python main.py", language="powershell")


def model_setup_help() -> None:
    st.code(SETUP_SNIPPET, language="dotenv")
    st.caption("Create `.env` safely, then restart the API:")
    st.code(START_SNIPPET, language="powershell")
    st.caption("Verify with `python -m backend.check_model`.")


def show_api_error(message: str, code: str | None = None) -> None:
    st.error(message)
    if code == "model_not_configured":
        with st.expander("Model setup", expanded=True):
            model_setup_help()
    elif code == "vision_model_not_configured":
        with st.expander("Vision setup", expanded=True):
            image_setup_help()
    elif code == "image_generation_not_configured":
        with st.expander("ComfyUI setup", expanded=True):
            comfy_setup_help()
    elif code == "vision_model_unavailable":
        st.caption("This is a real vision-provider error, not a model answer.")
    elif code in {"comfyui_unavailable", "comfyui_workflow_failed"}:
        st.caption("This is a real ComfyUI error, not a fabricated image.")


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def render_connection_status() -> dict:
    status = backend_status(API)
    model = status.get("model") or {}
    st.markdown("### Connections")
    if not status["backend"]:
        st.error(status["error"])
        st.caption("Start the backend, then refresh:")
        st.code("uvicorn backend.app:app --reload --port 8000", language="powershell")
        return status

    st.success("Backend reachable")
    if status.get("model_error"):
        st.warning(status["model_error"])
        st.caption("Model connection has not been verified.")
        return status

    state = model.get("status")
    if state == "available":
        st.success(f"Model endpoint reachable: {model.get('model')}")
        st.caption("Run **Test real model response** below to verify generation.")
        if model.get("vision_configured"):
            st.caption(f"Images use {model.get('vision_model')} in Image AI.")
    elif state == "not_configured":
        st.warning("Model not configured")
        with st.expander("Model setup", expanded=True):
            model_setup_help()
    elif state == "model_missing":
        st.error(f"Ollama is reachable, but `{model.get('model')}` is not installed.")
        st.code(f"ollama pull {model.get('model')}", language="powershell")
        st.caption("Then retry this check.")
    elif state == "unreachable":
        st.error("Model endpoint unreachable")
        st.caption(model.get("detail") or "Start Ollama and verify MODEL_BASE_URL.")
        st.code("ollama serve", language="powershell")
    else:
        st.warning("Model connection error")
        st.caption(model.get("detail") or "Check the API logs and MODEL_BASE_URL.")

    if model.get("configured"):
        if st.button("Test real model response", use_container_width=True):
            with st.spinner("Sending a small real request…"):
                live_result, live_message, _ = api_request(
                    "GET",
                    "/model-status",
                    params={"live": "true"},
                    timeout=CHAT_TIMEOUT_SECONDS,
                    request_label="Model test",
                    timeout_setting="CHAT_TIMEOUT_SECONDS",
                )
                st.session_state["live_model_test"] = (
                    live_result
                    if live_result is not None
                    else {"live_ok": False, "detail": live_message}
                )
        live_result = st.session_state.get("live_model_test")
        if live_result:
            if live_result.get("live_ok"):
                st.success(
                    "Real model response received"
                    f" ({live_result.get('live_latency_seconds', '?')}s)."
                )
            else:
                st.error(live_result.get("detail") or "The live model test failed.")
    return status


def activate_workspace(name: str, *, sync_picker: bool = True) -> None:
    st.session_state["workspace"] = name
    if sync_picker:
        st.session_state["workspace_pick"] = name
    for key in (
        "workspace_file_pick",
        "workspace_context",
        "file_filter",
        "proposal",
        "workspace_chat",
        "zip_data",
        "workspace_notice",
    ):
        st.session_state.pop(key, None)


def on_workspace_pick() -> None:
    name = st.session_state.get("workspace_pick")
    if name:
        activate_workspace(name, sync_picker=False)


def render_workspace_sidebar(backend_ok: bool) -> list[dict]:
    st.markdown("### Workspaces")
    if not backend_ok:
        st.caption("Workspace controls appear when the backend is reachable.")
        return []

    payload, message, _ = api_request("GET", "/workspaces", timeout=10.0)
    if payload is None:
        st.warning(message)
        return []
    items = payload.get("workspaces", []) if isinstance(payload, dict) else []
    names = [item["name"] for item in items]

    current = st.session_state.get("workspace")
    if current not in names:
        if names:
            activate_workspace(names[0])
        else:
            st.session_state["workspace"] = None
            st.session_state.pop("workspace_pick", None)

    if names:
        if st.session_state.get("workspace_pick") not in names:
            st.session_state["workspace_pick"] = st.session_state.get("workspace")
        st.selectbox(
            "Active workspace",
            names,
            key="workspace_pick",
            on_change=on_workspace_pick,
        )
        active = st.session_state.get("workspace")
        info = next((item for item in items if item["name"] == active), None)
        if info:
            st.caption(
                f"{info['file_count']} files · {format_bytes(info['size_bytes'])}"
            )
    else:
        st.caption("No workspaces yet. Create one below.")

    with st.form("create_workspace_form", clear_on_submit=True):
        name = st.text_input("New workspace", placeholder="my-project")
        submitted = st.form_submit_button("Create workspace", use_container_width=True)
    if submitted:
        if not name.strip():
            st.error("Enter a workspace name.")
        else:
            created, created_message, created_code = api_request(
                "POST",
                "/workspaces",
                json_body={"name": name.strip()},
                timeout=15.0,
            )
            if created:
                st.session_state["_pending_workspace"] = created["name"]
                st.rerun()
            else:
                show_api_error(created_message, created_code)

    active = st.session_state.get("workspace")
    if active:
        if st.button("Prepare project ZIP", use_container_width=True, key="prepare_zip"):
            content, filename, zip_error = fetch_zip(active)
            if content is None:
                st.error(zip_error)
            else:
                st.session_state["zip_data"] = {
                    "workspace": active,
                    "content": content,
                    "filename": filename,
                }
        zip_data = st.session_state.get("zip_data")
        if zip_data and zip_data.get("workspace") == active:
            st.download_button(
                "Download project ZIP",
                data=zip_data["content"],
                file_name=zip_data["filename"],
                mime="application/zip",
                use_container_width=True,
                key="download_zip",
            )
            st.caption("Review all generated code before running it.")
    st.caption("Workspace files are local. Secrets and dependency folders are excluded.")
    return items


def render_file_panel(workspace: str, path: str | None) -> None:
    if not path:
        st.info("Select a file from the explorer, or propose a project to create files.")
        return
    content, message, code = api_request(
        "GET",
        f"/workspaces/{quote(workspace, safe='')}/file",
        params={"path": path},
        timeout=20.0,
    )
    if content is None:
        show_api_error(message, code)
        return
    language = content.get("language") or "text"
    st.markdown("#### File preview")
    st.caption(f"{path} · {language} · {format_bytes(content['size'])}")
    st.code(content["content"], language=language, line_numbers=True)

    digest = hashlib.sha1(
        f"{path}\x00{content['content']}".encode("utf-8")
    ).hexdigest()[:10]
    with st.expander("Edit this file"):
        with st.form(f"save_file_{digest}"):
            edited = st.text_area(
                "File content",
                value=content["content"],
                height=360,
                key=f"editor_text_{digest}",
            )
            if st.form_submit_button("Save file", type="primary"):
                saved, saved_message, saved_code = api_request(
                    "PUT",
                    f"/workspaces/{quote(workspace, safe='')}/file",
                    json_body={"path": path, "content": edited},
                    timeout=30.0,
                )
                if saved:
                    st.session_state["workspace_notice"] = f"Saved {path}."
                    st.rerun()
                else:
                    show_api_error(saved_message, saved_code)


def render_proposal_panel(workspace: str) -> None:
    proposal = st.session_state.get("proposal")
    if not proposal or proposal.get("workspace") != workspace:
        return
    proposal_id = proposal["proposal_id"]
    st.divider()
    st.markdown("#### Proposed changes")
    st.write(proposal.get("summary") or "No summary supplied.")
    st.caption(
        f"{proposal.get('created_count', 0)} create · "
        f"{proposal.get('updated_count', 0)} update · model {proposal.get('model', '?')}"
    )
    if proposal.get("requires_overwrite_approval"):
        st.warning("This proposal updates existing files and needs overwrite approval.")
    for warning in proposal.get("warnings", []):
        st.warning(warning)
    for file in proposal.get("files", []):
        with st.expander(f"{file['action'].title()} · {file['path']}"):
            st.code(file["content"], language=file.get("language") or "text")

    approved = st.checkbox(
        "I reviewed this proposal and approve writing these files.",
        key=f"approve_{proposal_id}",
    )
    approve_overwrites = True
    if proposal.get("requires_overwrite_approval"):
        approve_overwrites = st.checkbox(
            "I approve overwriting the existing files listed above.",
            key=f"overwrite_{proposal_id}",
        )
    ready = approved and (
        approve_overwrites or not proposal.get("requires_overwrite_approval")
    )
    apply_col, discard_col = st.columns([2, 1])
    with apply_col:
        apply_clicked = st.button(
            "Apply approved changes",
            type="primary",
            disabled=not ready,
            use_container_width=True,
            key=f"apply_{proposal_id}",
        )
    with discard_col:
        if st.button(
            "Discard proposal",
            use_container_width=True,
            key=f"discard_{proposal_id}",
        ):
            st.session_state.pop("proposal", None)
            st.rerun()
    if not ready:
        st.caption("Check the approval box before applying.")
    if apply_clicked:
        applied, message, code = api_request(
            "POST",
            f"/workspaces/{quote(workspace, safe='')}/proposals/{proposal_id}/apply",
            json_body={
                "approved": True,
                "approve_overwrites": bool(approve_overwrites),
            },
            timeout=60.0,
        )
        if applied:
            st.session_state.pop("proposal", None)
            st.session_state["workspace_notice"] = (
                f"Applied {len(applied.get('created', [])) + len(applied.get('modified', []))} "
                "file(s)."
            )
            st.rerun()
        else:
            show_api_error(message, code)


def render_workspace_chat(workspace: str) -> None:
    history = st.session_state.get("workspace_chat", [])
    st.divider()
    st.markdown("#### Work with this project")
    with st.container(height=260, border=True):
        if not history:
            st.caption(
                "Ask about selected files, or request a change. Changes are previewed "
                "and never written without approval."
            )
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("details"):
                    st.caption(message["details"])
                for warning in message.get("warnings", []):
                    st.caption(f"Note: {warning}")
    action = st.radio(
        "Action",
        WORKSPACE_ACTIONS,
        horizontal=True,
        key="workspace_action",
        label_visibility="collapsed",
    )
    prompt = st.chat_input(
        "Describe a change or ask about the selected project files…",
        key="workspace_prompt",
    )
    if not prompt:
        return
    context_files = list(st.session_state.get("workspace_context", []))
    history_payload = [
        {"role": message["role"], "content": message["content"]}
        for message in history[-20:]
    ]
    if action == WORKSPACE_ACTIONS[0]:
        with st.spinner("Generating a project proposal…"):
            proposal, message, code = api_request(
                "POST",
                f"/workspaces/{quote(workspace, safe='')}/proposals",
                json_body={
                    "message": prompt,
                    "context_files": context_files,
                    "history": history_payload,
                },
                timeout=240.0,
            )
        if proposal:
            proposal["workspace"] = workspace
            st.session_state["proposal"] = proposal
            st.rerun()
        else:
            show_api_error(message, code)
        return

    with st.spinner("Reading the selected project files…"):
        answer, message, code = api_request(
            "POST",
            f"/workspaces/{quote(workspace, safe='')}/ask",
            json_body={
                "message": prompt,
                "context_files": context_files,
                "history": history_payload,
            },
            timeout=180.0,
        )
    if answer:
        used = answer.get("context_files", [])
        details = f"Model: {answer.get('model', '?')}"
        if used:
            details += f" · Context: {', '.join(used)}"
        st.session_state["workspace_chat"] = history + [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": answer["answer"],
                "details": details,
                "warnings": answer.get("warnings", []),
            },
        ]
        st.rerun()
    else:
        show_api_error(message, code)


def render_workspace_tab(backend_ok: bool) -> None:
    if not backend_ok:
        st.info("Start the backend to use the coding workspace.")
        st.code("uvicorn backend.app:app --reload --port 8000", language="powershell")
        return
    workspace = st.session_state.get("workspace")
    if not workspace:
        st.info("Create a workspace in the sidebar to start building.")
        return

    notice = st.session_state.pop("workspace_notice", None)
    if notice:
        st.success(notice)

    files, message, code = api_request(
        "GET",
        f"/workspaces/{quote(workspace, safe='')}/files",
        timeout=20.0,
    )
    if files is None:
        show_api_error(message, code)
        return
    total_size = sum(file["size"] for file in files)
    title_col, refresh_col = st.columns([5, 1])
    with title_col:
        st.subheader(workspace)
        st.caption(f"{len(files)} files · {format_bytes(total_size)} · local workspace")
    with refresh_col:
        if st.button("Refresh", use_container_width=True, key="refresh_workspace"):
            st.rerun()

    explorer, editor = st.columns([0.9, 2.1], gap="large")
    with explorer:
        filter_text = st.text_input(
            "Filter files",
            placeholder="src/…",
            key="file_filter",
        ).strip().lower()
        filtered = [
            file for file in files if filter_text in file["path"].lower()
        ]
        all_paths = [file["path"] for file in files]
        filtered_paths = [file["path"] for file in filtered]
        selected_path = None
        if filtered_paths:
            if st.session_state.get("workspace_file_pick") not in filtered_paths:
                st.session_state["workspace_file_pick"] = filtered_paths[0]
            selected_path = st.radio(
                "Project files",
                filtered_paths,
                key="workspace_file_pick",
                label_visibility="collapsed",
            )
            st.caption(f"{len(filtered_paths)} of {len(all_paths)} files")
        else:
            st.caption("No matching files." if files else "This workspace is empty.")
        st.markdown("**Context for chat**")
        context_state = [
            path
            for path in st.session_state.get("workspace_context", [])
            if path in all_paths
        ]
        if context_state != st.session_state.get("workspace_context", []):
            st.session_state["workspace_context"] = context_state
        st.multiselect(
            "Include files",
            all_paths,
            key="workspace_context",
            label_visibility="collapsed",
            placeholder="Choose files to send as context",
        )
        st.caption("Context is read-only. Secrets and dependency folders are excluded.")
    with editor:
        render_file_panel(workspace, selected_path)

    render_proposal_panel(workspace)
    render_workspace_chat(workspace)


def render_image_tab(backend_ok: bool, model_info: dict) -> None:
    st.subheader("Image AI")
    st.caption(
        "Analyze images with a local vision model. Image generation is optional and "
        "runs only through a configured local ComfyUI server."
    )
    if not backend_ok:
        st.info("Start the backend to use image tools.")
        st.code("uvicorn backend.app:app --reload --port 8000", language="powershell")
        return

    status_col, refresh_col = st.columns([5, 1])
    with refresh_col:
        if st.button("Refresh", use_container_width=True, key="refresh_image_status"):
            image_capabilities.clear()
            st.rerun()

    capabilities, status_error = image_capabilities(API)
    if capabilities is None:
        st.error(status_error)
        return
    limits = capabilities.get("limits") or {}
    max_upload_bytes = int(limits.get("max_upload_bytes") or 5_242_880)
    vision = capabilities.get("vision") or {}
    vision_status = vision.get("status")
    vision_model = vision.get("model") or model_info.get("vision_model") or "vision model"
    vision_ready = vision_status == "available"

    with status_col:
        if vision_ready:
            st.success(f"Vision ready: {vision_model}")
            st.caption("Uses Ollama's native `/api/chat` image payload.")
        elif vision_status == "model_missing":
            st.error(vision.get("detail") or f"{vision_model} is not installed.")
            st.code(f"ollama pull {vision_model}", language="powershell")
        elif vision_status == "not_configured":
            st.warning("Vision is not configured.")
            with st.expander("Vision setup", expanded=True):
                image_setup_help()
        elif vision_status == "unreachable":
            st.error(vision.get("detail") or "Ollama is unreachable.")
        else:
            st.warning(vision.get("detail") or "Vision status is unavailable.")

    st.markdown("#### Analyze an image")
    upload = st.file_uploader(
        "Image",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_ai_upload",
        help="PNG, JPEG, or WebP. The backend validates the file signature and dimensions.",
    )
    raw = upload.getvalue() if upload else b""
    too_large = bool(raw and len(raw) > max_upload_bytes)
    if too_large:
        st.error(f"Image exceeds the {max_upload_bytes / (1024 * 1024):g} MB limit.")
    if upload:
        st.image(upload, caption=upload.name, width=520)

    mode_label = st.selectbox(
        "Task",
        list(IMAGE_MODES),
        key="image_ai_mode",
    )
    mode = IMAGE_MODES[mode_label]
    prompt = st.text_area(
        "Question or instruction (optional)",
        placeholder=IMAGE_MODE_PROMPTS[mode],
        height=90,
        key="image_ai_prompt",
    ).strip()
    analyze_clicked = st.button(
        "Analyze image",
        type="primary",
        disabled=not upload or too_large or not vision_ready,
        use_container_width=True,
        key="analyze_image",
    )
    if analyze_clicked and upload and not too_large:
        data_url = (
            f"data:{upload.type or 'image/png'};base64,"
            f"{base64.b64encode(raw).decode('ascii')}"
        )
        with st.spinner(f"Reading the image with {vision_model}…"):
            result, message, code = api_request(
                "POST",
                "/images/analyze",
                json_body={
                    "image_data_url": data_url,
                    "prompt": prompt or IMAGE_MODE_PROMPTS[mode],
                    "mode": mode,
                },
                timeout=IMAGE_ANALYZE_TIMEOUT_SECONDS,
                request_label="Image analysis",
                timeout_setting="IMAGE_ANALYZE_TIMEOUT_SECONDS",
            )
        if result is None:
            show_api_error(message, code)
        else:
            st.session_state["image_analysis"] = {
                "digest": hashlib.sha1(raw).hexdigest(),
                "answer": result["answer"],
                "model": result["model"],
                "mode": result["mode"],
                "image": result["image"],
            }
            st.rerun()

    if upload:
        digest = hashlib.sha1(raw).hexdigest()
        analysis = st.session_state.get("image_analysis")
        if analysis and analysis.get("digest") == digest:
            image_meta = analysis.get("image") or {}
            st.markdown("#### Analysis")
            st.markdown(analysis["answer"])
            st.caption(
                f"Model: {analysis['model']} · Task: {analysis['mode']} · "
                f"{image_meta.get('width', '?')}×{image_meta.get('height', '?')} · "
                f"{format_bytes(int(image_meta.get('size_bytes', 0)))}"
            )

    st.divider()
    st.markdown("#### Generate an image")
    generation = capabilities.get("generation") or {}
    generation_enabled = bool(generation.get("enabled"))
    generation_status = generation.get("status")

    if not generation_enabled:
        st.info(
            "Image generation is disabled. It is optional and requires a local "
            "ComfyUI installation."
        )
        with st.expander("ComfyUI setup"):
            comfy_setup_help()
        return
    if generation.get("status") == "not_configured":
        st.warning(generation.get("detail") or "ComfyUI is not configured.")
        with st.expander("ComfyUI setup", expanded=True):
            comfy_setup_help()
        return
    if generation_status == "checkpoint_missing":
        st.error(generation.get("detail") or "The configured checkpoint is missing.")
        with st.expander("ComfyUI setup", expanded=True):
            comfy_setup_help()
        return
    if generation_status == "unreachable":
        st.error(generation.get("detail") or "ComfyUI is unreachable.")
        st.caption("Start ComfyUI, then refresh the image status.")
        with st.expander("ComfyUI setup"):
            comfy_setup_help()
        return
    if generation_status != "available":
        st.warning(generation.get("detail") or "ComfyUI status is unavailable.")
        return

    st.success(f"ComfyUI reachable · checkpoint {generation.get('checkpoint')}")
    st.caption("Generation has not been tested until you submit a prompt.")
    with st.form("image_generation_form"):
        image_prompt = st.text_area(
            "Image prompt",
            placeholder="A cinematic mountain lake at sunrise, detailed, natural light",
            height=100,
        )
        negative_prompt = st.text_input(
            "Negative prompt",
            placeholder="blurry, low quality, watermark, text",
        )
        width_col, height_col, steps_col, cfg_col = st.columns(4)
        with width_col:
            width = st.selectbox(
                "Width",
                [256, 384, 512, 640, 768, 896, 1024],
                index=2,
            )
        with height_col:
            height = st.selectbox(
                "Height",
                [256, 384, 512, 640, 768, 896, 1024],
                index=2,
            )
        with steps_col:
            steps = st.slider("Steps", 1, 60, 20)
        with cfg_col:
            cfg = st.slider("CFG", 0.0, 30.0, 7.0, 0.5)
        seed = st.number_input(
            "Seed (-1 = random)",
            min_value=-1,
            max_value=4_294_967_295,
            value=-1,
            step=1,
        )
        generate_clicked = st.form_submit_button(
            "Generate image",
            type="primary",
            use_container_width=True,
        )
    if generate_clicked:
        if not image_prompt.strip():
            st.error("Enter an image prompt.")
        else:
            with st.spinner("ComfyUI is generating the image. This can take a while…"):
                generated, message, code = api_request(
                    "POST",
                    "/images/generate",
                    json_body={
                        "prompt": image_prompt.strip(),
                        "negative_prompt": negative_prompt.strip(),
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "cfg": cfg,
                        "seed": None if seed < 0 else int(seed),
                    },
                    timeout=IMAGE_GENERATION_TIMEOUT_SECONDS,
                    request_label="Image generation",
                    timeout_setting="IMAGE_GENERATION_TIMEOUT_SECONDS",
                )
            if generated is None:
                show_api_error(message, code)
            else:
                st.session_state["generated_image"] = generated
                st.rerun()

    generated = st.session_state.get("generated_image")
    if generated:
        content, image_error = fetch_generated_image(API, generated["image_id"])
        if content is None:
            st.error(image_error)
        else:
            st.image(content, caption=generated.get("prompt"), use_container_width=True)
            st.download_button(
                "Download generated image",
                data=content,
                file_name=generated["image_id"],
                mime=generated.get("media_type") or "image/png",
                use_container_width=True,
                key="download_generated_image",
            )
            st.caption(
                f"{generated.get('width')}×{generated.get('height')} · "
                f"seed {generated.get('seed')} · "
                f"{format_bytes(int(generated.get('size_bytes', 0)))} · "
                f"{generated.get('checkpoint')}"
            )
            st.caption("Generated locally. Review before sharing.")


def render_general_tab(backend_ok: bool, model_info: dict) -> None:
    st.subheader("General chat")
    st.caption(
        "Coding uses the configured coding model. Images are routed to the "
        "dedicated local vision model."
    )
    controls, new_col = st.columns([4, 1])
    with controls:
        skill = st.selectbox(
            "Working mode",
            [
                "auto",
                "coding",
                "reasoning",
                "research",
                "teacher",
                "design",
                "vision",
                "empathetic",
            ],
            key="general_skill",
        )
    with new_col:
        st.write("")
        if st.button("New conversation", use_container_width=True, key="new_general"):
            st.session_state["general_conversation_id"] = str(uuid.uuid4())
            st.session_state["general_messages"] = []
            st.session_state["general_history"] = []
            st.session_state.pop("live_model_test", None)
            st.rerun()
    st.caption("Modes guide the same model; they are not separately trained models.")

    messages = st.session_state.get("general_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("model"):
                st.caption(f"Model: {message['model']}")

    upload = st.file_uploader(
        "Optional image",
        type=["png", "jpg", "jpeg", "webp"],
        key="general_upload",
    )
    image_too_large = bool(upload and len(upload.getvalue()) > 5 * 1024 * 1024)
    capabilities, _ = image_capabilities(API) if upload else (None, "")
    vision = (capabilities or {}).get("vision") or {}
    vision_status = vision.get("status")
    vision_blocked = bool(
        upload
        and vision_status
        in {"not_configured", "model_missing", "unreachable", "error"}
    )
    vision_model = (
        vision.get("model")
        or model_info.get("vision_model")
        or "the local vision model"
    )
    if image_too_large:
        st.warning("Image exceeds the 5 MB general-chat limit.")
    elif vision_status == "model_missing":
        st.error(vision.get("detail") or f"{vision_model} is not installed.")
        st.code(f"ollama pull {vision_model}", language="powershell")
    elif vision_status == "not_configured":
        st.warning("Vision is not configured.")
        with st.expander("Vision setup"):
            image_setup_help()
    elif vision_status in {"unreachable", "error"}:
        st.error(vision.get("detail") or "The vision model is unavailable.")
    elif upload:
        st.caption(f"This image will be sent to {vision_model}.")
    prompt = st.chat_input(
        "Ask Vups to build, explain, debug, analyze, or design…",
        key="general_prompt",
        disabled=not backend_ok or image_too_large or vision_blocked,
    )
    if not prompt:
        return

    image_url = None
    if upload:
        raw = upload.getvalue()
        if len(raw) > 5 * 1024 * 1024:
            st.error("Image exceeds 5 MB.")
            return
        image_url = (
            f"data:{upload.type or 'image/png'};base64,"
            f"{base64.b64encode(raw).decode('ascii')}"
        )

    with st.chat_message("user"):
        st.markdown(prompt)
        if upload:
            st.image(upload, caption=upload.name, use_container_width=True)
    with st.chat_message("assistant"):
        with st.spinner("Vups Gen 4 is thinking…"):
            data, message, code = api_request(
                "POST",
                "/chat",
                json_body={
                    "message": prompt,
                    "conversation_id": st.session_state["general_conversation_id"],
                    "history": st.session_state.get("general_history", [])[-20:],
                    "skill": skill,
                    "image_data_url": image_url,
                },
                timeout=CHAT_TIMEOUT_SECONDS,
                request_label="General chat",
                timeout_setting="CHAT_TIMEOUT_SECONDS",
            )
        if data is None:
            show_api_error(message, code)
            return
        answer = str(data.get("answer", "")).strip()
        if not answer:
            st.error("The AI backend returned an empty response.")
            return
        st.markdown(answer)
        st.caption(f"Mode: {data.get('skill')} · Model: {data.get('model')}")

    st.session_state["general_messages"].extend(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": answer,
                "model": data.get("model"),
            },
        ]
    )
    st.session_state["general_history"].extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )
    st.rerun()


st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1100px 620px at 18% -12%, #1b2b50 0%, #080d19 58%) fixed;
    }
    [data-testid="stSidebar"] {
        background: #0b1222;
        border-right: 1px solid #1d2a45;
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    .vups-hero {
        border: 1px solid #223252;
        border-radius: 22px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.1rem;
        background:
            linear-gradient(120deg, rgba(52, 86, 160, .22), rgba(15, 24, 44, .55)),
            #0d162a;
        box-shadow: 0 20px 60px rgba(0, 0, 0, .22);
    }
    .vups-kicker {
        color: #91b0ff;
        font-size: .72rem;
        font-weight: 760;
        letter-spacing: .18em;
        text-transform: uppercase;
    }
    .vups-hero h1 {
        margin: .35rem 0 .15rem;
        color: #f2f6ff;
        font-size: clamp(1.8rem, 3.4vw, 2.8rem);
        letter-spacing: -.045em;
        line-height: 1.05;
    }
    .vups-hero p {
        margin: 0;
        color: #a9b8d8;
        max-width: 850px;
    }
    .vups-hint {
        color: #8fa1c2;
        font-size: .86rem;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #223252;
        border-radius: 16px;
        padding: .75rem;
        background: #0f1830;
    }
    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: .45rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: .45rem 1rem;
    }
    .stButton button[kind="primary"] {
        background: #4f74e8;
        border-color: #6d8dff;
    }
    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; animation: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("general_conversation_id", str(uuid.uuid4()))
st.session_state.setdefault("general_messages", [])
st.session_state.setdefault("general_history", [])
st.session_state.setdefault("workspace_chat", [])
st.session_state.setdefault("workspace", None)

pending_workspace = st.session_state.pop("_pending_workspace", None)
if pending_workspace:
    activate_workspace(pending_workspace)

with st.sidebar:
    st.markdown("## ✦ Vups Gen 4")
    st.caption("Local AI coding and image studio.")
    connection = render_connection_status()
    st.divider()
    render_workspace_sidebar(bool(connection.get("backend")))
    st.divider()
    st.caption(
        "Do not upload secrets. Generated code is never executed automatically."
    )

st.markdown(
    """
    <div class="vups-hero">
      <div class="vups-kicker">Vups Gen 4 · Local AI studio</div>
      <h1>Build, inspect, and create without leaving your machine.</h1>
      <p>Review every code change, ask questions with full-file context, analyze
      images with a local vision model, and optionally generate images with ComfyUI.
      Nothing is executed or published automatically.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_info = (connection.get("model") or {}) if connection else {}
backend_ok = bool(connection.get("backend"))
workspace_tab, image_tab, general_tab = st.tabs(
    ["Workspace", "Image AI", "General chat"]
)
with workspace_tab:
    render_workspace_tab(backend_ok)
with image_tab:
    render_image_tab(backend_ok, model_info)
with general_tab:
    render_general_tab(backend_ok, model_info)
