# Vups Gen 4 — safe local AI coding workspace

A working local application for one AI assistant with coding, reasoning, teaching, research, design, image-analysis, image generation, and empathetic-conversation modes. Vups includes a real coding workspace: propose multiple files, inspect every change, approve the write, edit files, export a ZIP, and ask questions with selected-file context. A dedicated Image AI tab analyzes uploaded images with a local vision model and can generate images through an optional local ComfyUI server.

## Reality check
This is a **platform and local coding workspace**, not a new frontier model trained from scratch. It calls an OpenAI-compatible model server when configured. Without one, the API returns an explicit `503` setup error and never fabricates a model answer. Generated code is **never executed, installed, or deployed** by Vups. No model can guarantee “knowing everything,” perfect code, or human feelings. The UI modes are prompt guidance, not separate trained skills.

## Architecture
- Streamlit UI (`frontend/`) with Workspace, Image AI, and General chat tabs
- FastAPI orchestration API (`backend/`) with workspace, chat, and image endpoints
- Safe local workspace service: path validation, secret filtering, atomic writes, approval-gated proposals, ZIP export
- Text-model adapter for OpenAI-compatible endpoints; separate native Ollama adapter for vision
- Optional ComfyUI text-to-image adapter with validated local output storage
- Turso/libSQL persistence with SQLite fallback for general chat
- Docker Compose and Kubernetes manifests

Workspace files live under `workspaces/` by default. Generated images live under `generated_images/` by default. Proposals are held in API memory for 30 minutes and are **not written until explicitly approved**. Overwriting existing files requires separate overwrite approval.

## $0 approach and constraints
You can develop for $0 if you already own the computer and use local open-weight models. Hosted Kubernetes nodes generally cost money; use `kind` or `minikube` locally. Streamlit Community Cloud is not Kubernetes. Turso plan limits can change; verify current pricing/limits. Model weights belong on model-serving storage, not Turso. Local CPU inference can be slow; training a competitive LLM from scratch is not realistic on ordinary consumer hardware.

## Local setup
Python 3.11+ recommended. Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```
The API loads `.env` from the project root regardless of the terminal's current directory. Existing operating-system environment variables take precedence. Never commit `.env`.

## Local Ollama setup (Windows)
This section is for running Ollama and Vups directly on Windows. Docker/Kubernetes use different networking; see [Docker Compose](#docker-compose).

1. Install Ollama, then confirm the command is available:
```powershell
ollama --version
```
2. Start Ollama (the desktop app, or `ollama serve` in its own terminal) and pull the coding and vision models:
```powershell
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5vl:7b
ollama list
```
3. Confirm `.env` contains:
```dotenv
MODEL_BASE_URL=http://127.0.0.1:11434/v1
MODEL_NAME=qwen2.5-coder:7b
MODEL_API_KEY=ollama
MODEL_TIMEOUT_SECONDS=1000
MODEL_SUPPORTS_VISION=auto
VISION_MODEL_NAME=qwen2.5vl:7b
VISION_BASE_URL=http://127.0.0.1:11434
VISION_TIMEOUT_SECONDS=900
CHAT_TIMEOUT_SECONDS=1100
```
`MODEL_SUPPORTS_VISION` applies only to the primary text model and defaults to `auto`. Images are not sent to the coding model: `/chat` and `/images/analyze` use the separate native Ollama vision endpoint. `VISION_BASE_URL` may be omitted; it then falls back to `MODEL_BASE_URL`. `CHAT_TIMEOUT_SECONDS` is the Streamlit request deadline and should be longer than both `MODEL_TIMEOUT_SECONDS` and `VISION_TIMEOUT_SECONDS`; when omitted it defaults to the larger value plus 30 seconds.

4. Check the real connection before starting the UI:
```powershell
python -m backend.check_model
```
The command checks the primary model, Ollama's native model list, the vision model installation, and a tiny real chat completion. It prints no secret values. The vision check verifies installation only; use the Image AI tab or a live smoke test to verify an actual image response.
5. Start the API and UI in separate terminals:
```powershell
uvicorn backend.app:app --reload --port 8000
```
```powershell
streamlit run frontend/app.py
```
Open `http://127.0.0.1:8501`. The sidebar shows backend, model, and Ollama status, plus a **Test real model response** action.

`qwen2.5-coder:7b` is text-only. The Image AI tab and image uploads in General chat use `qwen2.5vl:7b` instead, with the image sent as base64 bytes through Ollama's native `/api/chat` endpoint.

## Coding workspace

Start the API and UI as above, then open the **Workspace** tab:

1. Create a workspace in the sidebar. Names are limited to letters, digits, dots, underscores, and hyphens.
2. Select files in the explorer and choose any files to send as read-only context.
3. Use **Propose file changes** to ask for a multi-file project or edit. The model must return complete JSON file contents; one bounded repair attempt is made if the response is malformed.
4. Review the proposal, its summary, warnings, and every generated file. Nothing is written before approval. Updating an existing file requires a second overwrite approval.
5. Apply the proposal, edit individual files directly in the file panel, or discard the proposal.
6. Ask project questions with **Ask about the project**, using the selected file context.
7. Use **Prepare project ZIP** then **Download project ZIP** to export. ZIPs exclude secrets and dependency artifacts and include a review warning in the archive comment.

Limits are configurable in `.env`:

| Setting | Default |
|---|---|
| `WORKSPACE_MAX_FILES` | 24 files per proposal |
| `WORKSPACE_MAX_FILE_BYTES` | 65,536 bytes per file |
| `WORKSPACE_MAX_TOTAL_BYTES` | 524,288 bytes per workspace |
| `WORKSPACE_MAX_CONTEXT_FILES` | 8 files |
| `WORKSPACE_MAX_CONTEXT_BYTES` | 48,000 bytes |
| `WORKSPACE_MAX_READ_BYTES` | 262,144 bytes |
| `WORKSPACE_MODEL_MAX_TOKENS` | 4,096 tokens |

`WORKSPACE_ROOT` defaults to `workspaces/` under the project root. Relative paths are resolved against the project root.

Safety behavior:
- Absolute paths, drive letters, `..`, backslashes, null bytes, home paths, symlinks, junctions, and reparse-point escapes are rejected.
- `.env*`, credentials, private keys, databases, logs, `.git`, `.venv`, `node_modules`, caches, and build artifacts are excluded from listings, model context, editing, and ZIP export.
- Writes use temporary files, `fsync`, and atomic replacement. A failed write must not leave a partial file.
- Pending proposals are in memory and lost when the API restarts. Proposal approvals are explicit; direct file-editor saves are also explicit user actions.
- Generated code is untrusted. Review it before running it yourself. The API does not execute, install, or deploy anything.

Do not expose Ollama (`11434`), the API (`8000`), or Streamlit (`8501`) to the public internet. This development setup has no authentication or rate limiting.

## Image AI

### Understanding images
The Image AI tab and image uploads in General chat use a dedicated vision model. Configure:

```dotenv
VISION_MODEL_NAME=qwen2.5vl:7b
VISION_BASE_URL=http://127.0.0.1:11434
VISION_TIMEOUT_SECONDS=900
VISION_MODEL_MAX_TOKENS=2048
IMAGE_MAX_UPLOAD_BYTES=5242880
IMAGE_MAX_PIXELS=20000000
IMAGE_MAX_DIMENSION=8192
```

- Only PNG, JPEG, and WebP are accepted. The backend checks the base64 data URL, file signature, actual Pillow-decoded format, byte limit, dimensions, and pixel count. A declared MIME type must match the image contents.
- Tasks are `describe`, `ocr`, `ui_review`, and `code`. The answer distinguishes observations from inference and treats text inside an image as untrusted data.
- The vision request uses Ollama's native `/api/chat` payload with raw base64 image bytes in `messages[].images`. It does not send filenames, local paths, or the OpenAI-compatible text request.
- `VISION_TIMEOUT_SECONDS` is a total inference deadline, not just a network read timeout. The default is 900 seconds because CPU-only vision encoding and prompt processing for a large image can take several minutes before the first token. Streamlit waits an additional 30 seconds beyond that limit.
- Missing configuration returns `503`; an unreachable endpoint or model failure returns `502`; invalid images return `400`, `413`, `415`, or `422`. The API never substitutes a fake answer.
- `qwen2.5vl:7b` performs image understanding only. It does not generate images.

### Optional ComfyUI generation
Image generation is disabled by default. To enable it, install and run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) locally, install a checkpoint under its `models/checkpoints` directory, then set:

```dotenv
IMAGE_GENERATION_PROVIDER=comfyui
COMFYUI_BASE_URL=http://127.0.0.1:8188
COMFYUI_CHECKPOINT=v1-5-pruned-emaonly.safetensors
COMFYUI_TIMEOUT_SECONDS=300
IMAGE_OUTPUT_DIR=generated_images
IMAGE_GENERATION_MAX_OUTPUT_BYTES=20000000
```

`COMFYUI_CHECKPOINT` must exactly match the checkpoint filename shown by ComfyUI. Vups uses a standard text-to-image workflow with `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, and `SaveImage`.

The API probes ComfyUI's `/system_stats` and `/object_info/CheckpointLoaderSimple`, queues the workflow through `/prompt`, polls `/history/{prompt_id}`, downloads the resulting image through `/view`, validates it, and stores it with a random safe filename under `IMAGE_OUTPUT_DIR`. Status values distinguish disabled, not configured, unreachable, missing checkpoint, and available. A ComfyUI error never produces a placeholder image. The configured output directory is not exposed as an arbitrary filesystem path; downloads use generated IDs only.

Generated images are not added to coding workspaces and are not cleaned up automatically. Add your own retention policy before using this beyond local development. In containers, mount a persistent writable volume at `IMAGE_OUTPUT_DIR` if images must survive replacement.

## Turso
Create a database and token using Turso's current docs/dashboard. Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`. Schema initializes on startup. Never commit `.env`; use Kubernetes Secrets for deployment.

## Tests
```bash
pytest -q
```
Tests use temporary workspace and image roots plus SQLite. They mock model and ComfyUI calls and do not require Turso, Ollama, ComfyUI, model weights, or a GPU. The suite covers workspace path traversal, secret filtering, symlink/junction escapes, limits, approval behavior, atomic writes, ZIP safety, invalid model output, repair handling, image signature/dimension/MIME validation, native Ollama vision payloads, ComfyUI workflow and output validation, safe image IDs, and API responses.

## Docker Compose
```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up --build
```
UI: `http://localhost:8501`; API health: `http://localhost:8000/health`.

Docker networking is not the same as the Windows host:
- Inside a container, `127.0.0.1`/`localhost` points at that container, not at Ollama on Windows. Set `MODEL_BASE_URL=http://host.docker.internal:11434/v1` where supported, or run an Ollama container on the same Docker network.
- Do not publish Ollama or the API directly to the public internet. The current setup has no authentication.
- `docker-compose.yml` mounts named volumes at `/app/workspaces` and `/app/generated_images`. If you run the API image directly, mount volumes for `WORKSPACE_ROOT` and `IMAGE_OUTPUT_DIR`; otherwise those files are lost when the container is replaced.
- A ComfyUI server must be reachable from the API container. `127.0.0.1` inside the container is not the host; use a reachable address or run ComfyUI on the same Docker network. Ollama has the same networking consideration.

## Local Kubernetes
Install Docker, `kubectl`, and `kind` or `minikube`.
```bash
docker build -f backend/Dockerfile -t vups-gen4-api:dev .
docker build -f frontend/Dockerfile -t vups-gen4-ui:dev .
kind load docker-image vups-gen4-api:dev
kind load docker-image vups-gen4-ui:dev
kubectl apply -f deploy/kubernetes/
kubectl -n vups-gen4 port-forward svc/vups-ui 8501:8501
```
Before deploying, create a secret in the namespace:
```bash
kubectl create secret generic vups-secrets -n vups-gen4 \
  --from-literal=TURSO_DATABASE_URL='libsql://YOUR_DB-YOUR_ORG.turso.io' \
  --from-literal=TURSO_AUTH_TOKEN='YOUR_TOKEN' \
  --from-literal=MODEL_API_KEY='ollama'
```
Edit `deploy/kubernetes/configmap.yaml` with reachable Ollama and ComfyUI URLs and model names. Pods cannot normally reach a host's `127.0.0.1`; configure endpoints reachable from the cluster or serve the models in-cluster. For a remote cluster, push images to a registry and update image names. The API manifest mounts `emptyDir` volumes at `/app/workspaces` and `/app/generated_images`, which are lost when the pod is replaced; add PersistentVolumeClaims for durable files. Keep one API replica unless proposal storage is made durable and shared.

## API
- `GET /health` — process health only
- `GET /ready` — application/database readiness plus whether a model is configured
- `GET /model-info` — safe model configuration details (never the API key)
- `GET /model-status` — live text endpoint/model probe; add `?live=true` for a tiny real completion; also reports the configured vision-model name
- `GET /image-status` — Ollama vision and optional ComfyUI capability status plus image limits
- `POST /images/analyze` — validate an image data URL and analyze it with the vision model
- `POST /images/generate` — generate an image through configured ComfyUI
- `GET /images/generated/{image_id}` — retrieve a generated image by safe generated ID
- `POST /chat` — `503` when unconfigured, `502` on real provider failure; image requests are routed to the vision model; failures never produce a fake answer
- `GET /conversations/{conversation_id}`
- `GET /workspaces` — list workspaces
- `POST /workspaces` — create a workspace
- `GET /workspaces/{name}` — workspace metadata
- `GET /workspaces/{name}/files` — list safe files
- `GET /workspaces/{name}/file?path=...` — read a text file
- `PUT /workspaces/{name}/file` — explicitly save a text file
- `POST /workspaces/{name}/proposals` — generate a previewable proposal
- `POST /workspaces/{name}/proposals/{id}/apply` — apply after approval
- `POST /workspaces/{name}/ask` — ask about selected context files
- `GET /workspaces/{name}/export` — download a safe ZIP

Workspace error responses use `400` invalid path/name, `403` sensitive path, `404` missing workspace/file/proposal, `409` approval or name conflict, `410` expired proposal, `413` size limit, `415` binary file, `422` invalid proposal/limit, and `502` invalid model output or provider failure. Apply requests require `{"approved": true}`; overwrites additionally require `{"approve_overwrites": true}`.

Image error responses use `400` malformed image data, `413` upload size limit, `415` unsupported type or signature mismatch, `422` unsafe dimensions, `503` unconfigured vision/generation service, `502` real vision or ComfyUI failure, and `404` unknown generated image ID. Error responses use `{"detail": {"code": "...", "message": "..."}}`.

Run `python -m backend.check_model` for a human-readable diagnostic of the same configuration and connections.

## Production gaps
Before public launch add authentication, per-user workspace and generated-image ownership, authorization checks, rate limiting, request size limits, TLS/Ingress, secret rotation, backups, retention/deletion controls, observability, and abuse protection. The current workspace and generated-image roots are shared by all API users and pending proposals are process-local; a multi-user deployment needs per-user storage and a durable proposal store. Never execute generated code in the API pod; use a separate ephemeral sandbox with strict CPU, memory, filesystem and network limits. Treat ComfyUI and Ollama as sensitive local services and never expose them publicly.

## Developing your own model
Start with a tiny tokenizer/transformer experiment on a licensed dataset to learn the training pipeline; keep train/validation splits and track benchmarks. Fine-tune open-weight models only after checking the license. A toy model is educational, not competitive with current general-purpose coding models.
