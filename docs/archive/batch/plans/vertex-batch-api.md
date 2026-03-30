# Vertex AI Batch API Support

## Background

pdf2epub currently supports Gemini batch API via `GeminiBatchClient` in `batch_utils.py`.
This plan adds Vertex AI batch API support using the **same `google-genai` SDK** — the SDK
abstracts both backends behind `client.batches.create()`.

## Why Vertex Batch

- Vertex batch API is a completely different system from Gemini batch API at the REST level
  (different endpoints, auth, storage model), BUT the `google-genai` SDK unifies them.
- With `genai.Client(vertexai=True, ...)`, `client.batches.create()` works for Vertex.
- Vertex batch uses GCS for input/output (not Gemini File API).
- Vertex batch has separate quota from Gemini API — useful when Gemini quota is exhausted.

## GCP Environment (Already Configured)

- **Project**: `project-20ca3d19-8a71-475f-89b` ("My First Project")
- **ADC credentials**: `~/.config/gcloud/application_default_credentials.json` (authorized_user, has refresh_token)
- **Quota project**: set to `project-20ca3d19-8a71-475f-89b`
- **Enabled APIs** (all confirmed):
  - `aiplatform.googleapis.com` (Vertex AI)
  - `storage.googleapis.com` (GCS)
  - `bigquery.googleapis.com` (optional)
- **gcloud CLI**: no active auth (all stale accounts revoked), but ADC works independently
- **No GCS bucket exists yet** — code should auto-create one

## API Differences (SDK Level)

| | Gemini Batch (current) | Vertex Batch (to add) |
|---|---|---|
| Client init | `genai.Client(api_key=...)` | `genai.Client(vertexai=True, project=..., location=...)` |
| Input source (`src`) | File API upload name (`files/abc123`) | GCS URI (`gs://bucket/input.jsonl`) |
| Output | `job.dest.inlined_responses` or `job.dest.file_name` (File API) | GCS URI (need to download from GCS) |
| Model format | `models/gemini-2.5-pro` | `gemini-2.5-pro` (SDK adds prefix) |
| Auth | API key | ADC (Application Default Credentials) |
| `batches.get()` / `batches.cancel()` | Same interface | Same interface |
| JSONL line format | `{"key": "...", "request": {...}}` | `{"request": {...}}` (verify if `key` is supported) |

## Implementation Plan

### 1. Create `VertexBatchClient` in `batch_utils.py`

Same interface as `GeminiBatchClient`. Key differences:

#### `__init__`
```python
def __init__(
    self,
    project: str,
    location: str = "us-central1",
    model: str = "gemini-2.5-flash",
    poll_interval: int = 60,
    bucket_name: Optional[str] = None,  # Auto-generated if None
):
```
- No `api_key` — uses ADC
- Needs `project` and `location`
- `bucket_name` for GCS input/output (auto-create if not provided)

#### `_get_client()`
```python
from google import genai
from google.genai import types
self._client = genai.Client(
    vertexai=True,
    project=self.project,
    location=self.location,
    http_options=types.HttpOptions(api_version="v1"),
)
```

#### `_ensure_bucket()`
- Auto-create GCS bucket `pdf2epub-batch-{project_id[:8]}` if not exists
- Use `google.cloud.storage` or raw REST API
- Bucket location should match Vertex location

#### `submit()`
1. Write JSONL to temp file (same as current)
2. Upload temp file to GCS bucket (`gs://bucket/batch-inputs/{timestamp}.jsonl`)
3. `client.batches.create(model=self.model, src=gcs_uri)`
4. Return job name

#### `get_results()`
1. `client.batches.get(name=job_name)`
2. Get output GCS URI from job response
3. Download JSONL from GCS
4. Parse into `List[BatchResponse]` (same format)

#### `_cleanup_gcs(job_name)`
- Delete input/output JSONL files from GCS after results are retrieved
- Called automatically after `get_results()`

### 2. Update `factory_v2.py::_create_batch_client_from_config()`

```python
if batch_entry.provider == "vertex":
    provider_config = credentials.get("vertex", {})
    return VertexBatchClient(
        project=provider_config.get("project"),
        location=provider_config.get("location", "us-central1"),
        model=batch_entry.model,
        poll_interval=poll_interval,
        bucket_name=provider_config.get("bucket"),  # Optional
    )
elif batch_entry.provider == "gemini":
    # ... existing code ...
```

### 3. Config format

```yaml
credentials:
  providers:
    vertex:
      type: google
      project: "project-20ca3d19-8a71-475f-89b"
      location: "us-central1"        # optional, default us-central1
      bucket: "my-custom-bucket"     # optional, auto-created if omitted

translation:
  models:
    - provider: vertex
      model: gemini-2.5-pro
      mode: batch
    - provider: gemini
      model: gemini-2.5-flash
      mode: online  # fallback
```

### 4. No changes needed

These components are already provider-agnostic:
- `executor.py` — uses batch client through uniform interface
- `batch_state.py` — `MegaUnitState` persistence
- `BatchRequest` / `BatchResponse` / `BatchJobInfo` / `BatchJobState` dataclasses
- Hooks, validators, pipeline, resume logic

## Key Risks / Open Questions

1. **JSONL format**: Verify Vertex batch accepts `{"key": "...", "request": {...}}` format.
   If not, `key` correlation needs to use line-order or a workaround.
2. **GCS dependency**: Need `google-cloud-storage` pip package for bucket operations.
   Alternatively, use raw REST with ADC token to avoid adding a heavy dependency.
3. **Location**: Vertex batch availability varies by region. `us-central1` is safest.
   The `location` parameter in `genai.Client()` can also be `"global"` — verify which works for batch.
4. **Result format**: Verify how Vertex batch exposes output — does `job.dest` work the same
   way via `google-genai` SDK, or does it return a GCS URI that needs manual download?
5. **Bucket cleanup**: Decide policy — clean up after each job? Keep for N days? Configurable?

## Estimated Scope

- ~100-150 lines new code in `batch_utils.py` (VertexBatchClient class)
- ~10 lines change in `factory_v2.py` (provider routing)
- ~5 lines config schema change
- New dependency: `google-cloud-storage` (for GCS operations)
- Total: small, well-contained change
