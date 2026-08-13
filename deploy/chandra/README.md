# Chandra 2 OCR service

This service runs one persistent Chandra vLLM replica on one 24 GB NVIDIA GPU.
The pdf2epub process remains a small HTTP client; it does not import torch,
vLLM, or the upstream Chandra package.

## Native Windows deployment

The native deployment deliberately avoids Docker Desktop and WSL. One tested
layout is:

```text
C:\Services\chandra-native\.venv
C:\ProgramData\Chandra\Run-ChandraService.ps1
C:\ProgramData\Chandra\Set-RTX3090PowerLimit.ps1
```

The versioned copies of the two operational scripts are in [`windows/`](windows/).
They pin the model revision, 275 W default power limit, CUDA environment,
offline Hugging Face cache, port, and vLLM capacity. The task environment must
provide `CHANDRA_SERVICE_ROOT` and `CHANDRA_GPU_UUID`; `CHANDRA_CACHE_ROOT` and
`CHANDRA_POWER_LIMIT_W` are optional. After changing a versioned copy, install
it in `C:\ProgramData\Chandra` and compare its SHA-256 hash before restarting
the named task.

Two SYSTEM/highest-privilege Scheduled Tasks provide boot persistence:

```text
Chandra RTX 3090 Power Limit   boot + 30 seconds
Chandra OCR Service           boot + 60 seconds
```

The service task runs a persistent supervisor. It requires three consecutive
failed health probes before killing only command-line-validated Chandra
processes, waits 60 seconds, and starts a fresh server. This in-process loop is
intentional: the Task Scheduler `RestartOnFailure` metadata did not actually
relaunch the task in a real forced-process-failure test on this host.

The native endpoint listens on `0.0.0.0:8100`. Restrict Windows Firewall TCP
8100 ingress to the actual client addresses. The pdf2epub client URL is:

```text
http://WINDOWS_HOST:8100/v1
```

Operational checks from the Mac:

```bash
curl -fsS http://WINDOWS_HOST:8100/v1/models
powershell -NoProfile -Command "Get-ScheduledTask -TaskName 'Chandra OCR Service' | Select TaskName,State"
powershell -NoProfile -Command "Get-Content C:\ProgramData\Chandra\logs\service.log -Tail 30"
```

The first boot after a driver, torch, vLLM, or model change can spend several
minutes compiling. Do not treat the server as ready until `/health` returns
HTTP 200. Normal restarts use the persistent Hugging Face and vLLM caches under
the configured cache root and do not need network access.

## Linux/Docker alternative

`compose.yaml` remains a Linux deployment option.

### Start or update

From this directory:

```bash
docker compose pull
docker compose up -d
docker compose ps
```

The endpoint binds to `127.0.0.1:8100` by default. Set `CHANDRA_PORT` to select
another host port. A separate workstation may set
`CHANDRA_BIND_ADDRESS=0.0.0.0` and use an OS firewall rule restricted to the
trusted client/subnet. Hugging Face weights and vLLM compile artifacts persist
in Docker-managed `huggingface-cache` and `vllm-cache` volumes. Named volumes
avoid slow NTFS checkpoint reads and Docker Desktop/WSL cross-distro mount
availability during reboot.

The container uses `restart: always`. Once Docker starts after a host reboot,
the healthy container comes back without running another repository command.
Docker Desktop must itself be enabled at login on Windows/WSL hosts.

## pdf2epub configuration

```yaml
ocr:
  backend: chandra
  backends:
    chandra:
      base_url: http://WINDOWS_HOST:8100/v1
      model: chandra
      model_revision: af93b47dba1b47b6640c86ccf487ed2260ab9a09
      max_workers: 4
      max_output_tokens: 12384
      dpi: 192
      min_dimension: 1024
      include_headers_footers: false
```

The native server admits at most eight active sequences and continuously
batches the four page requests normally submitted by the client. On this 24 GB
RTX 3090, four different pages achieved nearly the same aggregate throughput as
eight with substantially lower per-page latency, so `max_workers: 4` is the
production default.

## Durable page artifacts

For each completed page, the backend writes:

```text
pages/page_NNN.md        compatibility view consumed by refine/polish
pages/page_NNN.html      materialized HTML retaining data-label/data-bbox
pages/page_NNN.raw.html  exact model response
pages/page_NNN.ocr.json  ordered blocks, both bbox forms, assets and model data
images/page_NNN_img_K.png
```

The raw response is never reconstructed from Markdown. Images nested inside
tables, diagrams, and complex blocks are recursively cropped before Markdown
conversion. Page headers and footers may be omitted from Markdown using the
configuration above; they remain present in HTML, raw HTML, and JSON.

No hard-coded blank-page detector discards model output. Blank or questionable
pages remain auditable in the lossless artifacts and can be cleaned by the
normal structure/polish stages.

## Docker health and shutdown

```bash
curl -fsS http://127.0.0.1:8100/v1/models
docker compose logs --tail 100 chandra
docker compose down
```

Initial startup can take several minutes while downloading and compiling the
model. The production Compose service uses Hugging Face offline mode so a
reboot cannot fail while reconstructing an already cached Xet blob. On a new
machine, populate the named `huggingface-cache` volume with the pinned model
snapshot once before starting this service. Subsequent starts reuse both named
volumes without network access.
