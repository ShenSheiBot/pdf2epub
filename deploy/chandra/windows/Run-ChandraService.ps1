[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ServiceRoot = $env:CHANDRA_SERVICE_ROOT
if ([string]::IsNullOrWhiteSpace($ServiceRoot)) {
    throw 'CHANDRA_SERVICE_ROOT must point to the native Windows vLLM environment'
}
$Python = Join-Path $ServiceRoot '.venv\Scripts\python.exe'
$PowerScript = 'C:\ProgramData\Chandra\Set-RTX3090PowerLimit.ps1'
$LogRoot = 'C:\ProgramData\Chandra\logs'
$PidPath = 'C:\ProgramData\Chandra\chandra-vllm.pid'
$TreePath = 'C:\ProgramData\Chandra\chandra-process-tree.json'
$NvidiaSmi = 'C:\Windows\System32\nvidia-smi.exe'
$GpuUuid = $env:CHANDRA_GPU_UUID
$DesiredPowerWatts = if ($env:CHANDRA_POWER_LIMIT_W) {
    [double]$env:CHANDRA_POWER_LIMIT_W
} else {
    275.0
}
$Model = 'datalab-to/chandra-ocr-2'
$Revision = 'af93b47dba1b47b6640c86ccf487ed2260ab9a09'
$Port = 8100
$HealthUrl = "http://127.0.0.1:$Port/health"

function Write-ServiceLog([string]$Message) {
    $timestamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK'
    Add-Content -LiteralPath (Join-Path $LogRoot 'service.log') -Value "$timestamp $Message" -Encoding UTF8
}

function Test-Health {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 5
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Get-ChandraProcesses {
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -like '*vllm.entrypoints.cli.main serve datalab-to/chandra-ocr-2*' -and
        $_.CommandLine -like '*--served-model-name chandra*' -and
        $_.CommandLine -like '*--port 8100*'
    })
}

function Get-ManagedProcessTree([int[]]$RootProcessIds) {
    $all = @(Get-CimInstance Win32_Process)
    $children = @{}
    foreach ($process in $all) {
        $parentId = [int]$process.ParentProcessId
        if (-not $children.ContainsKey($parentId)) {
            $children[$parentId] = @()
        }
        $children[$parentId] += $process
    }
    $byId = @{}
    foreach ($process in $all) {
        $byId[[int]$process.ProcessId] = $process
    }
    $queue = [System.Collections.Generic.Queue[object]]::new()
    foreach ($rootId in $RootProcessIds) {
        $queue.Enqueue([pscustomobject]@{ ProcessId = $rootId; Depth = 0 })
    }
    $seen = @{}
    $result = @()
    while ($queue.Count -gt 0) {
        $item = $queue.Dequeue()
        $processId = [int]$item.ProcessId
        if ($seen.ContainsKey($processId) -or -not $byId.ContainsKey($processId)) {
            continue
        }
        $seen[$processId] = $true
        $process = $byId[$processId]
        $isManagedPython = $process.Name -eq 'python.exe' -and (
            $process.CommandLine -like '*vllm.entrypoints.cli.main serve datalab-to/chandra-ocr-2*' -or
            $process.CommandLine -like '*multiprocessing.spawn*spawn_main*'
        )
        if (-not $isManagedPython) {
            continue
        }
        $result += [pscustomobject]@{
            ProcessId = $processId
            ParentProcessId = [int]$process.ParentProcessId
            Depth = [int]$item.Depth
            CreationTicksUtc = $process.CreationDate.ToUniversalTime().Ticks
            ExecutablePath = $process.ExecutablePath
            CommandLine = $process.CommandLine
        }
        foreach ($child in @($children[$processId])) {
            $queue.Enqueue([pscustomobject]@{
                ProcessId = [int]$child.ProcessId
                Depth = [int]$item.Depth + 1
            })
        }
    }
    return @($result)
}

function Save-ManagedProcessTree([object[]]$Processes) {
    if ($Processes.Count -eq 0) {
        throw 'Refusing to replace the Chandra process record with an empty tree'
    }
    $temporary = "$TreePath.new"
    $Processes | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $TreePath -Force
}

function Stop-RecordedProcessTree([object[]]$Processes) {
    foreach ($record in @($Processes | Sort-Object Depth -Descending)) {
        $processId = [int]$record.ProcessId
        if ($processId -le 4) {
            throw "Refusing invalid recorded process id $processId"
        }
        $current = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
        if ($null -eq $current) {
            continue
        }
        if ($current.CreationDate.ToUniversalTime().Ticks -ne [long]$record.CreationTicksUtc -or
            $current.ExecutablePath -ne [string]$record.ExecutablePath -or
            $current.CommandLine -ne [string]$record.CommandLine) {
            throw "Refusing PID $processId because it no longer matches the recorded Chandra process"
        }
        Stop-Process -Id $processId -Force
    }
    Remove-Item -LiteralPath $TreePath -Force -ErrorAction SilentlyContinue
}

function Stop-ValidatedProcessTree([int]$ProcessId) {
    if ($ProcessId -le 4) {
        throw "Refusing invalid Chandra process id $ProcessId"
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return
    }
    if ($process.Name -ne 'python.exe' -or
        $process.CommandLine -notlike '*vllm.entrypoints.cli.main serve datalab-to/chandra-ocr-2*' -or
        $process.CommandLine -notlike '*--port 8100*') {
        throw "Refusing to stop PID $ProcessId because its command line is not the managed Chandra service"
    }
    & C:\Windows\System32\taskkill.exe /PID $ProcessId /T /F | Out-Null
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

while ($true) {
try {
$managedTree = @()
if ([string]::IsNullOrWhiteSpace($GpuUuid)) {
    throw 'CHANDRA_GPU_UUID must identify the dedicated GPU'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Chandra Python is missing: $Python"
}
if (-not (Test-Path -LiteralPath $PowerScript -PathType Leaf)) {
    throw "RTX 3090 power-limit script is missing: $PowerScript"
}

& C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe `
    -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $PowerScript
if ($LASTEXITCODE -ne 0) {
    throw "RTX 3090 power-limit script exited with $LASTEXITCODE"
}
$powerLimit = [double]((& $NvidiaSmi -i $GpuUuid `
    --query-gpu=power.limit --format=csv,noheader,nounits).Trim())
if ([math]::Abs($powerLimit - $DesiredPowerWatts) -gt 0.1) {
    throw "RTX 3090 power limit is $powerLimit W instead of $DesiredPowerWatts W"
}

$existing = Get-ChandraProcesses
if ($existing.Count -gt 0) {
    if (-not (Test-Health)) {
        throw "Found existing Chandra processes but port $Port is not healthy; refusing a duplicate start"
    }
    $existingIds = @($existing.ProcessId)
    $rootIds = @($existing | Where-Object { $_.ParentProcessId -notin $existingIds } | ForEach-Object ProcessId)
    if ($rootIds.Count -ne 1) {
        throw "Expected exactly one Chandra root while adopting; found $($rootIds.Count)"
    }
    $managedTree = Get-ManagedProcessTree $rootIds
    Save-ManagedProcessTree $managedTree
    Set-Content -LiteralPath $PidPath -Value $rootIds[0] -Encoding ASCII
    Write-ServiceLog "ADOPT healthy existing_pids=$($existing.ProcessId -join ',')"
}
else {
    if (Test-Path -LiteralPath $TreePath -PathType Leaf) {
        # Windows PowerShell 5.1 emits a JSON array as one nested pipeline item.
        # Re-enumerate it so each process record reaches the typed cleanup code.
        $recordedTree = @((Get-Content -LiteralPath $TreePath -Raw | ConvertFrom-Json) |
            ForEach-Object { $_ })
        Stop-RecordedProcessTree $recordedTree
    }
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Port $Port is already owned by PID $($listener.OwningProcess), not Chandra"
    }

    $cudaRoot = Join-Path $ServiceRoot '.venv\Lib\site-packages\nvidia\cu13'
    Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:CUDA_LIB_PATH -ErrorAction SilentlyContinue
    $env:CUDA_HOME = $cudaRoot
    $env:VLLM_USE_FLASHINFER_SAMPLER = '0'
    $env:PYTHONUTF8 = '1'
    $cacheRoot = if ($env:CHANDRA_CACHE_ROOT) {
        $env:CHANDRA_CACHE_ROOT
    } else {
        Join-Path $ServiceRoot 'cache'
    }
    $env:HF_HOME = Join-Path $cacheRoot 'huggingface'
    $env:VLLM_CACHE_ROOT = Join-Path $cacheRoot 'vllm'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:PATH = "$(Join-Path $ServiceRoot '.venv\Scripts');$(Join-Path $ServiceRoot '.venv\Lib\site-packages\torch\lib');$(Join-Path $cudaRoot 'bin\x86_64');$(Join-Path $cudaRoot 'bin');$env:PATH"

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $LogRoot "vllm-$stamp.stdout.log"
    $stderr = Join-Path $LogRoot "vllm-$stamp.stderr.log"
    $arguments = @(
        '-m', 'vllm.entrypoints.cli.main', 'serve', $Model,
        '--revision', $Revision,
        '--host', '0.0.0.0',
        '--port', "$Port",
        '--served-model-name', 'chandra',
        '--dtype', 'bfloat16',
        '--max-model-len', '18000',
        '--max-num-seqs', '8',
        '--max-num-batched-tokens', '2048',
        '--gpu-memory-utilization', '0.70',
        '--disable-log-stats'
    )
    $child = Start-Process -FilePath $Python -ArgumentList $arguments `
        -WorkingDirectory $ServiceRoot -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath $PidPath -Value $child.Id -Encoding ASCII
    Write-ServiceLog "START root_pid=$($child.Id) stdout='$stdout' stderr='$stderr' power_limit_w=$powerLimit"

    $startupDeadline = (Get-Date).AddMinutes(8)
    while (-not (Test-Health)) {
        if ($child.HasExited) {
            throw "Chandra exited during startup with code $($child.ExitCode); see $stderr"
        }
        if ((Get-Date) -ge $startupDeadline) {
            Stop-ValidatedProcessTree $child.Id
            throw "Chandra did not become healthy within eight minutes; see $stderr"
        }
        Start-Sleep -Seconds 5
        $child.Refresh()
    }
    $managedTree = Get-ManagedProcessTree @($child.Id)
    Save-ManagedProcessTree $managedTree
    Write-ServiceLog "READY root_pid=$($child.Id) url='$HealthUrl'"
}

$consecutiveFailures = 0
while ($true) {
    Start-Sleep -Seconds 30
    if (Test-Health) {
        $consecutiveFailures = 0
        $current = Get-ChandraProcesses
        $currentIds = @($current.ProcessId)
        $rootIds = @($current | Where-Object { $_.ParentProcessId -notin $currentIds } | ForEach-Object ProcessId)
        if ($rootIds.Count -ne 1) {
            throw "Expected exactly one healthy Chandra root; found $($rootIds.Count)"
        }
        $managedTree = Get-ManagedProcessTree $rootIds
        Save-ManagedProcessTree $managedTree
        Set-Content -LiteralPath $PidPath -Value $rootIds[0] -Encoding ASCII
        continue
    }
    $consecutiveFailures++
    Write-ServiceLog "UNHEALTHY consecutive_failures=$consecutiveFailures"
    if ($consecutiveFailures -lt 3) {
        continue
    }
    $managed = Get-ChandraProcesses
    foreach ($process in $managed) {
        Stop-ValidatedProcessTree $process.ProcessId
    }
    throw "Chandra failed three consecutive health checks"
}
}
catch {
    $failure = $_.Exception.Message.Replace("`r", ' ').Replace("`n", ' ')
    Write-ServiceLog "RESTART_PENDING reason='$failure' delay_seconds=60"
    try {
        if ($managedTree.Count -eq 0 -and (Test-Path -LiteralPath $TreePath -PathType Leaf)) {
            # Keep the recovery path compatible with Windows PowerShell 5.1 too.
            $managedTree = @((Get-Content -LiteralPath $TreePath -Raw | ConvertFrom-Json) |
                ForEach-Object { $_ })
        }
        Stop-RecordedProcessTree $managedTree
        $managed = Get-ChandraProcesses
        foreach ($process in $managed) {
            Stop-ValidatedProcessTree $process.ProcessId
        }
    }
    catch {
        $cleanupFailure = $_.Exception.Message.Replace("`r", ' ').Replace("`n", ' ')
        Write-ServiceLog "CLEANUP_FAILED reason='$cleanupFailure'"
    }
    Start-Sleep -Seconds 60
}
}
