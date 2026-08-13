[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$NvidiaSmi = 'C:\Windows\System32\nvidia-smi.exe'
$GpuUuid = $env:CHANDRA_GPU_UUID
$ExpectedName = 'NVIDIA GeForce RTX 3090'
$DesiredWatts = if ($env:CHANDRA_POWER_LIMIT_W) {
    [double]$env:CHANDRA_POWER_LIMIT_W
} else {
    275.0
}
$LogPath = 'C:\ProgramData\Chandra\rtx3090-power-limit.log'

function Write-Log([string]$Message) {
    $timestamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK'
    Add-Content -LiteralPath $LogPath -Value "$timestamp $Message" -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $NvidiaSmi -PathType Leaf)) {
    throw "nvidia-smi not found at $NvidiaSmi"
}
if ([string]::IsNullOrWhiteSpace($GpuUuid)) {
    throw 'CHANDRA_GPU_UUID must identify the dedicated GPU'
}

$lastError = $null
for ($attempt = 1; $attempt -le 12; $attempt++) {
    try {
        $rows = @(& $NvidiaSmi `
            --query-gpu=uuid,name,power.default_limit,power.min_limit,power.max_limit `
            --format=csv,noheader,nounits)
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi inventory exited with $LASTEXITCODE"
        }

        $matches = @($rows | ForEach-Object {
            $fields = $_ -split ',\s*'
            if ($fields.Count -ne 5) {
                throw "Unexpected nvidia-smi inventory row: $_"
            }
            [pscustomobject]@{
                Uuid = $fields[0]
                Name = $fields[1]
                DefaultWatts = [double]$fields[2]
                MinWatts = [double]$fields[3]
                MaxWatts = [double]$fields[4]
            }
        } | Where-Object Uuid -eq $GpuUuid)

        if ($matches.Count -ne 1) {
            throw "Expected exactly one GPU with UUID $GpuUuid; found $($matches.Count)"
        }

        $gpu = $matches[0]
        if ($gpu.Name -ne $ExpectedName) {
            throw "GPU name mismatch for ${GpuUuid}: $($gpu.Name)"
        }
        if ($gpu.DefaultWatts -ne 350.0 -or
            $DesiredWatts -lt $gpu.MinWatts -or
            $DesiredWatts -gt $gpu.MaxWatts) {
            throw "Power contract mismatch: default=$($gpu.DefaultWatts), range=$($gpu.MinWatts)-$($gpu.MaxWatts), desired=$DesiredWatts"
        }

        & $NvidiaSmi -i $GpuUuid -pl $DesiredWatts | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi power-limit update exited with $LASTEXITCODE"
        }

        $actual = [double]((& $NvidiaSmi -i $GpuUuid `
            --query-gpu=power.limit --format=csv,noheader,nounits).Trim())
        if ($LASTEXITCODE -ne 0 -or [math]::Abs($actual - $DesiredWatts) -gt 0.1) {
            throw "Power-limit verification failed: expected $DesiredWatts W, observed $actual W"
        }

        Write-Log "OK uuid=$GpuUuid name='$ExpectedName' power_limit_w=$actual attempt=$attempt"
        exit 0
    }
    catch {
        $lastError = $_.Exception.Message
        if ($attempt -lt 12) {
            Start-Sleep -Seconds 5
        }
    }
}

Write-Log "ERROR uuid=$GpuUuid desired_w=$DesiredWatts message='$lastError'"
throw "Unable to set RTX 3090 power limit after 12 attempts: $lastError"
