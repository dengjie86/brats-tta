param(
    [string]$ExperimentRoot = "D:\PersonalData\Desktop\brats_ped_ssa_source_tent_20260920",
    [string]$RepositoryRoot = "D:\Workspace\deeplearn\3dunet",
    [string]$Checkpoint = "D:\PersonalData\Desktop\brats2023_gli_source_training_results_20260919\run\checkpoints\best.pt",
    [string]$HdBetWeights = "C:\Users\Yuuhi\hd-bet_params\release_2.0.0",
    [int]$PatchSize = 128
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$cvPython = "D:\Environment\miniforge3\envs\cv311\python.exe"
$hdBetPython = "D:\Environment\venvs\brats-hdbet\Scripts\python.exe"
$logs = Join-Path $ExperimentRoot "logs"
$statusPath = Join-Path $ExperimentRoot "pipeline_status.json"
$pedRawManifest = Join-Path $ExperimentRoot "manifests\ped_kaggle_legacy260_raw.json"
$ssaManifest = Join-Path $ExperimentRoot "manifests\ssa_raw.json"
$pedBrainManifest = Join-Path $ExperimentRoot "manifests\ped_brain.json"
$pedMaskRoot = Join-Path $ExperimentRoot "preprocessed\ped_brain_masks"

New-Item -ItemType Directory -Force -Path $logs, $pedMaskRoot | Out-Null
$env:PYTHONPATH = (Join-Path $RepositoryRoot "src") + [IO.Path]::PathSeparator + $env:PYTHONPATH

function Write-PipelineStatus {
    param(
        [string]$Stage,
        [string]$State,
        [string]$Message = ""
    )
    $payload = [ordered]@{
        pid = $PID
        state = $State
        stage = $Stage
        message = $Message
        patch_size = @($PatchSize, $PatchSize, $PatchSize)
        updated_at = [DateTimeOffset]::Now.ToString("o")
    }
    $temporary = "$statusPath.tmp"
    [IO.File]::WriteAllText(
        $temporary,
        ($payload | ConvertTo-Json -Depth 5),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Invoke-LoggedPython {
    param(
        [string]$Stage,
        [string]$Python,
        [string[]]$Arguments,
        [string]$LogName
    )
    $logPath = Join-Path $logs $LogName
    Write-PipelineStatus -Stage $Stage -State "running" -Message "See $logPath"
    $header = "{0} stage={1} command={2} {3}" -f [DateTimeOffset]::Now.ToString("o"), $Stage, $Python, ($Arguments -join " ")
    Add-Content -LiteralPath $logPath -Value $header -Encoding utf8
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Stage $Stage failed with exit code $exitCode. See $logPath"
    }
}

try {
    foreach ($required in @($cvPython, $hdBetPython, $Checkpoint, $pedRawManifest, $ssaManifest)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Required path does not exist: $required"
        }
    }

    Invoke-LoggedPython -Stage "ped_hdbet" -Python $hdBetPython -LogName "ped_hdbet.log" -Arguments @(
        "-m", "brats_tta.cli.extract_brain_masks",
        "--manifest", $pedRawManifest,
        "--output-root", $pedMaskRoot,
        "--output-manifest", $pedBrainManifest,
        "--weights", $HdBetWeights,
        "--device", "cuda",
        "--modality", "t1c",
        "--mirroring"
    )

    $pedManifest = Get-Content -Raw -LiteralPath $pedBrainManifest | ConvertFrom-Json
    if ($pedManifest.cases.Count -ne 260) {
        throw "Expected 260 brain-masked PED cases, found $($pedManifest.cases.Count)"
    }

    Invoke-LoggedPython -Stage "ped_source" -Python $cvPython -LogName "ped_source.log" -Arguments @(
        "-m", "brats_tta.cli.evaluate_tta",
        "--checkpoint", $Checkpoint,
        "--manifest", $pedBrainManifest,
        "--output-dir", (Join-Path $ExperimentRoot "source\ped"),
        "--methods", "source",
        "--device", "cuda",
        "--patch-size", "$PatchSize", "$PatchSize", "$PatchSize",
        "--overlap", "0.5",
        "--sw-batch-size", "1",
        "--no-amp",
        "--stall-trace-seconds", "300"
    )

    Invoke-LoggedPython -Stage "ssa_tent" -Python $cvPython -LogName "ssa_tent.log" -Arguments @(
        "-m", "brats_tta.cli.evaluate_tta",
        "--checkpoint", $Checkpoint,
        "--manifest", $ssaManifest,
        "--output-dir", (Join-Path $ExperimentRoot "tent\ssa"),
        "--methods", "tent",
        "--device", "cuda",
        "--patch-size", "$PatchSize", "$PatchSize", "$PatchSize",
        "--overlap", "0.5",
        "--sw-batch-size", "1",
        "--tent-lr", "0.001",
        "--tent-steps", "1",
        "--no-amp",
        "--stall-trace-seconds", "300"
    )

    Invoke-LoggedPython -Stage "ped_tent" -Python $cvPython -LogName "ped_tent.log" -Arguments @(
        "-m", "brats_tta.cli.evaluate_tta",
        "--checkpoint", $Checkpoint,
        "--manifest", $pedBrainManifest,
        "--output-dir", (Join-Path $ExperimentRoot "tent\ped"),
        "--methods", "tent",
        "--device", "cuda",
        "--patch-size", "$PatchSize", "$PatchSize", "$PatchSize",
        "--overlap", "0.5",
        "--sw-batch-size", "1",
        "--tent-lr", "0.001",
        "--tent-steps", "1",
        "--no-amp",
        "--stall-trace-seconds", "300"
    )

    Write-PipelineStatus -Stage "complete" -State "complete" -Message "PED Source and SSA/PED TENT completed"
}
catch {
    Write-PipelineStatus -Stage "failed" -State "failed" -Message $_.Exception.Message
    Add-Content -LiteralPath (Join-Path $logs "pipeline_error.log") -Value ("{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $_) -Encoding utf8
    throw
}
