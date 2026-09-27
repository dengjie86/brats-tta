param(
    [string]$ExperimentRoot = "D:\PersonalData\Desktop\brats_advanced_tta_full_local_20260921",
    [string]$RepositoryRoot = "D:\Workspace\deeplearn\3dunet",
    [string]$Checkpoint = "D:\PersonalData\Desktop\brats_tta_results_20260921\checkpoint\best.pt",
    [string]$SsaManifest = "D:\PersonalData\Desktop\brats_tta_results_20260921\manifests\ssa_raw.json",
    [string]$PedManifest = "D:\PersonalData\Desktop\brats_tta_results_20260921\manifests\ped_brain.json",
    [int]$PatchSize = 128,
    [double]$Overlap = 0.5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$python = "D:\Environment\miniforge3\envs\cv311\python.exe"
$logs = Join-Path $ExperimentRoot "logs"
$statusPath = Join-Path $ExperimentRoot "pipeline_status.json"
$pipelineLog = Join-Path $logs "pipeline.log"
$gpuLog = Join-Path $logs "gpu_monitor.csv"
$sourceArchive = Join-Path $ExperimentRoot "source_snapshot.tar.gz"
$completedStages = [System.Collections.Generic.List[string]]::new()
$gpuMonitor = $null

New-Item -ItemType Directory -Force -Path $ExperimentRoot, $logs | Out-Null

foreach ($required in @($python, $Checkpoint, $SsaManifest, $PedManifest)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path does not exist: $required"
    }
}

$env:PYTHONPATH = (Join-Path $RepositoryRoot "src") + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

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
        completed_stages = @($script:completedStages)
        expected_runs = 8
        patch_size = @($PatchSize, $PatchSize, $PatchSize)
        overlap = $Overlap
        precision = "fp32"
        updated_at = [DateTimeOffset]::Now.ToString("o")
    }
    $temporary = "$statusPath.tmp"
    [IO.File]::WriteAllText(
        $temporary,
        ($payload | ConvertTo-Json -Depth 6),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Invoke-AdvancedRun {
    param(
        [hashtable]$Run
    )
    $stage = "{0}_{1}" -f $Run.Variant, $Run.Dataset
    $output = Join-Path $ExperimentRoot (Join-Path $Run.Variant $Run.Dataset)
    $logPath = Join-Path $logs ("{0}.log" -f $stage)
    $arguments = @(
        "-m", "brats_tta.cli.evaluate_advanced_tta",
        "--checkpoint", $Checkpoint,
        "--manifest", $Run.Manifest,
        "--output-dir", $output,
        "--method", $Run.Method,
        "--device", "cuda",
        "--patch-size", "$PatchSize", "$PatchSize", "$PatchSize",
        "--overlap", "$Overlap",
        "--sw-batch-size", "1",
        "--seed", "2025",
        "--hd95",
        "--stall-trace-seconds", "300",
        "--verbose"
    )
    $arguments += $Run.Extra

    Write-PipelineStatus -Stage $stage -State "running" -Message "See $logPath"
    $header = "{0} starting {1}; command={2} {3}" -f (
        [DateTimeOffset]::Now.ToString("o"),
        $stage,
        $python,
        ($arguments -join " ")
    )
    $header | Tee-Object -FilePath $pipelineLog -Append
    $header | Tee-Object -FilePath $logPath -Append
    & $python @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Stage $stage failed with exit code $exitCode. See $logPath"
    }
    $script:completedStages.Add($stage)
    Write-PipelineStatus -Stage $stage -State "stage_complete" -Message "Completed $stage"
}

$datasets = [ordered]@{
    ssa = $SsaManifest
    ped = $PedManifest
}
$runs = [System.Collections.Generic.List[hashtable]]::new()

foreach ($dataset in $datasets.GetEnumerator()) {
    $runs.Add(@{
        Variant = "sar"
        Method = "sar"
        Dataset = $dataset.Key
        Manifest = $dataset.Value
        Extra = @(
            "--sar-lr", "0.001",
            "--sar-momentum", "0.9",
            "--sar-rho", "0.05",
            "--sar-entropy-margin", "0.5545177444479562",
            "--sar-recovery-threshold", "0.2",
            "--sar-steps", "1"
        )
    })
}
foreach ($dataset in $datasets.GetEnumerator()) {
    $runs.Add(@{
        Variant = "sar_no_recovery"
        Method = "sar"
        Dataset = $dataset.Key
        Manifest = $dataset.Value
        Extra = @(
            "--sar-lr", "0.001",
            "--sar-momentum", "0.9",
            "--sar-rho", "0.05",
            "--sar-entropy-margin", "0.5545177444479562",
            "--sar-no-recovery",
            "--sar-steps", "1"
        )
    })
}
foreach ($dataset in $datasets.GetEnumerator()) {
    $runs.Add(@{
        Variant = "vptta"
        Method = "vptta"
        Dataset = $dataset.Key
        Manifest = $dataset.Value
        Extra = @(
            "--vptta-lr", "0.05",
            "--vptta-prompt-alpha", "0.03",
            "--vptta-memory-size", "40",
            "--vptta-neighbors", "16",
            "--vptta-warm-n", "5.0",
            "--vptta-steps", "1"
        )
    })
}
foreach ($dataset in $datasets.GetEnumerator()) {
    $runs.Add(@{
        Variant = "tegda"
        Method = "tegda"
        Dataset = $dataset.Key
        Manifest = $dataset.Value
        Extra = @(
            "--tegda-lr", "0.0001",
            "--tegda-scope", "full",
            "--tegda-dropout-samples", "10",
            "--tegda-dropout-probability", "0.5",
            "--tegda-feature-bank-size", "10",
            "--tegda-feature-percentile", "0.90",
            "--tegda-steps", "1"
        )
    })
}

try {
    if (-not (Test-Path -LiteralPath $sourceArchive)) {
        Push-Location $RepositoryRoot
        try {
            & tar.exe -czf $sourceArchive `
                --exclude="__pycache__" `
                --exclude=".pytest_cache" `
                --exclude=".ruff_cache" `
                --exclude="*.egg-info" `
                configs docs scripts src tests README.md pyproject.toml uv.lock .gitignore .gitattributes
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create source snapshot"
            }
        }
        finally {
            Pop-Location
        }
    }

    $settings = [ordered]@{
        created_at = [DateTimeOffset]::Now.ToString("o")
        experiment = "formal local advanced TTA matrix"
        checkpoint = $Checkpoint
        checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint).Hash.ToLowerInvariant()
        datasets = [ordered]@{
            ssa = $SsaManifest
            ssa_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $SsaManifest).Hash.ToLowerInvariant()
            ped = $PedManifest
            ped_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PedManifest).Hash.ToLowerInvariant()
        }
        source_snapshot = $sourceArchive
        source_snapshot_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceArchive).Hash.ToLowerInvariant()
        method_order = @($runs | ForEach-Object { "{0}/{1}" -f $_.Variant, $_.Dataset })
        patch_size = @($PatchSize, $PatchSize, $PatchSize)
        overlap = $Overlap
        sw_batch_size = 1
        precision = "fp32"
        hd95 = $true
        seed = 2025
        continual_across_patches_and_cases = $true
        tegda = [ordered]@{
            scope = "full"
            learning_rate = 0.0001
            dropout_probability = 0.5
            dropout_samples = 10
            feature_bank_size = 10
            feature_percentile = 0.90
        }
    }
    $settingsPath = Join-Path $ExperimentRoot "pipeline_settings.json"
    if (-not (Test-Path -LiteralPath $settingsPath)) {
        [IO.File]::WriteAllText(
            $settingsPath,
            ($settings | ConvertTo-Json -Depth 8),
            [Text.UTF8Encoding]::new($false)
        )
    }

    git -C $RepositoryRoot status --short | Set-Content -LiteralPath (
        Join-Path $ExperimentRoot "git_status.txt"
    ) -Encoding utf8
    & $python -m pip freeze | Set-Content -LiteralPath (
        Join-Path $ExperimentRoot "pip_freeze.txt"
    ) -Encoding utf8
    nvidia-smi -q | Set-Content -LiteralPath (
        Join-Path $ExperimentRoot "nvidia_smi_start.txt"
    ) -Encoding utf8

    $nvidiaSmi = (Get-Command nvidia-smi).Source
    $gpuMonitor = Start-Process -FilePath $nvidiaSmi -ArgumentList @(
        "--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv",
        "-l", "60"
    ) -RedirectStandardOutput $gpuLog -WindowStyle Hidden -PassThru

    Write-PipelineStatus -Stage "starting" -State "running" -Message "Formal matrix starting"
    foreach ($run in $runs) {
        Invoke-AdvancedRun -Run $run
    }
    Write-PipelineStatus -Stage "complete" -State "complete" -Message "All eight runs completed"
    "{0} all runs complete" -f [DateTimeOffset]::Now.ToString("o") |
        Tee-Object -FilePath $pipelineLog -Append
}
catch {
    Write-PipelineStatus -Stage "failed" -State "failed" -Message $_.Exception.Message
    ("{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $_) |
        Tee-Object -FilePath (Join-Path $logs "pipeline_error.log") -Append
    throw
}
finally {
    if ($null -ne $gpuMonitor -and -not $gpuMonitor.HasExited) {
        Stop-Process -Id $gpuMonitor.Id -Force -ErrorAction SilentlyContinue
    }
}
