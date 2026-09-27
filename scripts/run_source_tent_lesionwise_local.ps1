param(
    [string]$ExperimentRoot = "D:\PersonalData\Desktop\brats_source_tent_lesionwise_20260921",
    [string]$RepositoryRoot = "D:\Workspace\deeplearn\3dunet",
    [string]$Checkpoint = "D:\PersonalData\Desktop\brats_tta_results_20260921\checkpoint\best.pt",
    [string]$SsaManifest = "D:\PersonalData\Desktop\brats_tta_results_20260921\manifests\ssa_raw.json",
    [string]$PedManifest = "D:\PersonalData\Desktop\brats_tta_results_20260921\manifests\ped_brain.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$python = "D:\Environment\miniforge3\envs\cv311\python.exe"
$logs = Join-Path $ExperimentRoot "logs"
$statusPath = Join-Path $ExperimentRoot "pipeline_status.json"
$pipelineLog = Join-Path $logs "pipeline.log"
$completedStages = [System.Collections.Generic.List[string]]::new()

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
        expected_runs = 4
        metric = "BraTS 2023 lesion-wise Dice"
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

function Invoke-Evaluation {
    param([hashtable]$Run)

    $stage = "{0}_{1}" -f $Run.Method, $Run.Dataset
    $output = Join-Path $ExperimentRoot $Run.Dataset
    $logPath = Join-Path $logs "$stage.log"
    $arguments = @(
        "-m", "brats_tta.cli.evaluate_tta",
        "--checkpoint", $Checkpoint,
        "--manifest", $Run.Manifest,
        "--output-dir", $output,
        "--methods", $Run.Method,
        "--device", "cuda",
        "--patch-size", "128", "128", "128",
        "--overlap", "0.5",
        "--sw-batch-size", "1",
        "--no-amp",
        "--tent-lr", "0.001",
        "--tent-steps", "1",
        "--lesion-wise",
        "--lesion-dilation-factor", "3",
        "--lesion-volume-threshold-mm3", "50",
        "--stall-trace-seconds", "300",
        "--verbose"
    )

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
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $stage failed with exit code $LASTEXITCODE. See $logPath"
    }
    $script:completedStages.Add($stage)
    Write-PipelineStatus -Stage $stage -State "stage_complete" -Message "Completed $stage"
}

$runs = @(
    @{Method = "source"; Dataset = "ssa"; Manifest = $SsaManifest},
    @{Method = "source"; Dataset = "ped"; Manifest = $PedManifest},
    @{Method = "tent"; Dataset = "ssa"; Manifest = $SsaManifest},
    @{Method = "tent"; Dataset = "ped"; Manifest = $PedManifest}
)

try {
    $settings = [ordered]@{
        created_at = [DateTimeOffset]::Now.ToString("o")
        experiment = "Source and episodic TENT with BraTS 2023 lesion-wise Dice"
        checkpoint = $Checkpoint
        checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint).Hash.ToLowerInvariant()
        ssa_manifest = $SsaManifest
        ped_manifest = $PedManifest
        method_order = @($runs | ForEach-Object { "{0}/{1}" -f $_.Method, $_.Dataset })
        patch_size = @(128, 128, 128)
        overlap = 0.5
        sw_batch_size = 1
        precision = "fp32"
        tent_learning_rate = 0.001
        tent_steps_per_patch = 1
        tent_episodic = $true
        lesion_connectivity = 26
        lesion_dilation_structure_connectivity = 18
        lesion_dilation_factor = 3
        lesion_volume_threshold_mm3 = 50.0
    }
    [IO.File]::WriteAllText(
        (Join-Path $ExperimentRoot "pipeline_settings.json"),
        ($settings | ConvertTo-Json -Depth 8),
        [Text.UTF8Encoding]::new($false)
    )
    git -C $RepositoryRoot status --short | Set-Content -LiteralPath (
        Join-Path $ExperimentRoot "git_status.txt"
    ) -Encoding utf8

    Write-PipelineStatus -Stage "starting" -State "running" -Message "Lesion-wise evaluation starting"
    foreach ($run in $runs) {
        Invoke-Evaluation -Run $run
    }
    Write-PipelineStatus -Stage "complete" -State "complete" -Message "All four runs completed"
    "{0} all runs complete" -f [DateTimeOffset]::Now.ToString("o") |
        Tee-Object -FilePath $pipelineLog -Append
}
catch {
    Write-PipelineStatus -Stage "failed" -State "failed" -Message $_.Exception.Message
    ("{0} {1}" -f [DateTimeOffset]::Now.ToString("o"), $_) |
        Tee-Object -FilePath (Join-Path $logs "pipeline_error.log") -Append
    throw
}
