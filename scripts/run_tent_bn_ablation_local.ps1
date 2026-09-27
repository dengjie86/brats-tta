param(
    [string]$ExperimentRoot = "D:\PersonalData\Desktop\brats_tta_ablation_20260922",
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
$settingsPath = Join-Path $ExperimentRoot "ablation_settings.json"

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

$variants = @(
    @{ Name = "tent_shallow_bn"; Extra = @("--tent-bn-scope", "shallow") },
    @{ Name = "tent_deep_bn"; Extra = @("--tent-bn-scope", "deep") },
    @{ Name = "bn_stats_only"; Extra = @("--no-tent-update-affine") },
    @{ Name = "tent_source_stats"; Extra = @("--no-tent-use-batch-stats") }
)
$datasets = @(
    @{ Name = "ssa"; Manifest = $SsaManifest },
    @{ Name = "ped"; Manifest = $PedManifest }
)

$settings = [ordered]@{
    created_at = [DateTimeOffset]::Now.ToString("o")
    experiment = "TENT BatchNorm scope and statistics ablation"
    checkpoint = $Checkpoint
    ssa_manifest = $SsaManifest
    ped_manifest = $PedManifest
    patch_size = @(128, 128, 128)
    overlap = 0.5
    sw_batch_size = 1
    precision = "fp32"
    learning_rate = 0.001
    steps_per_patch = 1
    episodic = $true
    variants = @($variants | ForEach-Object { $_.Name })
}
[IO.File]::WriteAllText(
    $settingsPath,
    ($settings | ConvertTo-Json -Depth 8),
    [Text.UTF8Encoding]::new($false)
)

foreach ($dataset in $datasets) {
    foreach ($variant in $variants) {
        $output = Join-Path $ExperimentRoot (Join-Path $dataset.Name $variant.Name)
        $logPath = Join-Path $logs ("{0}_{1}.log" -f $dataset.Name, $variant.Name)
        New-Item -ItemType Directory -Force -Path $output | Out-Null
        $arguments = @(
            "-m", "brats_tta.cli.evaluate_tta",
            "--checkpoint", $Checkpoint,
            "--manifest", $dataset.Manifest,
            "--output-dir", $output,
            "--methods", "tent",
            "--device", "cuda",
            "--patch-size", "128", "128", "128",
            "--overlap", "0.5",
            "--sw-batch-size", "1",
            "--tent-lr", "0.001",
            "--tent-steps", "1",
            "--no-amp",
            "--stall-trace-seconds", "300",
            "--verbose"
        ) + $variant.Extra
        $header = "{0} starting {1}/{2}; command={3} {4}" -f (
            [DateTimeOffset]::Now.ToString("o"),
            $dataset.Name,
            $variant.Name,
            $python,
            ($arguments -join " ")
        )
        $header | Tee-Object -FilePath $logPath -Append
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $python @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorActionPreference
        if ($exitCode -ne 0) {
            throw "Ablation failed for $($dataset.Name)/$($variant.Name). See $logPath"
        }
    }
}

"{0} all TENT BN ablations complete" -f [DateTimeOffset]::Now.ToString("o") |
    Tee-Object -FilePath (Join-Path $logs "complete.log") -Append
