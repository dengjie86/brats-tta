param(
    [string]$ExperimentRoot = "D:\PersonalData\Desktop\brats_advanced_tta_pilot_20260921",
    [string]$RepositoryRoot = "D:\Workspace\deeplearn\3dunet",
    [string]$Checkpoint = "D:\PersonalData\Desktop\brats_tta_results_20260921\checkpoint\best.pt",
    [string]$SsaManifest = "D:\PersonalData\Desktop\brats_tta_results_20260921\manifests\ssa_raw.json",
    [string]$PedManifest = "D:\PersonalData\Desktop\brats_tta_results_20260921\manifests\ped_brain.json",
    [string[]]$Methods = @("sar", "vptta", "tegda"),
    [int]$Limit = 1,
    [int]$PatchSize = 96,
    [double]$Overlap = 0.0,
    [bool]$IncludeBaselines = $true,
    [bool]$IncludeSarNoRecovery = $true,
    [bool]$Hd95 = $true,
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$python = "D:\Environment\miniforge3\envs\cv311\python.exe"
$env:PYTHONPATH = (Join-Path $RepositoryRoot "src") + [IO.Path]::PathSeparator + $env:PYTHONPATH
$logs = Join-Path $ExperimentRoot "logs"
New-Item -ItemType Directory -Force -Path $ExperimentRoot, $logs | Out-Null

$settings = [ordered]@{
    created_at = [DateTimeOffset]::Now.ToString("o")
    purpose = "execution and memory pilot; not final-cohort results"
    checkpoint = $Checkpoint
    datasets = [ordered]@{ ssa = $SsaManifest; ped = $PedManifest }
    methods = $Methods
    limit = $Limit
    patch_size = @($PatchSize, $PatchSize, $PatchSize)
    overlap = $Overlap
    tegda_dropout_samples = 3
    tegda_scope = "norm"
    include_matching_source_tent_baselines = $IncludeBaselines
    include_sar_no_recovery_ablation = $IncludeSarNoRecovery
    hd95 = $Hd95
}
$settings | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
    Join-Path $ExperimentRoot "pilot_settings.json"
) -Encoding utf8

foreach ($dataset in ([ordered]@{ ssa = $SsaManifest; ped = $PedManifest }).GetEnumerator()) {
    if ($IncludeBaselines) {
        $baselineOutput = Join-Path $ExperimentRoot (Join-Path "baseline" $dataset.Key)
        $baselineLog = Join-Path $logs ("baseline_{0}.log" -f $dataset.Key)
        $baselineArguments = @(
            "-m", "brats_tta.cli.evaluate_tta",
            "--checkpoint", $Checkpoint,
            "--manifest", $dataset.Value,
            "--output-dir", $baselineOutput,
            "--methods", "source", "tent",
            "--device", "cuda",
            "--patch-size", "$PatchSize", "$PatchSize", "$PatchSize",
            "--overlap", "$Overlap",
            "--sw-batch-size", "1",
            "--limit", "$Limit",
            "--no-amp",
            "--stall-trace-seconds", "300"
        )
        if ($Hd95) {
            $baselineArguments += "--hd95"
        }
        if ($Overwrite) {
            $baselineArguments += "--overwrite"
        }
        "$(Get-Date -Format o) starting source+tent/$($dataset.Key)" |
            Tee-Object -FilePath $baselineLog -Append
        & $python @baselineArguments 2>&1 | Tee-Object -FilePath $baselineLog -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Baseline $($dataset.Key) failed with exit code $LASTEXITCODE. See $baselineLog"
        }
    }
    foreach ($method in $Methods) {
        $output = Join-Path $ExperimentRoot (Join-Path $method $dataset.Key)
        $log = Join-Path $logs ("{0}_{1}.log" -f $method, $dataset.Key)
        $arguments = @(
            "-m", "brats_tta.cli.evaluate_advanced_tta",
            "--checkpoint", $Checkpoint,
            "--manifest", $dataset.Value,
            "--output-dir", $output,
            "--method", $method,
            "--device", "cuda",
            "--patch-size", "$PatchSize", "$PatchSize", "$PatchSize",
            "--overlap", "$Overlap",
            "--sw-batch-size", "1",
            "--limit", "$Limit",
            "--stall-trace-seconds", "300"
        )
        if ($Hd95) {
            $arguments += "--hd95"
        }
        if ($Overwrite) {
            $arguments += "--overwrite"
        }
        if ($method -eq "tegda") {
            $arguments += @(
                "--tegda-scope", "norm",
                "--tegda-dropout-samples", "3"
            )
        }
        "$(Get-Date -Format o) starting $method/$($dataset.Key)" | Tee-Object -FilePath $log -Append
        & $python @arguments 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Pilot $method/$($dataset.Key) failed with exit code $LASTEXITCODE. See $log"
        }

        if ($method -eq "sar" -and $IncludeSarNoRecovery) {
            $ablationOutput = Join-Path $ExperimentRoot (
                Join-Path "sar_no_recovery" $dataset.Key
            )
            $ablationLog = Join-Path $logs (
                "sar_no_recovery_{0}.log" -f $dataset.Key
            )
            $ablationArguments = @(
                "-m", "brats_tta.cli.evaluate_advanced_tta",
                "--checkpoint", $Checkpoint,
                "--manifest", $dataset.Value,
                "--output-dir", $ablationOutput,
                "--method", "sar",
                "--device", "cuda",
                "--patch-size", "$PatchSize", "$PatchSize", "$PatchSize",
                "--overlap", "$Overlap",
                "--sw-batch-size", "1",
                "--limit", "$Limit",
                "--sar-no-recovery",
                "--stall-trace-seconds", "300"
            )
            if ($Hd95) {
                $ablationArguments += "--hd95"
            }
            if ($Overwrite) {
                $ablationArguments += "--overwrite"
            }
            "$(Get-Date -Format o) starting sar_no_recovery/$($dataset.Key)" |
                Tee-Object -FilePath $ablationLog -Append
            & $python @ablationArguments 2>&1 |
                Tee-Object -FilePath $ablationLog -Append
            if ($LASTEXITCODE -ne 0) {
                throw "SAR no-recovery $($dataset.Key) failed with exit code $LASTEXITCODE. See $ablationLog"
            }
        }
    }
}
