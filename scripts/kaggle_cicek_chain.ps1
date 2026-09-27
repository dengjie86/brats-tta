param(
    [string]$InitialKernel = "nagisa0v0/brats-gli-cicek-fp32-segment-01",
    [int]$FinalEpoch = 300,
    [int]$PollSeconds = 300,
    [string]$Root = "D:\PersonalData\Desktop\kaggle_cicek_chain_20260923"
)

$ErrorActionPreference = "Stop"
$Kaggle = "D:\Environment\miniforge3\envs\cv311\Scripts\kaggle.exe"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SourceDataset = "nagisa0v0/brats-tta-source-cicek-20260923"
$GliDataset = "nagisa0v0/brats2023-gli-training-data"
$LogPath = Join-Path $Root "chain.log"

New-Item -ItemType Directory -Force -Path $Root | Out-Null

function Write-ChainLog {
    param([string]$Message)
    $line = "{0} {1}" -f ([DateTime]::UtcNow.ToString("o")), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

function Get-KernelState {
    param([string]$KernelId)
    $output = & $Kaggle kernels status $KernelId 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle status failed for ${KernelId}: $output"
    }
    $match = [regex]::Match($output, 'KernelWorkerStatus\.([A-Z_]+)')
    if (-not $match.Success) {
        throw "Could not parse Kaggle status for ${KernelId}: $output"
    }
    return $match.Groups[1].Value
}

function Download-KernelOutput {
    param(
        [string]$KernelId,
        [int]$SegmentNumber
    )
    $destination = Join-Path $Root ("segment_{0:D2}" -f $SegmentNumber)
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Write-ChainLog "Downloading output for $KernelId to $destination"
    & $Kaggle kernels output $KernelId -p $destination -q -o
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle output download failed for ${KernelId}"
    }
    return (Resolve-Path $destination).Path
}

function Find-LatestCheckpoint {
    param([string]$OutputRoot)
    $matches = @(Get-ChildItem -LiteralPath $OutputRoot -Recurse -File -Filter "latest.pt")
    if ($matches.Count -eq 0) {
        return $null
    }
    return $matches[0]
}

function Get-CompletedEpoch {
    param([string]$OutputRoot)
    $summary = @(Get-ChildItem -LiteralPath $OutputRoot -Recurse -File -Filter "training_job_summary.json") | Select-Object -First 1
    if ($null -ne $summary) {
        $payload = Get-Content -Raw -LiteralPath $summary.FullName | ConvertFrom-Json
        if ($null -ne $payload.completed_epoch) {
            return [int]$payload.completed_epoch
        }
    }
    $latest = Find-LatestCheckpoint $OutputRoot
    if ($null -eq $latest) {
        return $null
    }
    $inspect = @"
import torch
checkpoint = torch.load(r'''$($latest.FullName)''', map_location='cpu', weights_only=True)
print(int(checkpoint['epoch']) + 1)
"@
    $epoch = & "D:\Environment\miniforge3\envs\cv311\python.exe" -c $inspect
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect checkpoint $($latest.FullName)"
    }
    return [int]($epoch | Select-Object -Last 1)
}

function New-ResumeDataset {
    param(
        [string]$OutputRoot,
        [int]$SegmentNumber
    )
    $latest = Find-LatestCheckpoint $OutputRoot
    if ($null -eq $latest) {
        throw "No latest.pt found in $OutputRoot"
    }
    $runRoot = $latest.Directory.Parent.FullName
    $stage = Join-Path $Root ("resume_dataset_{0:D2}" -f $SegmentNumber)
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    $runDestination = Join-Path $stage "run"
    $checkpointDestination = Join-Path $runDestination "checkpoints"
    New-Item -ItemType Directory -Force -Path $checkpointDestination | Out-Null
    Copy-Item -LiteralPath $latest.FullName -Destination $checkpointDestination -Force
    $best = Join-Path $latest.Directory.FullName "best.pt"
    if (Test-Path -LiteralPath $best) {
        Copy-Item -LiteralPath $best -Destination $checkpointDestination -Force
    }
    foreach ($filename in @("config.yaml", "history.jsonl", "training.log")) {
        $sourceFile = Join-Path $runRoot $filename
        if (Test-Path -LiteralPath $sourceFile) {
            Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $runDestination $filename) -Force
        }
    }
    $slug = "brats-gli-cicek-resume-{0:D2}" -f $SegmentNumber
    $metadata = @{
        title = $slug
        id = "nagisa0v0/$slug"
        subtitle = "Cicek 3D U-Net FP32 resume checkpoint segment $SegmentNumber"
        description = "Private checkpoint-only dataset for resumable BraTS GLI Cicek 3D U-Net training."
        licenses = @(@{ name = "other" })
    }
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage "dataset-metadata.json") -Encoding utf8
    Write-ChainLog "Creating resume dataset nagisa0v0/$slug from $runRoot"
    & $Kaggle datasets create -p $stage -r zip
    if ($LASTEXITCODE -ne 0) {
        throw "Resume dataset creation failed for nagisa0v0/$slug"
    }
    return "nagisa0v0/$slug"
}

function Submit-NextKernel {
    param(
        [int]$SegmentNumber,
        [string]$ResumeDataset
    )
    $slug = "brats-gli-cicek-fp32-segment-{0:D2}" -f $SegmentNumber
    $stage = Join-Path $Root ("kernel_{0:D2}" -f $SegmentNumber)
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item -LiteralPath (Join-Path $RepoRoot "scripts\kaggle_cicek_train.py") -Destination (Join-Path $stage "kaggle_cicek_train.py") -Force
    $metadata = @{
        id = "nagisa0v0/$slug"
        title = $slug
        code_file = "kaggle_cicek_train.py"
        language = "python"
        kernel_type = "script"
        is_private = $true
        enable_gpu = $true
        enable_tpu = $false
        keywords = @()
        dataset_sources = @($GliDataset, $SourceDataset, $ResumeDataset)
        kernel_sources = @()
        competition_sources = @()
        model_sources = @()
        docker_image = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        machine_shape = "NvidiaTeslaT4"
    }
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage "kernel-metadata.json") -Encoding utf8
    Write-ChainLog "Submitting $slug with resume dataset $ResumeDataset"
    & $Kaggle kernels push -p $stage --timeout 39600
    if ($LASTEXITCODE -ne 0) {
        throw "Kernel submission failed for nagisa0v0/$slug"
    }
    return "nagisa0v0/$slug"
}

$kernel = $InitialKernel
$segment = 1
while ($true) {
    Write-ChainLog "Waiting for $kernel"
    do {
        $state = Get-KernelState $kernel
        Write-ChainLog "$kernel state=$state"
        if ($state -match "RUNNING|QUEUED|PENDING|STARTING") {
            Start-Sleep -Seconds $PollSeconds
        }
    } while ($state -match "RUNNING|QUEUED|PENDING|STARTING")

    $outputRoot = Download-KernelOutput $kernel $segment
    $latest = Find-LatestCheckpoint $outputRoot
    if ($null -eq $latest) {
        throw "$kernel ended with state=$state but produced no latest.pt. See $outputRoot"
    }
    $completedEpoch = Get-CompletedEpoch $outputRoot
    Write-ChainLog "$kernel ended with state=$state and completed_epoch=$completedEpoch"
    if ($completedEpoch -ge $FinalEpoch) {
        $finalRoot = Join-Path $Root "final"
        $finalOutput = Join-Path $finalRoot "output"
        New-Item -ItemType Directory -Force -Path $finalRoot | Out-Null
        Copy-Item -LiteralPath $outputRoot -Destination $finalOutput -Recurse -Force
        $finalLatest = Find-LatestCheckpoint $finalOutput
        if ($null -eq $finalLatest) {
            throw "Final Kaggle output lost latest.pt while assembling $finalOutput"
        }
        $finalCheckpointDirectory = $finalLatest.Directory.FullName
        foreach ($segmentDirectory in Get-ChildItem -LiteralPath $Root -Directory -Filter "segment_*") {
            foreach ($periodicCheckpoint in Get-ChildItem -LiteralPath $segmentDirectory.FullName -Recurse -File -Filter "epoch_*.pt") {
                $destination = Join-Path $finalCheckpointDirectory $periodicCheckpoint.Name
                if (-not (Test-Path -LiteralPath $destination)) {
                    Copy-Item -LiteralPath $periodicCheckpoint.FullName -Destination $destination
                }
            }
        }
        Write-ChainLog "Training reached $completedEpoch epochs; final output is $finalRoot"
        break
    }

    $resumeDataset = New-ResumeDataset $outputRoot $segment
    $segment += 1
    $kernel = Submit-NextKernel $segment $resumeDataset
}
