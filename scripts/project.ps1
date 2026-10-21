[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [ValidateSet('check', 'teacher', 'test', 'lint', 'experiments', 'adapt')]
    [string]$Action = 'check',
    [string]$Python = 'D:\Environment\miniforge3\envs\cv311\python.exe',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python interpreter not found: $Python" }
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Join-Path $projectRoot 'src')
    if ($previousPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $previousPythonPath }
    $pythonArgs = switch ($Action) {
        'check' { @('-m', 'brats_tta.cli.project', '--root', $projectRoot, '--check') }
        'teacher' { @('-m', 'brats_tta.cli.context_teacher') }
        'experiments' { @('-m', 'brats_tta.cli.experiments', '--root', $projectRoot) }
        'adapt' { @('-m', 'brats_tta.cli.adapt_context', '--root', $projectRoot) }
        'lint' { @('-m', 'ruff', 'check', '--select', 'E9,F63,F7,F82', (Join-Path $projectRoot 'src'), (Join-Path $projectRoot 'tests')) }
        'test' {
            @('-m', 'pytest', '-q') + @(
                'tests/test_native_context.py',
                'tests/test_native_adaptation.py',
                'tests/test_native_efficiency.py',
                'tests/test_transition_projection.py',
                'tests/test_project_paths.py',
                'tests/test_experiments.py',
                'tests/test_context_pipeline.py'
            ).ForEach({ Join-Path $projectRoot $_ })
        }
    }
    & $Python -X utf8 @pythonArgs @Arguments
    $result = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
exit $result
