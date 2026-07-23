[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Config = "experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/config/smoke.json",
    [string]$OutputRoot = "build/poc-local-validation"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ResolvedOutput = Join-Path $RepositoryRoot $OutputRoot
$Generated = Join-Path $ResolvedOutput "generated"

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $RepositoryRoot
try {
    $env:PYTHONPATH = Join-Path $RepositoryRoot "src"
    Invoke-Checked { & $Python -m unittest discover -s tests -v }
    Invoke-Checked { & $Python -m wormctx.poc doctor --config $Config }

    if (Test-Path -LiteralPath $Generated) {
        throw "Generation target already exists; choose a fresh -OutputRoot: $Generated"
    }
    Invoke-Checked {
        & $Python -m wormctx.poc generate --config $Config --output $Generated
    }
    foreach ($Split in @("train", "val", "test")) {
        Invoke-Checked {
            & $Python -m wormctx.poc validate `
                --input (Join-Path $Generated "$Split.jsonl")
        }
    }
    Write-Host "Local contracts, tests, deterministic generation, and guards passed."
    Write-Host "Generated evidence: $Generated"
}
finally {
    Pop-Location
}
