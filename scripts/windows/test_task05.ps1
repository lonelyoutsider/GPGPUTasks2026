param(
    [string]$Executable = "$PSScriptRoot/../../build/Release/main_radix_sort.exe",
    [int]$Device = 0,
    [int[]]$Sizes = @(10, 256, 257, 600, 65537, 100000000),
    [string[]]$Patterns = @('random'),
    [string[]]$Scans = @('baseline', 'hierarchical'),
    [string[]]$LocalCounters = @('baseline'),
    [string]$OutputDirectory = "$PSScriptRoot/../../build/task05-results"
)

$ErrorActionPreference = 'Stop'
$Executable = (Resolve-Path -LiteralPath $Executable).Path
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$rows = @()
$oldSize = $env:TASK05_SIZE
$oldPattern = $env:TASK05_PATTERN
$oldScan = $env:TASK05_SCAN
$oldLocal = $env:TASK05_LOCAL
try {
    foreach ($pattern in $Patterns) {
        foreach ($n in $Sizes) {
            foreach ($scan in $Scans) {
                foreach ($local in $LocalCounters) {
                    $env:TASK05_SIZE = "$n"
                    $env:TASK05_PATTERN = $pattern
                    $env:TASK05_SCAN = $scan
                    $env:TASK05_LOCAL = $local
                    $log = Join-Path $OutputDirectory "$scan-$local-$pattern-$n.log"
                    Write-Host "Running scan=$scan local=$local pattern=$pattern n=$n"
                    $output = & $Executable $Device 2>&1
                    $code = $LASTEXITCODE
                    $output | Set-Content -LiteralPath $log
                    if ($code -ne 0 -or !($output -match 'PASS: std::sort and unchanged input_gpu')) {
                        throw "Task05 failed: $log (exit $code)"
                    }
                    if (!($output -match "scan=$scan local=$local")) {
                        throw "Executable did not confirm requested algorithms: $log"
                    }
                    $match = [regex]::Match(($output -join "`n"), 'median_ms=([0-9.eE+-]+) measurements=10')
                    if (!$match.Success) { throw "Missing ten measurements: $log" }
                    $rows += [pscustomobject]@{
                        scan = $scan; local = $local; pattern = $pattern; n = $n
                        median_ms = [double]::Parse($match.Groups[1].Value, [cultureinfo]::InvariantCulture)
                    }
                    $rows | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $OutputDirectory 'results.csv')
                    Write-Host "PASS median_ms=$($match.Groups[1].Value)"
                }
            }
        }
    }
} finally {
    $env:TASK05_SIZE = $oldSize
    $env:TASK05_PATTERN = $oldPattern
    $env:TASK05_SCAN = $oldScan
    $env:TASK05_LOCAL = $oldLocal
}
$rows | Format-Table -AutoSize
