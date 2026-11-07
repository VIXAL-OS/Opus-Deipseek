# Start Hydra detached with UTF-8 logs and a single-instance guard.
# Used by the HydraOpusDeipseekAutostart logon task; also safe to run manually.
# -CheckOnly validates prerequisites and checks for an existing bot without starting it.
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$botFile = Join-Path $repo 'bot.py'
foreach ($required in @($python, $botFile, (Join-Path $repo '.env'), (Join-Path $repo 'config.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing startup prerequisite: $required"
    }
}

# Serialize the check-and-start so two simultaneous launcher invocations cannot race.
$mutex = New-Object System.Threading.Mutex($false, 'Local\HydraOpusDeipseekLauncher')
$locked = $false
try {
    try { $locked = $mutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $locked = $true }
    if (-not $locked) {
        Write-Output 'Hydra launcher is already active; no second launch.'
        exit 0
    }

    $pythonPattern = [regex]::Escape($python)
    $botPattern = [regex]::Escape($botFile)
    $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object {
            $_.CommandLine -match $botPattern -or
            (($_.CommandLine -match $pythonPattern -or $_.ExecutablePath -eq $python) -and
             $_.CommandLine -match '(?i)(?:^|[\s"\\/])bot\.py(?:[\s"]|$)')
        })
    if ($running.Count -gt 0) {
        Write-Output ('Hydra is already running (PID ' + (($running | ForEach-Object { $_.ProcessId }) -join ', ') + '); no second launch.')
        exit 0
    }
    if ($CheckOnly) {
        Write-Output 'Ready: prerequisites present; no Hydra instance detected. CheckOnly did not start the bot.'
        exit 0
    }

    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $logDir = Join-Path $repo 'logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stdout = Join-Path $logDir "boot_${stamp}_stdout.log"
    $stderr = Join-Path $logDir "boot_${stamp}_stderr.log"
    $process = Start-Process -FilePath $python -ArgumentList ('"' + $botFile + '"') `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Output "Started Hydra (PID $($process.Id)); logs: logs/boot_${stamp}_*.log"
}
finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
