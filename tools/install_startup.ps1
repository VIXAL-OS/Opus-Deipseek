# Install/update Hydra's per-user logon task without storing a password.
# Matching Ressapanda: run one minute after sign-in, including on battery power.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot 'launch_bot.ps1'
$taskName = 'HydraOpusDeipseekAutostart'
$userName = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$powerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) { throw "Missing launcher: $launcher" }
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $launcher + '"'
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $userName
$trigger.Delay = 'PT1M'
$principal = New-ScheduledTaskPrincipal -UserId $userName -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description 'Starts Hydra (Opus-Deipseek) one minute after Windows sign-in. Hidden, single-instance guarded, with UTF-8 logs in the repository logs folder.'

# Refuse an unrelated task with a colliding name; updating this installation is safe.
$existing = Get-ScheduledTask -TaskName $taskName -TaskPath '\' -ErrorAction SilentlyContinue
if ($existing -and ($existing.Actions.Count -ne 1 -or $existing.Actions[0].Arguments -notlike ('*"' + $launcher + '"*'))) {
    throw "Task $taskName already exists with a different action; refusing to overwrite it."
}
Register-ScheduledTask -TaskName $taskName -TaskPath '\' -InputObject $task -Force | Out-Null
Write-Output "Installed $taskName for $userName (one minute after sign-in)."
