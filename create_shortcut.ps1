# Legt eine Verknüpfung "Game of Life" mit App-Icon auf dem Desktop an.
# Aufruf:  powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
# Optional: -StartMenu  legt sie zusätzlich im Startmenü an (lässt sich dann an die Taskleiste heften).
param([switch]$StartMenu)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    Write-Error "Environment fehlt: $pythonw. Zuerst 'python -m venv .venv' und 'pip install -r requirements.txt' ausführen."
    exit 1
}

$targets = @([Environment]::GetFolderPath("Desktop"))
if ($StartMenu) { $targets += [Environment]::GetFolderPath("Programs") }

$shell = New-Object -ComObject WScript.Shell
foreach ($dir in $targets) {
    $link = $shell.CreateShortcut((Join-Path $dir "Game of Life.lnk"))
    $link.TargetPath = $pythonw
    $link.Arguments = '"' + (Join-Path $root "main.py") + '"'
    $link.WorkingDirectory = $root
    $link.IconLocation = (Join-Path $root "assets\icon.ico") + ",0"
    $link.Description = "Conway's Game of Life"
    $link.Save()
    Write-Host "Verknüpfung angelegt: $(Join-Path $dir 'Game of Life.lnk')"
}
