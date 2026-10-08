param([string]$Title = "Select a folder", [string]$Initial = "")
# Pure ASCII on purpose: PowerShell reads .ps1 as ANSI(GBK) on zh-CN Windows,
# so any non-ASCII byte here would break parsing.
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = $Title
$dlg.ShowNewFolderButton = $true
if ($Initial -and (Test-Path -LiteralPath $Initial)) {
    try { $dlg.SelectedPath = (Resolve-Path -LiteralPath $Initial).Path } catch { }
}
if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dlg.SelectedPath
}
