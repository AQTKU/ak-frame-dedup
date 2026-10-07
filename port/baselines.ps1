# Fresh v1.2.0 (Python) baseline runs for the Rust port's end-to-end check.
# One at a time, 8 workers, intermediates kept. Usage: pwsh port/baselines.ps1 [names...]
param([string[]]$Names = @('sample', 'clip', 'smoke', 'foliage'))
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$out = 'work\port\baseline'
New-Item -ItemType Directory -Force $out | Out-Null
$inputs = @{ sample = 'sample.mkv'; foliage = 'foliage.mkv'; smoke = 'smoke.mkv'; clip = 'work\mmco\clip.mkv'; nodupes = 'nodupes.mkv' }
foreach ($n in $Names) {
    $t = Get-Date
    python -m frame_dedup $inputs[$n] -o "$out\$n.py.mkv" --report "$out\$n.py.json" --verify --workers 8 --work "$out\$n.py.work" --keep-work *> "$out\$n.py.log"
    "{0}: exit {1}, {2:n0} s" -f $n, $LASTEXITCODE, ((Get-Date) - $t).TotalSeconds | Tee-Object -Append "$out\status.txt"
}
