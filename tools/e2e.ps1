# End-to-end acceptance runs of the Rust build against the reference outputs.
# Usage: pwsh tools/e2e.ps1 [names...]   (one at a time; per-stream MD5 compared)
param([string[]]$Names = @('smoke', 'foliage', 'full'))
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$out = 'work\port\e2e'
New-Item -ItemType Directory -Force $out | Out-Null
$inputs = @{ sample = 'sample.mkv'; foliage = 'foliage.mkv'; smoke = 'smoke.mkv'; clip = 'work\mmco\clip.mkv'
             full = 'mmco-or-ref-list-modification.mkv' }
# 2.2 references: sample and foliage are unchanged since 1.2.0; smoke, clip and full changed
# with residual corrections (2.1) and the pooled budget (2.2); the 2.0 outputs are what
# --no-corrections gives
$base = @{ sample = 'work\port\baseline\sample.py.mkv'; foliage = 'work\port\baseline\foliage.py.mkv'
           smoke = 'work\v22\final\smoke.mkv'; clip = 'work\v22\final\clip.mkv'; full = 'work\v22\final\full.mkv' }
$workers = @{ full = 16 }
function streamhash($f) { (ffmpeg -v error -i $f -map 0 -c copy -f streamhash -hash md5 - 2>$null) -join ';' }
foreach ($n in $Names) {
    $w = if ($workers[$n]) { $workers[$n] } else { 8 }
    $t = Get-Date
    & target\release\frame-dedup.exe $inputs[$n] -o "$out\$n.rs.mkv" --report "$out\$n.rs.json" --verify `
        --mb-report "$out\$n.rs.mb.json" --workers $w --work "$out\$n.rs.work" *> "$out\$n.rs.log"
    $code = $LASTEXITCODE
    $secs = ((Get-Date) - $t).TotalSeconds
    $a = streamhash "$out\$n.rs.mkv"
    $b = streamhash $base[$n]
    $same = if ($a -eq $b -and $a) { 'IDENTICAL' } else { "DIFFERENT`n  rust: $a`n  base: $b" }
    "{0}: exit {1}, {2:n0} s, {3} workers; streams {4}" -f $n, $code, $secs, $w, $same | Tee-Object -Append "$out\status.txt"
}
