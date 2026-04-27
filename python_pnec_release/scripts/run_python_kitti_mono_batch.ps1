param(
  [string[]]$Sequences = @("00","04","07"),
  [string]$DatasetRoot = "C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences",
  [string]$OutputRoot = "C:\Users\me\Nav\pnec\python_runs_mono",
  [string]$CondaEnv = "InEKF",
  [switch]$NecOnly,
  [switch]$PnecOnly,
  [string]$PnecVariant = "symmetric",
  [string]$PoseConvention = "target-to-host",
  [double]$MaxMonoRotationDeg = 5.0,
  [int]$MaxFrames = 0,
  [switch]$Diagnostics
)

$normalizedSequences = @()
foreach ($entry in $Sequences) {
  foreach ($seq in ($entry -split ",")) {
    $trimmed = $seq.Trim()
    if ($trimmed.Length -gt 0) {
      $normalizedSequences += $trimmed.PadLeft(2, "0")
    }
  }
}

$commonArgs = @()
if ($MaxFrames -gt 0) {
  $commonArgs += @("--max-frames", "$MaxFrames")
}
if ($Diagnostics) {
  $commonArgs += @("--diagnostics")
}

foreach ($seq in $normalizedSequences) {
  if (-not $PnecOnly) {
    Write-Host "Running Python mono NEC $seq ..."
    conda run -n $CondaEnv python run_mono_nec.py `
      --dataset-root $DatasetRoot `
      --sequence $seq `
      --output-root $OutputRoot `
      --pose-convention $PoseConvention `
      --max-mono-rotation-deg $MaxMonoRotationDeg `
      @commonArgs
  }

  if (-not $NecOnly) {
    Write-Host "Running Python mono PNEC $seq ($PnecVariant) ..."
    conda run -n $CondaEnv python run_mono_pnec.py `
      --dataset-root $DatasetRoot `
      --sequence $seq `
      --output-root $OutputRoot `
      --variant $PnecVariant `
      --pose-convention $PoseConvention `
      --max-mono-rotation-deg $MaxMonoRotationDeg `
      @commonArgs
  }
}
