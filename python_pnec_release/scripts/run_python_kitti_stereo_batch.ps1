param(
  [string[]]$Sequences = @("00","03","05","07","09"),
  [string]$DatasetRoot = "C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences",
  [string]$OutputRoot = "C:\Users\me\Nav\pnec\python_runs_full",
  [string]$CondaEnv = "InEKF",
  [switch]$NecOnly,
  [switch]$PnecOnly,
  [int]$MaxFrames = 0
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

$maxFrameArgs = @()
if ($MaxFrames -gt 0) {
  $maxFrameArgs = @("--max-frames", "$MaxFrames")
}

foreach ($seq in $normalizedSequences) {
  if (-not $PnecOnly) {
    Write-Host "Running Python stereo NEC $seq ..."
    conda run -n $CondaEnv python run_stereo_nec.py `
      --dataset-root $DatasetRoot `
      --sequence $seq `
      --output-root $OutputRoot `
      @maxFrameArgs
  }

  if (-not $NecOnly) {
    Write-Host "Running Python stereo PNEC $seq ..."
    conda run -n $CondaEnv python run_stereo_pnec.py `
      --dataset-root $DatasetRoot `
      --sequence $seq `
      --output-root $OutputRoot `
      @maxFrameArgs
  }
}
