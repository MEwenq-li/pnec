param(
  [string]$Sequence = "07",
  [string]$ResultsRoot = "C:\Users\me\Nav\kitti\pnec_results",
  [string]$DatasetRoot = "C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences",
  [string]$ImageName = "pnec:latest",
  [string[]]$Experiments = @()
)

function Set-YamlScalar {
  param(
    [string]$Content,
    [string]$Key,
    [string]$Value
  )

  $escapedKey = [regex]::Escape($Key)
  $pattern = "(?m)^${escapedKey}:.*$"
  $replacement = "${Key}: ${Value}"
  return [regex]::Replace($Content, $pattern, $replacement)
}

if ($Sequence -in @("00", "01", "02")) {
  $cameraConfig = "/app/pnec/data/config_kitti00-02.yaml"
} elseif ($Sequence -eq "03") {
  $cameraConfig = "/app/pnec/data/config_kitti03.yaml"
} else {
  $cameraConfig = "/app/pnec/data/config_kitti04-10.yaml"
}

$baseConfigPath = Join-Path $PSScriptRoot "..\data\test_config.yaml"
$baseConfig = Get-Content -Path $baseConfigPath -Raw
$tempConfigDir = Join-Path $ResultsRoot "_temp_configs"
New-Item -ItemType Directory -Force -Path $tempConfigDir | Out-Null

$normalizedExperiments = @()
foreach ($item in $Experiments) {
  if ($null -eq $item) {
    continue
  }
  $normalizedExperiments += ($item -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}
$Experiments = $normalizedExperiments

$experimentMatrix = @(
  @{
    Name = "${Sequence}_nec_ref"
    Settings = @{
      "PNEC.NEC" = "1"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_base"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_w1_noceres"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "1"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "0"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_w10_noceres"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "0"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_w10_noceres_noscf"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "0"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "0"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_w10_ceres_noscf"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "0"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_isotropic"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Isotropic"'
      "PNEC.isotropicCovarianceValue" = "1.0e-6"
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_diagonal"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Diagonal"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_normalized"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Normalized"'
      "PNEC.normalizedCovarianceTrace" = "1.0"
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_freezerot"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"FreezeRotation"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_paperlike"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"PaperLike"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_paperlike_noceres"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "0"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"PaperLike"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_diagstats"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"Weighted"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "1"
    }
  },
  @{
    Name = "${Sequence}_pnec_necinit"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"NEC"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_necceresinit"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"NECCeres"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  },
  @{
    Name = "${Sequence}_pnec_sym_necceresinit"
    Settings = @{
      "PNEC.NEC" = "0"
      "PNEC.noiseFrame" = '"Both"'
      "PNEC.weightedIterations" = "10"
      "PNEC.SCF" = "1"
      "PNEC.ceresInitMode" = '"NECCeres"'
      "PNEC.ceres" = "1"
      "PNEC.covarianceMode" = '"Original"'
      "PNEC.weightedRotationUpdateMode" = '"ScaledBearing"'
      "PNEC.dumpCovarianceStats" = "0"
    }
  }
)

$selectedExperiments = @($experimentMatrix | Where-Object {
  $Experiments.Count -eq 0 -or ($Experiments -contains $_.Name)
})

if ($selectedExperiments.Count -eq 0) {
  $available = $experimentMatrix | ForEach-Object { $_.Name }
  throw "No experiments matched. Requested: $($Experiments -join ', '). Available: $($available -join ', ')"
}

foreach ($experiment in $selectedExperiments) {
  $resultDir = $experiment.Name
  $hostConfigPath = Join-Path $tempConfigDir "$resultDir.yaml"
  $configContent = $baseConfig

  foreach ($entry in $experiment.Settings.GetEnumerator()) {
    $configContent = Set-YamlScalar -Content $configContent -Key $entry.Key -Value $entry.Value
  }

  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($hostConfigPath, $configContent, $utf8NoBom)

  Write-Host "Running $resultDir ..."

  docker run -it --rm `
    --name "pnec_$resultDir" `
    --mount "type=bind,source=$DatasetRoot,target=/home/sequences,readonly" `
    --mount "type=bind,source=$ResultsRoot,target=/home/results" `
    $ImageName `
    /bin/sh -c "mkdir -p /home/results/$resultDir /home/results/$resultDir/vis && cp -f /home/sequences/$Sequence/poses.txt /home/results/$resultDir/poses.txt && ./build/pnec_vo $cameraConfig /home/results/_temp_configs/$resultDir.yaml /app/pnec/data/tracking/KITTI/kitti_calib.json /app/pnec/data/tracking/KITTI/$Sequence.json /home/sequences/$Sequence/image_0 /home/sequences/$Sequence/times.txt /home/results/$resultDir/ /home/results/$resultDir/vis/ true"
}
