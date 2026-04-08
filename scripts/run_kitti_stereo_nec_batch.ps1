$seqs = "01","02","03","04","05","06","07","08","09","10"

$resultsRoot = "C:\Users\me\Nav\kitti\pnec_results"
$datasetRoot = "C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences"
$imageName = "pnec:latest"

foreach ($seq in $seqs) {
  if ($seq -in @("00","01","02")) {
    $cam = "/app/pnec/data/config_kitti00-02.yaml"
  } elseif ($seq -eq "03") {
    $cam = "/app/pnec/data/config_kitti03.yaml"
  } else {
    $cam = "/app/pnec/data/config_kitti04-10.yaml"
  }

  $resultDir = "${seq}_stereo_nec"
  Write-Host "Running stereo NEC for sequence $seq ..."

  docker run -it --rm `
    --name "stereo_nec_$seq" `
    --mount "type=bind,source=$datasetRoot,target=/home/sequences,readonly" `
    --mount "type=bind,source=$resultsRoot,target=/home/results" `
    $imageName `
    /bin/sh -c "mkdir -p /home/results/$resultDir /home/results/$resultDir/vis && cp -f /home/sequences/$seq/poses.txt /home/results/$resultDir/poses.txt && ./build/stereo_nec_vo $cam /app/pnec/data/test_config_nec.yaml /app/pnec/data/tracking/KITTI/kitti_calib.json /app/pnec/data/tracking/KITTI/$seq.json /home/sequences/$seq/image_0 /home/sequences/$seq/image_1 /home/sequences/$seq/times.txt /home/sequences/$seq/calib.txt /home/results/$resultDir/ /home/results/$resultDir/vis/ true"
}
