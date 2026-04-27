# Python PNEC/NEC Release

This folder contains the standalone Python migration of the current NEC/PNEC KITTI odometry pipelines.

It is intended for reproducible experiments and sharing. It is not a byte-for-byte or frame-by-frame clone of the C++/Docker implementation. The Python version keeps the same output format and evaluation protocol, but includes Python-side robustness changes, especially for stereo 3D rigid fallback and monocular rotation rejection.

## Contents

- `python_pnec/`: shared Python package for IO, geometry, frontend, solvers, and pipelines.
- `run_mono_nec.py`: run monocular NEC.
- `run_mono_pnec.py`: run monocular PNEC.
- `run_stereo_nec.py`: run stereo NEC.
- `run_stereo_pnec.py`: run stereo PNEC.
- `scripts/run_python_kitti_mono_batch.ps1`: batch runner for monocular NEC/PNEC.
- `scripts/run_python_kitti_stereo_batch.ps1`: batch runner for stereo NEC/PNEC.
- `scripts/evaluate_kitti_pnec_experiments.py`: evaluate result folders and draw trajectories.
- `scripts/summarize_python_mono_representative.py`: summarize mono Python/C++ representative results.
- `scripts/summarize_python_stereo_representative.py`: summarize stereo Python/C++ representative results.
- `python_pnec_requirements.txt`: Python dependencies.

## Environment

The commands below assume the conda environment is named `InEKF`.

Install dependencies if needed:

```powershell
conda run -n InEKF python -m pip install -r python_pnec_requirements.txt
```

Required packages:

```text
opencv-python
scipy
matplotlib
numpy
```

## Dataset Layout

Default KITTI odometry gray root:

```powershell
C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences
```

Each sequence should look like:

```text
sequences\00\
  image_0\
  image_1\
  calib.txt
  times.txt
  poses.txt
```

## Output Format

Each run writes a result folder under the selected output root:

```text
XX_py_mono_nec\
XX_py_mono_pnec_symmetric\
XX_py_mono_pnec_target\
XX_py_stereo_nec\
XX_py_stereo_pnec\
```

Each result folder contains:

```text
rot_avg\poses.txt
timing.txt
diagnostics.csv    # only when --diagnostics is enabled
```

`poses.txt` uses the same format as the C++ pipeline:

```text
timestamp tx ty tz qx qy qz qw
```

## Recommended Defaults

Monocular:

- PNEC variant: `symmetric`
- RANSAC threshold: `1e-2`
- RANSAC iterations: `5000`
- Max monocular rotation rejection: `5 deg`
- Pose convention: `target-to-host`

Stereo:

- PNEC variant: `symmetric`
- Uses current Python stereo 3D rigid fallback for gross relative-pose failures.

## Single-Sequence Commands

Run monocular NEC:

```powershell
conda run -n InEKF python run_mono_nec.py `
  --dataset-root C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences `
  --sequence 07 `
  --output-root C:\Users\me\Nav\pnec\python_runs_mono `
  --diagnostics
```

Run monocular PNEC:

```powershell
conda run -n InEKF python run_mono_pnec.py `
  --dataset-root C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences `
  --sequence 07 `
  --output-root C:\Users\me\Nav\pnec\python_runs_mono `
  --variant symmetric `
  --diagnostics
```

Run stereo NEC:

```powershell
conda run -n InEKF python run_stereo_nec.py `
  --dataset-root C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences `
  --sequence 07 `
  --output-root C:\Users\me\Nav\pnec\python_runs_full
```

Run stereo PNEC:

```powershell
conda run -n InEKF python run_stereo_pnec.py `
  --dataset-root C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences `
  --sequence 07 `
  --output-root C:\Users\me\Nav\pnec\python_runs_full `
  --variant symmetric
```

## Batch Commands

Representative monocular runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python_kitti_mono_batch.ps1 `
  -Sequences 00,04,07 `
  -PnecVariant symmetric `
  -Diagnostics `
  -OutputRoot C:\Users\me\Nav\pnec\python_runs_mono
```

Full monocular runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python_kitti_mono_batch.ps1 `
  -Sequences 00,01,02,03,04,05,06,07,08,09,10 `
  -PnecVariant symmetric `
  -Diagnostics `
  -OutputRoot C:\Users\me\Nav\pnec\python_runs_mono
```

Representative stereo runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python_kitti_stereo_batch.ps1 `
  -Sequences 00,03,05,07,09 `
  -OutputRoot C:\Users\me\Nav\pnec\python_runs_full
```

Full stereo runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_python_kitti_stereo_batch.ps1 `
  -Sequences 00,01,02,03,04,05,06,07,08,09,10 `
  -OutputRoot C:\Users\me\Nav\pnec\python_runs_full
```

## Evaluation

Evaluate one sequence with multiple result folders:

```powershell
conda run -n InEKF python scripts\evaluate_kitti_pnec_experiments.py `
  --sequence 07 `
  --results-root C:\Users\me\Nav\pnec\python_runs_mono `
  --gt-root C:\Users\me\Nav\kitti\data_odometry_gray\dataset\sequences `
  --folders 07_py_mono_nec 07_py_mono_pnec_symmetric `
  --output-dir C:\Users\me\Nav\pnec\python_eval_mono_representative\07
```

The evaluator writes:

```text
XX_pnec_experiments_summary.csv
plots\*_vs_gt.png
plots\XX_experiments_overlay.png
```

## Notes

- KITTI `01` is known to be difficult for the KLT frontend and should be marked separately in summaries.
- Monocular translation has no metric scale. Use `RPE1/RPEn` as the main monocular metrics; `t_rel/ATE` are Sim3-aligned references.
- Stereo results are the primary source for metric trajectory quality.
- The Python version is a migration/enhanced implementation, not a strict C++ numerical clone.
