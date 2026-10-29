# Oracle (prior-informed) baselines: how much does knowing the scene help?

Date of the runs: 2026-10-12 to 2026-10-14 (log written 2026-10-29 from the archived runs and `docs/oracle_baselines.md` / `docs/oracle_results.md`; nothing was rerun).

## Question
Only the *viewpoint selection* is changed; the SplaTAM mapper stays online and never sees the ground-truth (GT) geometry. So any change in coverage, image quality or path length is the value of prior knowledge about the scene. How far above ActiveSplat's native planner can oracle planners get, and does more coverage give better maps?

## Setup
- Scenes (Matterport3D, `config/datasets/mp3d_high_resolution*.json`, 512² sensor, 10 mapping iterations): gZ6f7yhEvPG, GdvgFV5R1Z5, pLe4wQe7qrG, YmJkqBEsHnH. 1000 steps each, equal budget for every method.
- Methods (one label per run in `<run>/label.txt`):
  - **random walk**: `scripts/oracle/random_walk.py` (seed fixed), replayed. Lower bound.
  - **native**: ActiveSplat's planner. Repeated with `ACTIVESPLAT_SEED` (3 runs per scene, 2 on pLe4wQe7qrG) to measure run-to-run noise.
  - **oracle-info** (Tier 2): `planner.oracle.mode = info`. The mapper's look-around invisibility (opacity + depth) is replaced by GT-mesh raycasts against the set of already-observed surface voxels (5 cm). Same candidates, travel cost and rotation selection as native; only "what is still unseen" is known.
  - **tour** (budgeted offline tour, `scripts/oracle/plan_tour.py --max_steps 1000`): greedy set cover over candidate views (grid x yaw x pitch -30/0/30) on the GT navmesh, greedy insertion by new surface per added action, 2-opt, converted to discrete actions and replayed (`mode:=REPLAY`). Two orderings: distance order (`--no_heading_order`) and heading-aware.
  - **NBV** (Tier 3, `scripts/oracle/plan_nbv.py --max_steps 1000`): closed-loop greedy next-best-view with free candidate poses on the GT navmesh, replayed. `NBV w=1` / `w=3`: `--view_weight`, a second sighting from a direction at least 20° from the first is worth w new samples.
- Coverage (`scripts/oracle/coverage.py`, `src/oracle/`): share of the *observable* GT surface seen by the sensor (reachable from the start floor's navmesh island, within the 4 m mapper range and tilt limits), per step. "two directions" = share seen from two directions at least 20° apart. AUC = mean of the coverage curve over 1000 steps. Image metrics (`src/evaluation/image_eval.py`): PSNR / SSIM / LPIPS on trajectory views, and on 100 held-out "novel" views within 1.5 m of the trajectory.
- Launch pattern per scene (see `docs/oracle_baselines.md` for the design): `python scripts/oracle/precompute_observable.py --config <cfg> --user_config config/user_config.json --scene_id <scene>`, then `plan_tour.py` / `plan_nbv.py` / `random_walk.py` write an `actions.txt`, which is replayed through the normal launch (`mode:=REPLAY`); native and oracle-info are normal launches with the respective config. The exact roslaunch lines were not recorded at the time: **not reconstructible from the run directories**, only config, actions and labels are.
- Comparison: `python scripts/oracle/compare.py --results_dir results/archive --steps 1000 --out docs/oracle_results.md --plots_dir docs/oracle_plots`. The runs live in `results/archive/` (`MANIFEST.txt`).
- Metadata of every run (git commit, config, optimisations, image metrics): appendix below and `runs.json`, produced by `scripts/tools/experiment_log.py`. Runs are named `<method>_<scene prefix>_<MM-DD_HH-MM>`. All 39 runs predate the optimisation switches and the span tracer ("baseline" code), run on shared GPUs, one at a time or two concurrently. The git commit in the metadata is the one *at log time*, not the one the runs were made with (use the dates: code of 2026-10-12 to 14, commits `62f3ab3` to `9f6a67a`).

## Results (1000 steps, mean over the four scenes)

± is the spread between scenes, not noise. "steps to 80 %" averages only the scenes that reach it.

| method | cov@100 | cov@300 | cov@500 | cov@1000 | AUC | steps to 80 % | path (m) | two directions % | PSNR | novel PSNR | novel SSIM | novel LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random walk | 29.3 | 47.1 | 53.6 | 57.6 | 47.7 | – | 42.3 | 39.4 | 21.3 | 21.5 | 0.762 | 0.263 |
| native | 32.6 | 51.4 | 64.4 | 83.4 | 59.0 | 625 ± 227 | 21.8 | 58.5 | 21.0 | 19.9 | 0.705 | 0.303 |
| oracle-info | 32.2 | 52.6 | 68.9 | 88.9 | 62.3 | 695 ± 190 | 22.6 | 63.1 | 21.8 | 20.6 | 0.718 | 0.302 |
| tour (distance order) | 38.1 | 56.3 | 62.0 | 90.8 | 63.6 | 744 ± 166 | 14.8 | 68.8 | 22.0 | 21.3 | 0.724 | 0.304 |
| tour (heading order) | 38.4 | 55.7 | 62.9 | 92.6 | 64.5 | 720 ± 121 | 22.2 | 70.2 | 21.4 | 20.6 | 0.705 | 0.320 |
| NBV | 44.0 | 74.7 | 85.2 | 95.3 | 77.2 | 395 ± 134 | 29.4 | 74.2 | 19.9 | 19.4 | 0.668 | 0.346 |
| NBV w=1 | 43.4 | 66.5 | 82.0 | 94.2 | 73.3 | 469 ± 187 | 24.3 | 79.0 | 20.6 | 19.9 | 0.692 | 0.331 |
| NBV w=3 | 40.1 | 61.3 | 69.6 | 91.7 | 65.5 | 678 ± 279 | 26.2 | 76.7 | 21.2 | 20.7 | 0.699 | 0.334 |

![coverage, mean over scenes](coverage_all_scenes.jpg)

Per scene (coverage at 1000 steps / AUC; native = mean of 2 to 3 seeds ± s.d., the others single runs):

| method | gZ6f7yhEvPG | GdvgFV5R1Z5 | pLe4wQe7qrG | YmJkqBEsHnH |
|---|---|---|---|---|
| random walk | 65.1 / 51.7 | 78.5 / 67.5 | 33.4 / 28.3 | 53.5 / 43.1 |
| native | 92.4 ± 1.7 / 71.5 ± 0.9 | 89.0 ± 2.7 / 72.8 ± 0.7 | 82.9 ± 0.3 / 45.2 ± 0.4 | 69.4 ± 6.2 / 46.6 ± 0.4 |
| oracle-info | 95.7 / 71.2 | 92.9 / 74.0 | 85.4 / 54.4 | 81.7 / 49.7 |
| tour (distance) | 98.2 / 75.0 | 94.5 / 71.3 | 90.0 / 58.8 | 80.7 / 49.1 |
| tour (heading) | 99.0 / 71.9 | 95.2 / 73.4 | 91.1 / 59.9 | 85.1 / 52.8 |
| NBV | 98.8 / 84.2 | 96.6 / 83.7 | 92.7 / 71.2 | 93.0 / 70.0 |
| NBV w=1 | 98.2 / 83.8 | 96.5 / 80.8 | 89.3 / 66.3 | 93.1 / 62.4 |
| NBV w=3 | 97.6 / 81.9 | 95.5 / 74.0 | 90.8 / 52.7 | 83.1 / 53.3 |

Per-scene curves: `coverage_<scene>.jpg`. Full per-scene tables (all columns incl. SSIM/LPIPS, steps to 80/90 %): `docs/oracle_results.md`.

Trajectory image quality per scene (PSNR, native = seed mean): gZ6f7yhEvPG 17.6 native / 17.8 oracle-info / 17.7 tour / 17.1 NBV; GdvgFV5R1Z5 24.5 / 24.2 / 24.7 / 23.6; pLe4wQe7qrG 26.1 / 25.6 / 25.1 / 24.9; YmJkqBEsHnH 15.7 / 19.7 / 20.7 / 14.1.

## Reading
- **Knowing the scene helps exploration, and how it is used matters more than whether it is used.** Tier 3 (NBV: free poses, GT navmesh, myopic closed loop) is best on every scene: AUC 77 against 59 for native, 80 % coverage in about 400 steps against about 625, 95 % at 1000 steps against 83 %. It also walks more (29 m against 22 m).
- Oracle-info, which only knows what is unseen but keeps the native candidates and hierarchy, is a modest gain (AUC +3, coverage at 1000 steps +5.5), mostly on the scenes where native stalls (pLe4wQe7qrG, YmJkqBEsHnH). On gZ6f7yhEvPG and GdvgFV5R1Z5 it is within the native seed noise. The bottleneck is therefore the planner's candidate set and tour structure, not the unseen-geometry estimate.
- The budgeted tours reach high final coverage with a short path (distance order: 14.8 m) but are slow early (sweeps in route order); their curves cross native's. The heading-aware order cuts turns (62 % to 49 % on gZ6f7yhEvPG) and adds about 2 points of final coverage for 7 m more path.
- **Coverage does not translate into image quality.** Trajectory and novel-view PSNR/SSIM/LPIPS are the same or worse for NBV (19.9 PSNR against 21.0 native) and best for the random walk, which sees the least. These views are near where the agent has been, so they reward re-looking at mapped places; a first sighting does not map a surface well. Read image metrics as quality of the mapped part, beside coverage. YmJkqBEsHnH is the extreme: NBV 93 % coverage with PSNR 14.1, native 69 % with 15.7, tour/oracle-info about 20.
- Second-look weighting is a trade: w=1 gives up about 1 point of final coverage and 4 of AUC for +5 points of two-direction coverage and slightly better image metrics; w=3 gives up too much exploration.

## Caveats and known problems
- **Noise.** Native seeds differ by 0.3 to 6 points of coverage at 1000 steps depending on the scene (PSNR 0.1 to 0.5), identical for the first 100 steps and then diverging (native-planner runs are not bit-reproducible). All non-native methods are **one run per scene**: differences of a few points between them are not conclusive. Scenes (4) are few; the ± in the mean table is between-scene spread.
- pLe4wQe7qrG has only 2 native runs (the third hung at launch and was never rerun).
- The three tour runs on YmJkqBEsHnH, GdvgFV5R1Z5 (heading) and the distance tour on gZ6f7yhEvPG executed 982 to 996 actions instead of 1000 (replay cut); the comparison script truncates at 1000.
- The tour/NBV planners are heuristics, not upper bounds, and use GT navmesh and GT mesh visibility; "observable" is defined by the oracle's own range/tilt model (4 m, the mapper's far plane), so coverage is relative to it.
- The earlier tables in `docs/oracle_baselines.md` ("Equal-budget results", with its single native run) and the labelled runs here differ slightly (e.g. native gZ6f7yhEvPG 94.3 against 92.4 ± 1.7 at 1000 steps); the seed means above supersede them.
- Not done: GT navmesh as a separate navigation toggle (`planner.oracle.navigation`), grazing-angle weighting in NBV.
- Exact launch commands were not logged; this log is reconstructed. Future reruns should record them here.

## Appendix: run metadata (`experiment_log.py`)

Git at log time: `65293ad0ca` on `main`; uncommitted tracked files: none

| Run | Scene | Steps | Sensor config | Optimisations | Pose noise | Run directory |
|---|---|---|---|---|---|---|
| native_gZ6f_10-12_17-47 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-12_17-47-47_mp3d_gZ6f7yhEvPG` |
| native_pLe4_10-12_19-03 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-12_19-03-18_mp3d_pLe4wQe7qrG` |
| native_YmJk_10-12_19-37 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-12_19-37-38_mp3d_YmJkqBEsHnH` |
| oracle-info_Gdvg_10-12_21-56 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-12_21-56-00_mp3d_GdvgFV5R1Z5` |
| oracle-info_pLe4_10-12_22-35 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-12_22-35-55_mp3d_pLe4wQe7qrG` |
| oracle-info_YmJk_10-12_23-05 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-12_23-05-55_mp3d_YmJkqBEsHnH` |
| tour_(distance_order)_pLe4_10-13_13-35 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_13-35-46_mp3d_pLe4wQe7qrG` |
| NBV_gZ6f_10-13_13-36 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_13-36-02_mp3d_gZ6f7yhEvPG` |
| NBV_pLe4_10-13_13-45 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_13-45-34_mp3d_pLe4wQe7qrG` |
| oracle-info_gZ6f_10-13_13-45 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_13-45-43_mp3d_gZ6f7yhEvPG` |
| tour_(distance_order)_YmJk_10-13_13-56 | YmJkqBEsHnH | 983 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_13-56-47_mp3d_YmJkqBEsHnH` |
| tour_(distance_order)_Gdvg_10-13_14-03 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_14-03-52_mp3d_GdvgFV5R1Z5` |
| NBV_YmJk_10-13_14-05 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_14-05-03_mp3d_YmJkqBEsHnH` |
| NBV_Gdvg_10-13_14-12 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_14-12-09_mp3d_GdvgFV5R1Z5` |
| native_Gdvg_10-13_14-56 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_14-56-15_mp3d_GdvgFV5R1Z5` |
| tour_(distance_order)_gZ6f_10-13_14-56 | gZ6f7yhEvPG | 996 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_14-56-15_mp3d_gZ6f7yhEvPG` |
| tour_(heading_order)_gZ6f_10-13_16-15 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-15-28_mp3d_gZ6f7yhEvPG` |
| tour_(heading_order)_pLe4_10-13_16-16 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-16-51_mp3d_pLe4wQe7qrG` |
| NBV_w1_gZ6f_10-13_16-24 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-24-50_mp3d_gZ6f7yhEvPG` |
| NBV_w1_pLe4_10-13_16-27 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-27-03_mp3d_pLe4wQe7qrG` |
| NBV_w3_gZ6f_10-13_16-33 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-33-28_mp3d_gZ6f7yhEvPG` |
| NBV_w3_pLe4_10-13_16-36 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-36-53_mp3d_pLe4wQe7qrG` |
| random_walk_gZ6f_10-13_16-42 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-42-04_mp3d_gZ6f7yhEvPG` |
| random_walk_pLe4_10-13_16-46 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-46-10_mp3d_pLe4wQe7qrG` |
| native_gZ6f_10-13_16-49 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-49-02_mp3d_gZ6f7yhEvPG` |
| native_pLe4_10-13_16-56 | pLe4wQe7qrG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_16-56-18_mp3d_pLe4wQe7qrG` |
| native_gZ6f_10-13_17-04 | gZ6f7yhEvPG | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_17-04-18_mp3d_gZ6f7yhEvPG` |
| tour_(heading_order)_Gdvg_10-13_17-31 | GdvgFV5R1Z5 | 982 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_17-31-12_mp3d_GdvgFV5R1Z5` |
| NBV_w1_Gdvg_10-13_17-39 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_17-39-37_mp3d_GdvgFV5R1Z5` |
| NBV_w3_Gdvg_10-13_17-49 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_17-49-00_mp3d_GdvgFV5R1Z5` |
| random_walk_Gdvg_10-13_17-57 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_17-57-57_mp3d_GdvgFV5R1Z5` |
| native_Gdvg_10-13_18-04 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_18-04-08_mp3d_GdvgFV5R1Z5` |
| native_Gdvg_10-13_18-20 | GdvgFV5R1Z5 | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_18-20-32_mp3d_GdvgFV5R1Z5` |
| tour_(heading_order)_YmJk_10-13_18-49 | YmJkqBEsHnH | 983 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_18-49-31_mp3d_YmJkqBEsHnH` |
| NBV_w1_YmJk_10-13_18-57 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_18-57-48_mp3d_YmJkqBEsHnH` |
| NBV_w3_YmJk_10-13_19-06 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_19-06-39_mp3d_YmJkqBEsHnH` |
| random_walk_YmJk_10-13_19-15 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_19-15-58_mp3d_YmJkqBEsHnH` |
| native_YmJk_10-13_19-22 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_19-22-58_mp3d_YmJkqBEsHnH` |
| native_YmJk_10-13_19-39 | YmJkqBEsHnH | 1000 | `config/env/activesplat_high_resolution_pointnav.yaml` | before the switches (baseline) | off | `results/archive/2026-10-13_19-39-43_mp3d_YmJkqBEsHnH` |

| Run | Set | PSNR | SSIM | LPIPS | Depth L1 (m) | Coverage | ATE RMSE (cm) | Mean rot. err (deg) |
|---|---|---|---|---|---|---|---|---|
| native_gZ6f_10-12_17-47 | trajectory (201 views) | 17.6 | 0.72 | 0.271 | 0.01 | 92.9 % | 0 | 0 |
| native_gZ6f_10-12_17-47 | novel (100 views) | 17.9 | 0.70 | 0.270 | 0.02 | 93.2 % | 0 | 0 |
| native_pLe4_10-12_19-03 | trajectory (201 views) | 26.0 | 0.78 | 0.252 | 0.02 | 99.7 % | 0 | 0 |
| native_pLe4_10-12_19-03 | novel (100 views) | 23.9 | 0.74 | 0.311 | 0.04 | 98.8 % | 0 | 0 |
| native_YmJk_10-12_19-37 | trajectory (201 views) | 15.3 | 0.64 | 0.326 | 0.05 | 83.7 % | 0 | 0 |
| native_YmJk_10-12_19-37 | novel (100 views) | 15.2 | 0.67 | 0.331 | 0.06 | 90.3 % | 0 | 0 |
| oracle-info_Gdvg_10-12_21-56 | trajectory (201 views) | 24.2 | 0.78 | 0.220 | 0.01 | 99.7 % | 0 | 0 |
| oracle-info_Gdvg_10-12_21-56 | novel (100 views) | 22.6 | 0.74 | 0.285 | 0.04 | 99.4 % | 0 | 0 |
| oracle-info_pLe4_10-12_22-35 | trajectory (201 views) | 25.6 | 0.77 | 0.273 | 0.03 | 99.9 % | 0 | 0 |
| oracle-info_pLe4_10-12_22-35 | novel (100 views) | 24.1 | 0.73 | 0.325 | 0.03 | 99.4 % | 0 | 0 |
| oracle-info_YmJk_10-12_23-05 | trajectory (201 views) | 19.7 | 0.76 | 0.244 | 0.03 | 97.9 % | 0 | 0 |
| oracle-info_YmJk_10-12_23-05 | novel (100 views) | 17.5 | 0.72 | 0.290 | 0.04 | 96.7 % | 0 | 0 |
| tour_(distance_order)_pLe4_10-13_13-35 | trajectory (201 views) | 25.1 | 0.77 | 0.299 | 0.03 | 99.8 % | 0 | 0 |
| tour_(distance_order)_pLe4_10-13_13-35 | novel (100 views) | 24.1 | 0.73 | 0.347 | 0.03 | 99.6 % | 0 | 0 |
| NBV_gZ6f_10-13_13-36 | trajectory (201 views) | 17.1 | 0.73 | 0.278 | 0.02 | 93.8 % | 0 | 0 |
| NBV_gZ6f_10-13_13-36 | novel (100 views) | 18.1 | 0.68 | 0.305 | 0.02 | 94.0 % | 0 | 0 |
| NBV_pLe4_10-13_13-45 | trajectory (201 views) | 24.9 | 0.76 | 0.313 | 0.03 | 99.8 % | 0 | 0 |
| NBV_pLe4_10-13_13-45 | novel (100 views) | 23.2 | 0.70 | 0.383 | 0.04 | 99.5 % | 0 | 0 |
| oracle-info_gZ6f_10-13_13-45 | trajectory (201 views) | 17.8 | 0.71 | 0.288 | 0.02 | 93.0 % | 0 | 0 |
| oracle-info_gZ6f_10-13_13-45 | novel (100 views) | 17.9 | 0.68 | 0.310 | 0.02 | 93.5 % | 0 | 0 |
| tour_(distance_order)_YmJk_10-13_13-56 | trajectory (197 views) | 20.7 | 0.77 | 0.267 | 0.03 | 98.3 % | 0 | 0 |
| tour_(distance_order)_YmJk_10-13_13-56 | novel (100 views) | 18.3 | 0.70 | 0.314 | 0.04 | 97.2 % | 0 | 0 |
| tour_(distance_order)_Gdvg_10-13_14-03 | trajectory (201 views) | 24.7 | 0.79 | 0.229 | 0.01 | 99.9 % | 0 | 0 |
| tour_(distance_order)_Gdvg_10-13_14-03 | novel (100 views) | 23.3 | 0.75 | 0.264 | 0.02 | 99.8 % | 0 | 0 |
| NBV_YmJk_10-13_14-05 | trajectory (201 views) | 14.1 | 0.60 | 0.375 | 0.07 | 82.6 % | 0 | 0 |
| NBV_YmJk_10-13_14-05 | novel (100 views) | 13.4 | 0.57 | 0.407 | 0.10 | 85.9 % | 0 | 0 |
| NBV_Gdvg_10-13_14-12 | trajectory (201 views) | 23.6 | 0.75 | 0.271 | 0.01 | 99.8 % | 0 | 0 |
| NBV_Gdvg_10-13_14-12 | novel (100 views) | 23.0 | 0.72 | 0.290 | 0.01 | 99.8 % | 0 | 0 |
| native_Gdvg_10-13_14-56 | trajectory (201 views) | 25.2 | 0.78 | 0.213 | 0.01 | 99.8 % | 0 | 0 |
| native_Gdvg_10-13_14-56 | novel (100 views) | 22.1 | 0.72 | 0.296 | 0.02 | 99.2 % | 0 | 0 |
| tour_(distance_order)_gZ6f_10-13_14-56 | trajectory (200 views) | 17.7 | 0.74 | 0.258 | 0.02 | 94.2 % | 0 | 0 |
| tour_(distance_order)_gZ6f_10-13_14-56 | novel (100 views) | 19.4 | 0.71 | 0.292 | 0.02 | 95.4 % | 0 | 0 |
| tour_(heading_order)_gZ6f_10-13_16-15 | trajectory (201 views) | 17.4 | 0.72 | 0.288 | 0.02 | 94.9 % | 0 | 0 |
| tour_(heading_order)_gZ6f_10-13_16-15 | novel (100 views) | 17.9 | 0.66 | 0.334 | 0.02 | 93.7 % | 0 | 0 |
| tour_(heading_order)_pLe4_10-13_16-16 | trajectory (201 views) | 24.7 | 0.76 | 0.307 | 0.04 | 99.7 % | 0 | 0 |
| tour_(heading_order)_pLe4_10-13_16-16 | novel (100 views) | 23.5 | 0.71 | 0.355 | 0.04 | 99.6 % | 0 | 0 |
| NBV_w1_gZ6f_10-13_16-24 | trajectory (201 views) | 17.0 | 0.73 | 0.269 | 0.02 | 94.5 % | 0 | 0 |
| NBV_w1_gZ6f_10-13_16-24 | novel (100 views) | 18.1 | 0.70 | 0.298 | 0.02 | 95.5 % | 0 | 0 |
| NBV_w1_pLe4_10-13_16-27 | trajectory (201 views) | 24.7 | 0.75 | 0.325 | 0.03 | 99.8 % | 0 | 0 |
| NBV_w1_pLe4_10-13_16-27 | novel (100 views) | 22.8 | 0.69 | 0.368 | 0.03 | 99.4 % | 0 | 0 |
| NBV_w3_gZ6f_10-13_16-33 | trajectory (201 views) | 17.1 | 0.71 | 0.288 | 0.02 | 92.9 % | 0 | 0 |
| NBV_w3_gZ6f_10-13_16-33 | novel (100 views) | 18.7 | 0.70 | 0.315 | 0.02 | 95.0 % | 0 | 0 |
| NBV_w3_pLe4_10-13_16-36 | trajectory (201 views) | 25.0 | 0.76 | 0.339 | 0.03 | 99.8 % | 0 | 0 |
| NBV_w3_pLe4_10-13_16-36 | novel (100 views) | 23.8 | 0.70 | 0.395 | 0.04 | 99.6 % | 0 | 0 |
| random_walk_gZ6f_10-13_16-42 | trajectory (201 views) | 17.4 | 0.76 | 0.296 | 0.03 | 93.3 % | 0 | 0 |
| random_walk_gZ6f_10-13_16-42 | novel (100 views) | 17.5 | 0.71 | 0.292 | 0.02 | 94.3 % | 0 | 0 |
| random_walk_pLe4_10-13_16-46 | trajectory (201 views) | 23.0 | 0.82 | 0.320 | 0.08 | 93.4 % | 0 | 0 |
| random_walk_pLe4_10-13_16-46 | novel (100 views) | 25.1 | 0.80 | 0.285 | 0.05 | 98.3 % | 0 | 0 |
| native_gZ6f_10-13_16-49 | trajectory (201 views) | 17.2 | 0.69 | 0.298 | 0.02 | 90.7 % | 0 | 0 |
| native_gZ6f_10-13_16-49 | novel (100 views) | 17.8 | 0.69 | 0.289 | 0.02 | 92.5 % | 0 | 0 |
| native_pLe4_10-13_16-56 | trajectory (201 views) | 26.2 | 0.79 | 0.249 | 0.03 | 99.9 % | 0 | 0 |
| native_pLe4_10-13_16-56 | novel (100 views) | 24.0 | 0.75 | 0.315 | 0.04 | 99.1 % | 0 | 0 |
| native_gZ6f_10-13_17-04 | trajectory (201 views) | 18.1 | 0.74 | 0.277 | 0.01 | 91.1 % | 0 | 0 |
| native_gZ6f_10-13_17-04 | novel (100 views) | 17.4 | 0.68 | 0.314 | 0.02 | 92.5 % | 0 | 0 |
| tour_(heading_order)_Gdvg_10-13_17-31 | trajectory (197 views) | 24.6 | 0.79 | 0.232 | 0.01 | 99.4 % | 0 | 0 |
| tour_(heading_order)_Gdvg_10-13_17-31 | novel (100 views) | 22.9 | 0.75 | 0.276 | 0.03 | 99.3 % | 0 | 0 |
| NBV_w1_Gdvg_10-13_17-39 | trajectory (201 views) | 24.1 | 0.76 | 0.272 | 0.01 | 99.9 % | 0 | 0 |
| NBV_w1_Gdvg_10-13_17-39 | novel (100 views) | 22.9 | 0.72 | 0.310 | 0.02 | 99.8 % | 0 | 0 |
| NBV_w3_Gdvg_10-13_17-49 | trajectory (201 views) | 23.5 | 0.74 | 0.276 | 0.01 | 99.8 % | 0 | 0 |
| NBV_w3_Gdvg_10-13_17-49 | novel (100 views) | 22.0 | 0.70 | 0.309 | 0.02 | 99.8 % | 0 | 0 |
| random_walk_Gdvg_10-13_17-57 | trajectory (201 views) | 23.8 | 0.78 | 0.233 | 0.01 | 99.8 % | 0 | 0 |
| random_walk_Gdvg_10-13_17-57 | novel (100 views) | 23.2 | 0.77 | 0.218 | 0.01 | 99.6 % | 0 | 0 |
| native_Gdvg_10-13_18-04 | trajectory (201 views) | 24.1 | 0.77 | 0.211 | 0.01 | 99.9 % | 0 | 0 |
| native_Gdvg_10-13_18-04 | novel (100 views) | 22.7 | 0.74 | 0.269 | 0.04 | 99.5 % | 0 | 0 |
| native_Gdvg_10-13_18-20 | trajectory (201 views) | 24.2 | 0.78 | 0.203 | 0.01 | 99.9 % | 0 | 0 |
| native_Gdvg_10-13_18-20 | novel (100 views) | 22.8 | 0.74 | 0.257 | 0.02 | 99.6 % | 0 | 0 |
| tour_(heading_order)_YmJk_10-13_18-49 | trajectory (197 views) | 18.8 | 0.75 | 0.287 | 0.03 | 96.8 % | 0 | 0 |
| tour_(heading_order)_YmJk_10-13_18-49 | novel (100 views) | 18.2 | 0.71 | 0.316 | 0.03 | 97.2 % | 0 | 0 |
| NBV_w1_YmJk_10-13_18-57 | trajectory (201 views) | 16.4 | 0.70 | 0.319 | 0.03 | 94.0 % | 0 | 0 |
| NBV_w1_YmJk_10-13_18-57 | novel (100 views) | 16.1 | 0.66 | 0.350 | 0.05 | 94.5 % | 0 | 0 |
| NBV_w3_YmJk_10-13_19-06 | trajectory (201 views) | 19.0 | 0.74 | 0.296 | 0.02 | 97.8 % | 0 | 0 |
| NBV_w3_YmJk_10-13_19-06 | novel (100 views) | 18.5 | 0.69 | 0.316 | 0.04 | 97.9 % | 0 | 0 |
| random_walk_YmJk_10-13_19-15 | trajectory (201 views) | 21.1 | 0.81 | 0.234 | 0.02 | 98.7 % | 0 | 0 |
| random_walk_YmJk_10-13_19-15 | novel (100 views) | 20.1 | 0.77 | 0.258 | 0.03 | 97.8 % | 0 | 0 |
| native_YmJk_10-13_19-22 | trajectory (201 views) | 16.1 | 0.65 | 0.323 | 0.04 | 86.0 % | 0 | 0 |
| native_YmJk_10-13_19-22 | novel (100 views) | 15.5 | 0.65 | 0.335 | 0.05 | 91.3 % | 0 | 0 |
| native_YmJk_10-13_19-39 | trajectory (201 views) | 15.8 | 0.65 | 0.321 | 0.05 | 83.7 % | 0 | 0 |
| native_YmJk_10-13_19-39 | novel (100 views) | 15.5 | 0.66 | 0.338 | 0.06 | 89.8 % | 0 | 0 |
