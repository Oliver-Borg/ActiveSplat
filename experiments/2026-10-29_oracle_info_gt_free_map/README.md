# Oracle-info with the ground-truth free map (2026-10-29)

**Question.** The oracle-info planner (see `experiments/2026-10-14_oracle_baselines/`) still builds its Voronoi graph from the free map the mapper has seen so far. What changes if the graph is built from the *ground-truth* free map from step 0, so that only the view scoring is oracle and the navigation structure is complete from the start? (Config `planner.oracle.navigation: gt_free_map`; code in `src/oracle/gt_map.py`, override in `src/visualizer/visualizer.py`.)

## Setup
- Scenes: the four MP3D scenes of the oracle log; 1000 steps; sensor/mapping config `mp3d_high_resolution_oracle_info_gt_free_map.json` (oracle-info, voxel 0.05, spacing 0.02, `navigation: gt_free_map`); one run per scene; all optimisations on, JPEG output.
- GT free map: GT-mesh surface samples in the agent's height band, dilated 3x3, restricted to discs (0.6 m) around the observable-cache camera positions, 4-connected component containing the start. `visible_map_binary` is all zeros (everything "seen").
- Launch: `bash experiments/2026-10-29_oracle_info_gt_free_map/run.sh <gpuA> [<gpuB>]` (env `V=` tags manifest/log names, `SCENES_A`/`SCENES_B` choose scenes). Two GPUs at most, each waits for a free GPU. Per-run logs `launch_<v>_<scene>.log`, manifests `manifest_<v>.txt`. After each run: `scripts/oracle/coverage.py`, label written to `label.txt`.
- Versions (each a code change after seeing a failure, see Reading):
  - **v1** (label "no opening"): map as above. Scenes ran 14:28-14:40.
  - **v2**: additionally morphological opening with the agent's disc (radius from the dataset config), then re-take the start's component. Three scenes ran with this.
  - **v3**: v2 plus a disc of the agent radius around the agent's current position OR-ed into the map each time it is sent to the planner. Only GdvgFV5R1Z5 ran with this (its v2 run crashed, see below). gZ6f7yhEvPG, pLe4wQe7qrG, YmJkqBEsHnH are v2 runs; the reported "oracle-info (GT free map)" row mixes v2 (3 scenes) and v3 (1 scene).
- Comparison: `scripts/oracle/compare.py` over the archived baseline runs (`results/archive`, MANIFEST.txt of the oracle log) plus these runs. Native has 2-3 runs per scene, oracle-info and the new rows 1.

## Run metadata
Generated with `scripts/tools/experiment_log.py` (full table in `runs.json`; git `d7d1e00b70` + uncommitted `planner_node.py`, `coverage.py` unrelated to the planner logic here, `run.sh` edited for v3). All runs: 1000 steps, `activesplat_high_resolution_pointnav.yaml`, all optimisations, pose noise off.

| Run | Run directory |
|---|---|
| v1 gZ6f7yhEvPG / GdvgFV5R1Z5 / YmJkqBEsHnH / pLe4wQe7qrG | `results/2026-10-29_14-28-52_mp3d_gZ6f7yhEvPG`, `..._14-28-28_mp3d_GdvgFV5R1Z5`, `..._14-39-19_mp3d_YmJkqBEsHnH`, `..._14-40-40_mp3d_pLe4wQe7qrG` |
| v2 gZ6f7yhEvPG / pLe4wQe7qrG / YmJkqBEsHnH | `results/2026-10-29_14-55-15_mp3d_gZ6f7yhEvPG`, `..._15-05-43_mp3d_pLe4wQe7qrG`, `..._17-11-28_mp3d_YmJkqBEsHnH` |
| v2 GdvgFV5R1Z5 (crashed at step 626, excluded) | `results/2026-10-29_14-55-15_mp3d_GdvgFV5R1Z5` |
| v3 GdvgFV5R1Z5 | `results/2026-10-29_17-22-46_mp3d_GdvgFV5R1Z5` |

(Smoke runs for gZ6f7yhEvPG, 150 steps: `..._10-29-51` with the inverted map polarity, `..._10-38-46` good.)

## Results
Coverage of the observable surface at step 1000 (%), AUC in brackets.

| scene | native (2-3 runs) | oracle-info | GT free map v1 | GT free map (v2/v3) |
|---|---|---|---|---|
| GdvgFV5R1Z5 | 89.0 ± 2.7 (72.8) | 92.9 (74.0) | 92.4 (76.4) | 93.3 (73.3) [v3] |
| YmJkqBEsHnH | 69.4 ± 6.2 (46.6) | 81.7 (49.7) | 86.5 (57.9) | 75.2 (44.3) [v2] |
| gZ6f7yhEvPG | 92.4 ± 1.7 (71.5) | 95.7 (71.2) | 88.6 (69.2) | 96.5 (70.4) [v2] |
| pLe4wQe7qrG | 82.9 ± 0.3 (45.2) | 85.4 (54.4) | **30.4** (28.5), stalled | 83.2 (45.5) [v2] |

Mean over the four scenes (full tables incl. steps-to-80/90 %, path, two-direction coverage and image metrics for every method: re-run compare as in Setup):

| method | cov@100 | cov@500 | cov@1000 | AUC | path m | two dir. % | PSNR | novel PSNR | novel LPIPS |
|---|---|---|---|---|---|---|---|---|---|
| native | 32.6 | 64.4 | 83.4 | 59.0 | 21.8 | 58.5 | 21.0 | 19.9 | 0.303 |
| oracle-info | 32.2 | 68.9 | 88.9 | 62.3 | 22.6 | 63.1 | 21.8 | 20.6 | 0.302 |
| oracle-info, GT free map v1 | 30.6 | 65.6 | 74.5 (stall on pLe4) | 58.0 | 16.4 | 51.4 | 22.1 | 19.2 | 0.304 |
| oracle-info, GT free map v2/v3 | 30.9 | 61.4 | 87.0 | 58.4 | 23.0 | 57.8 | 22.3 | 20.7 | 0.290 |
| tour (heading order) | 38.4 | 62.9 | 92.6 | 64.5 | 22.2 | 70.2 | 21.4 | 20.6 | 0.320 |
| NBV | 44.0 | 85.2 | 95.3 | 77.2 | 29.4 | 74.2 | 19.9 | 19.4 | 0.346 |

Figure: `coverage_pLe4wQe7qrG.jpg` (all methods on the scene that stalled in v1).

## Reading
- Giving the planner the complete graph from step 0 did **not** help: mean coverage 87.0 % vs 88.9 % for plain oracle-info (and 83.4 % native), AUC 58.4 vs 62.3. Per scene it is within a few points of oracle-info on two scenes, better on one (gZ6f7yhEvPG, +0.8) and clearly worse on YmJkqBEsHnH (75.2 vs 81.7). With one run per scene and native noise of 0.3-6 points, only the YmJkqBEsHnH gap and the NBV gap are larger than the noise; the rest is inconclusive. The early curve is not faster (cov@100 30.9 vs 32.2): the first stretch is dominated by the same look-around scoring, and a complete graph does not make unexplored regions cheaper to reach.
- Neither the oracle-informed planner nor any variant using the native subregion/node logic comes close to NBV (95.3 %, AUC 77.2); the gap is in the view selection, not the navigation graph.
- Failures that shaped the versions: (v1) pLe4wQe7qrG stalled at 30.4 % after ~280 steps with 3.6 m of path (hundreds of turns, "No node is reachable" in the log); cause inferred, not proven: the GT map has pockets/passages that the planner's safe-path test (polyline of width 2x agent radius) rejects, and the global plan picks subregions without a reachability check. v2's opening removes them and the scene then reached 83.2 % with no "No node is reachable" lines. (v2) GdvgFV5R1Z5 crashed at step 626 (`get_obstacle_map`: "Robot position is not in the obstacle map", argmin of empty sequence) because the agent stood a pixel outside the opened map; v3 keeps a disc around the agent free and the rerun finished (93.3 %, no such lines). The other three v2 scenes did not have that fix but did not hit the problem. v1 results on the three scenes that did not stall are comparable to v2 (both listed); v1 YmJkqBEsHnH is better than v2 (86.5 vs 75.2) and v1 gZ6f7yhEvPG worse (88.6 vs 96.5): the version effect cannot be told from run-to-run noise with one run each.

## Confounds / caveats
- Single run per GT-free-map row; native is not bit-reproducible (diverges after ~250 steps), so each single run is one sample of a noisy process.
- The new runs use current code (optimisations, JPEG outputs); the archived native/oracle-info baselines come from earlier code. A control rerun of plain oracle-info on current code was not done.
- The reported row mixes v2 and v3 (see Setup); v3 differs only by the agent disc and was needed for one scene.
- Image metrics are only comparable between runs that explored the same area (the stalled v1 pLe4wQe7qrG has a high trajectory PSNR because it saw very little).
- The agent radius value used for the opening comes from the dataset config and was not independently checked against the planner's own 0.225 m.
- The crashed v2 GdvgFV5R1Z5 run (83.1 % at step 626, no image metrics) is excluded from all tables.
