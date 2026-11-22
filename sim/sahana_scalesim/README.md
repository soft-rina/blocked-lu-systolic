# Sahana's CSEN 318 SCALE-Sim workflow

This package covers the weight-stationary accelerator work through the mid-progress report and
queues the remaining workload, shard, and validation runs. Use the environment
and a working configuration from your CSEN 318 assignment. Run the
experiments yourself in that environment.

## 1. Connect to your assignment setup

1. Extract this ZIP. Open Terminal in your original SCALE-Sim folder with the
   assignment's Python environment active. Type `pwd` to get the absolute path.
2. In another Terminal command, change to the extracted folder. If you extracted
   in Downloads, use `cd ~/Downloads/sahana_scalesim`.
3. Copy your working **array config** into this folder as
   `base_array.cfg`. For example, replace the quoted path below with the actual one.

   ```bash
   cp "/absolute/path/to/your/assignment/array.cfg" base_array.cfg
   open -e settings.json
   ```

4. In settings.json, replace the `sim_cwd` value with the path from `pwd`.
   Leave `base_config` and the command list as supplied for the first run.
   The runner copies the base config for each job and changes only run name,
   dimensions, and dataflow. Keep your assignment's other settings.
5. Install Matplotlib into the active environment if it is missing.

   ```bash
   python --version
   python -c 'import matplotlib' || python -m pip install matplotlib
   ```

The bundled simulator adapter requests GEMM mode and disables the trace-writing
path with `save_disk_space=True`. If your setup uses a layout file, add
`"--layout", "/absolute/path/to/layout.csv"` to the command array after the
output argument pair. If your instructor supplied a patched launcher, you can
replace the entire command array with your known-working command, one argument
per JSON string. Use `{python}`, `{config}`, `{topology}`, and `{output}`
where those values should go, and preserve the assignment's traces-off option.
The driver writes a one-layer topology with positional columns Layer Name, M,
N, K and a trailing comma. Check that against your assignment's GEMM input
if the smoke case fails in topology parsing.

## 2. Run one case and check the report

```bash
python pipeline.py run smoke
find work_ws/raw_runs -name COMPUTE_REPORT.csv -print
```

Copy the path printed by find. Use `head -n 2 "PASTE THE REPORT PATH HERE"`
to inspect it. Confirm this is the 64×64×64 GEMM on a 128×128 array.
The script prints the cycle count and exact header it used. Open
`work_ws/array_runs.csv` and check that the count matches. This file also keeps
common raw cycle and stall columns when the report provides them.

The default comparison column is `Total Cycles`. Read the header and check the
cycle convention against your assignment before using a number in the report.
Rina's cycle formulas have no explicit memory stall term. If your
chosen simulator column includes additional costs, document that difference.
If your report uses a different header, copy the desired exact header into
`cycle_column` in settings.json. If a successful smoke run already exists,
switch to a fresh folder by adding `--out work_ws_revised` directly after
`pipeline.py` in **every** subsequent command.

If the smoke run fails, open its log.

```bash
cat work_ws/raw_runs/ws_M64_N64_K64_array_h128_array_w128/run.log
```

- `ModuleNotFoundError` usually means the assignment environment is inactive
  or the configured simulator folder is wrong. Activate your original
  environment and verify `sim_cwd`.
- A missing config key means the copied config needs to be checked. The runner
  expects a single run_name, ArrayHeight, ArrayWidth and Dataflow entry.
- If a config uses relative paths for other inputs, keep `sim_cwd` at the same
  location you used for your assignment commands.
- If the adapter needs a layout, use your original layout as described above.

## 3. Produce the files for the progress report

After the smoke test succeeds, run these in order.

```bash
python pipeline.py predict
python pipeline.py run pilot
python pipeline.py diagnostics
python pipeline.py run first-holdout
python pipeline.py validate
python pipeline.py plot
```

- **Predict:** Freezes Rina's exact OS and WS equations using physical R and C,
  with the final minus one. The script does not fit them. The resulting CSV
  includes the six reserved workload shapes before you see the simulator values.
- **Pilot:** Runs WS at trailing dimensions 64, 1024 and 1984 with K=64.
  It adds OS at (1024,1024,64) and WS at (1024,1024,128).
- **Diagnostics:** Writes two comparisons to pilot_diagnostics.csv. Compare
  OS and WS at the exact same shape. Then compare WS at K=64 and K=128 while
  holding M and N fixed. Rina's formula predicts equal cycles for the latter
  pair on 128×128, since both fit in one row fold.
- **First holdout:** Simulates the untouched WS (960,960,64) update on 128×128.
- **Validate:** Writes predicted cycles, observed cycles, and absolute
  percentage error. Use the holdout row in the progress report. If a result
  informs a revised model, use fresh untouched shapes for its validation.
- **Plot:** Plots effective utilization for the single WS 128×128 array at
  K=64, computed as useful MACs divided by available PE-cycles under the
  selected cycle convention. It writes the figure and its exact data.

For a model-only summary of Rina's OS and WS predictions across all LU
updates, run `python cycle_model.py`. Those totals exclude CPU and PCIe.
The number and size of updates change with the block size.

Send Rina these files from work_ws:

| File | What Rina or the report needs from it |
|---|---|
| array_runs.csv | Cycle counts keyed by dataflow,M,N,K,array_h,array_w, plus microseconds at 1 GHz and report paths |
| cycle_validation.csv | First held-out error and the predicted and simulated cycles |
| fig_prelim.pdf | Appendix figure |
| preliminary_figure_data.csv | Exact numbers for the figure caption |
| raw_runs/ | Configs, topologies, logs and reports for reproduction |

The CSV records simulated array timing for a GEMM. Rina adds CPU work and
host-to-accelerator PCIe time in her model. Tell her which simulator cycle
field you selected. Save separate assumptions for the trailing-matrix
subtraction if the modeled accelerator or host performs work beyond GEMM.

## 4. Run the layout pilot and queue the remaining jobs

```bash
python pipeline.py run pairs
python pipeline.py compare
```

The pairs group compares two WS layouts. It adds the late single-array case
and three 64×64 shard cases, reusing early and middle WS pilot reports.
For example, one (128,128,64) GEMM on 128×128 corresponds to four identical
(128,32,64) GEMMs on four 64×64 arrays. With concurrent starts and data ready,
the compute stage ends when the slowest shard ends. The supplied comparison
uses identical shards for the selected dimensions.

The comparison CSV reports measured cycles and **normalized screening EDP**
for relative idle and replicated-input cost scenarios. It charges extra local
delivery of the left operand to the four arrays as a sensitivity case. These
coefficients are scenario inputs. They do not represent measured joules or
model chip-wide energy or transfer latency. A replication coefficient of zero
models zero marginal energy for the extra deliveries. Report the coefficients
and assumptions alongside any result.

To start the remaining unattended batch after predictions are frozen:

```bash
caffeinate -i python pipeline.py run remaining > work_ws/batch.log 2>&1
```

Leave the Mac awake with the lid open and connected to power. The command
runs in the foreground. In another Terminal tab, change to the same folder
and run `tail -f work_ws/batch.log` to see progress. Avoid running two
writers against the same work folder. Rerun the same command to resume, since
completed runs with unchanged settings are skipped and failed attempts are
preserved. To refresh results, run `validate`, `plot` and `compare` again
after the batch ends.

The complete batch contains ten WS workload shapes spanning early through late
updates on the single array, the corresponding shard runs, six reserved
partial-fold workload shapes on each layout (12 simulations), and additional
diagnostics at K=128 and larger sizes. Run `python pipeline.py run holdouts`
if you want to prioritize the reserved shapes before the complete batch.

The six reserved workload cases (original n, trailing s, K) are
(2048,960,64), (2048,448,32), (2048,1344,32), (2048,1600,64),
(4096,4000,32), and (4096,3904,64). On 128×128 these include partial row
folds because K < 128. For the four-way layout they run with N=s/4 on
64×64. K=32 tests partial rows and the selected K=64 shard widths test
partial columns. Compare errors separately for the two array sizes.

These benchmark samples do not enumerate all steps of Rina's runtime
sweep. If an error changes a conclusion, use a direct simulator run or
report the uncertainty. Verify any OS/WS comparison uses the same cycle
column, memory settings, and shape. These scripts leave the old OS runs
in the original work folder and use work_ws for the revised experiments.
