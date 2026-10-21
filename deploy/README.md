# AMP_mjlab deployment

This directory contains a standalone copy of the SMP G1 deployment runtime,
adapted for AMP_mjlab policies. The C++ controller uses Unitree SDK2, ONNX
Runtime, yaml-cpp, Eigen, and Boost; it does not start the Python training
environment.

## G1 Getup sim2sim

AMP_mjlab's G1 Getup actor consumes one flattened `obs` tensor of shape
`[1, 372]` and returns `actions` with shape `[1, 29]`. The observation is four
chronological frames, oldest first. Each 93-value frame is ordered as
`base_ang_vel(3), projected_gravity(3), joint_pos_rel(29), joint_vel_rel(29),
last_action(29)`. The exported ONNX contains the actor observation
normalization. Joint vectors use the G1 XML order: left leg, right leg, waist,
left arm, right arm.

Export an ONNX policy from a trained checkpoint with:

```bash
python scripts/play.py Unitree-G1-AMP-Getup \
  --checkpoint-file /path/to/run/model_<iteration>.pt
```

The AMP play script exports the loaded actor to the run's `export/` directory
as `Unitree-G1-AMP-Getup_model_<iteration>.onnx`. Confirm the actual path in the
play output. Install the pinned ONNX Runtime bundle and build from the
AMP_mjlab repository root:

```bash
bash scripts/setup_sim2sim_deps.sh
cmake -S deploy/sim2sim/g1 -B deploy/sim2sim/g1/build
cmake --build deploy/sim2sim/g1/build -j
```

Start Unitree MuJoCo with its G1 scene in a fallen pose (DDS domain 0), then
start the controller in another terminal:

```bash
cd /path/to/unitree_mujoco/simulate
ln -s "$HOME/.mujoco/mujoco-3.3.6" mujoco  # only if simulate/mujoco is absent
cmake -S . -B build
cmake --build build -j
cd build
./unitree_mujoco -r g1 -s scene.xml -i 0 -n lo
```

Set the G1 initial pose to the fallen recovery state in the simulator scene.
In a second terminal, from the AMP_mjlab repository root:

```bash
./deploy/sim2sim/g1/build/g1_sim2sim_controller \
  --network=lo \
  --onnx=/absolute/path/to/run/export/Unitree-G1-AMP-Getup_model_<iteration>.onnx
```

The controller defaults to
`deploy/sim2sim/g1/config/policies/getup/params/deploy.yaml`, which contains
the AMP observation history, G1 action scales and offsets, PD gains, and joint
mapping. Those values describe the policy interface and must match the
checkpoint's training configuration. For a checkpoint trained with a changed
interface, pass a separate YAML file:

```bash
./deploy/sim2sim/g1/build/g1_sim2sim_controller \
  --network=lo \
  --onnx=/absolute/path/to/policy.onnx \
  --deploy-config=/absolute/path/to/deploy.yaml
```

Relative `--onnx` and `--deploy-config` paths resolve from the AMP_mjlab
repository root. There is no per-checkpoint deployment package; use the shared
task YAML for compatible runs and a separate YAML only when the interface
configuration differs.

### Selecting the play reset pose

AMP Getup play accepts `--reset-pose random|supine|prone|frame`. The default
`random` samples any Recovery frame. `supine` and `prone` sample from filtered
lying reference frames; `frame` repeats one flattened frame index and requires
`--reset-frame`:

```bash
python scripts/play.py Unitree-G1-AMP-Getup \
  --checkpoint-file /path/to/run/model_<iteration>.pt \
  --viewer viser \
  --reset-pose supine \
  --start-paused
```

In Viser, the **Getup start pose** panel lets you choose Random, Supine, Prone,
or Fixed frame, then click **Reset to selected pose**. That reset pauses the
viewer so the initial pose can be inspected. The selected mode also applies to
automatic resets after success or timeout. The same `--reset-pose` options
work with the native viewer; the Viser controls are optional. A fixed frame
index refers to the flattened, sorted Recovery frame pool.

FSM buttons: `L2 + Up` enters Getup, `L2 + Y` enters PDStand, and `L2 + B`
enters Passive. For keyboard control without a gamepad, run
`python3 deploy/sim2sim/g1/fsm_ui.py` on the same machine after starting the
controller. It requests FSM state changes over localhost UDP port 9876.

## Diagnostics

The task YAML enables a CSV at `logs/sim2sim_getup.csv` by default; set
`diagnostics_csv` to an empty string to disable it or change the path. It logs
one row per policy step with elapsed time, inference duration, raw policy
observations, actions, processed position targets, measured joint positions and
velocities, estimated torque, and target-position error. The CSV header sizes
are derived from runtime vectors, so they follow the loaded policy/config.
