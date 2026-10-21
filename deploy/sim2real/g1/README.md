# G1 sim2real

This directory marks the hardware deployment boundary and is intentionally
separate from `deploy/sim2sim/g1/`. No real-robot controller is implemented
here yet.

The sim2sim controller depends on `unitree_mujoco`'s `rt/sportmodestate`
velocity topic. A real G1 low-level controller needs a supported estimate of
base linear velocity, hardware-specific command safety and startup behavior,
and validation of gains and action limits. Do not point the sim2sim binary at a
physical robot as a substitute for that work.

The ONNX actor can be shared if its observation and action contracts match;
the state acquisition and command adapters must be implemented and checked for
the hardware target separately.
