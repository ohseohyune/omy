import mujoco
import mujoco.viewer as mj_view
import numpy as np
import time

# ----------------------
# Simulation Configuration
# ----------------------

MODEL_XML_PATH = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print("Actuators:", [model.actuator(i).name for i in range(model.nu)])

dt = model.opt.timestep

# ----------------------
# Admittance Parameters
# ----------------------

M = np.diag([5, 5, 5, 0.5, 0.5, 0.5]) * 0.5
D = np.diag([10, 10, 10, 1, 1, 1]) * 0.5
K = np.zeros((6, 6))

x_e = np.zeros(6)
dx_e = np.zeros(6)
ddx_e = np.zeros(6)
target_wrench = np.zeros(6)

# ----------------------
# Helper Functions
# ----------------------

def get_sensor_wrench():
    if model.nsensor >= 2:
        return data.sensordata[:6]
    else:
        return np.zeros(6)

def apply_deadband(wrench, force_band=0.5, torque_band=0.05):
    if np.linalg.norm(wrench[:3]) < force_band and np.linalg.norm(wrench[3:]) < torque_band:
        return np.zeros(6)
    return wrench

def get_zeroed_sensor_wrench():
    return get_sensor_wrench() - sensor_zero

# ----------------------
# Warm-up Step for Stability
# ----------------------

for _ in range(100):
    mujoco.mj_step(model, data)

sensor_zero = get_sensor_wrench()

# ----------------------
# Initial Joint Positions
# ----------------------

qpos_target = np.copy(data.qpos[:6])

# ----------------------
# Real-Time Simulation Loop
# ----------------------

try:
    with mj_view.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Read and filter wrench
            wrench = apply_deadband(get_zeroed_sensor_wrench())

            # Admittance Dynamics Calculation
            ddx_e_new = np.linalg.inv(M) @ (wrench - target_wrench - D @ dx_e - K @ x_e)
            dx_e += dt * 0.5 * (ddx_e + ddx_e_new)
            x_e += dt * dx_e
            ddx_e = ddx_e_new

            # Velocity integration to joint positions
            scaling_factor = 10.0  # ✅ 튜닝 필요
            qpos_target += scaling_factor * dx_e[:6] * dt

            # Send position commands to actuators
            for i in range(min(model.nu, 6)):
                data.ctrl[i] = qpos_target[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")
