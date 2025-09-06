import mujoco
import mujoco.viewer as mj_view
import numpy as np
import time
import pinocchio as pin
#from pinocchio.urdf import buildModelFromUrdf
from trajectory_generator import generate_trajectory  # Trajectory 생성기

# ----------------------
# MuJoCo & Simulation Setup
# ----------------------

MODEL_XML_PATH = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/scene.xml"
URDF_PATH = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/robot.urdf"

model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

dt = model.opt.timestep
n_act = 6  # 제어할 조인트 수 (6개)

# ----------------------
# Pinocchio Model Load
# ----------------------

#pin_model = buildModelFromUrdf(URDF_PATH)
pin_model = pin.buildModelFromUrdf(URDF_PATH)
pin_data = pin_model.createData()

ee_frame_name = "link6"  
ee_frame_id = pin_model.getFrameId(ee_frame_name)

# ----------------------
# Forward Kinematics로 현재 EE 위치 확인
# ----------------------

q0_full = np.zeros(pin_model.nq)  # 초기 조인트값 (전역)
#q0_full[:6] = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])  # 예시

pin.forwardKinematics(pin_model, pin_data, q0_full)
pin.updateFramePlacements(pin_model, pin_data)
x0 = pin_data.oMf[ee_frame_id].translation.copy()
print("초기 EE 위치 (x0):", x0)


# ----------------------
# 목표 EE 위치 설정 (예: 위로 10cm 이동)
# ----------------------

x1 = x0 + np.array([0.05, 0.01, -0.01])
print("목표 EE 위치 (x1):", x1)

# ----------------------
# Simple Inverse Kinematics Solver
# ----------------------

def solve_ik(initial_q, target_translation, max_iter=100, tol=1e-4):
    q = initial_q.copy()
    for _ in range(max_iter):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        current_translation = pin_data.oMf[ee_frame_id].translation
        error = target_translation - current_translation

        if np.linalg.norm(error) < tol:
            break

        J = pin.computeFrameJacobian(pin_model, pin_data, q, ee_frame_id)
        v = np.linalg.pinv(J[:3, :]) @ error
        q[:n_act] += v[:n_act] * 0.5  # 조정된 스텝 사이즈

    return q[:n_act]

q0 = q0_full[:n_act]
q1 = solve_ik(q0_full, x1)
print(" IK로 구한 최종 조인트 q1:", q1)

# ----------------------
# Generate Joint Trajectory
# ----------------------

v0 = np.zeros(n_act)
v1 = np.zeros(n_act)
V = np.array([0.3, 0.3, 0.3, 0.5, 0.8, 0.8])
A = np.array([0.5, 0.5, 0.5, 0.8, 1.0, 1.0])
t0 = 0.001
n_steps = 500

traj_t, traj_q, traj_qd = generate_trajectory(q0, q1, v0, v1, V, A, t0, n_steps)

# ----------------------
# Warm-up Step for Stability
# ----------------------

for _ in range(100):
    mujoco.mj_step(model, data)

# ----------------------
# Trajectory Execution Loop
# ----------------------

try:
    with mj_view.launch_passive(model, data) as viewer:
        idx = 0
        while viewer.is_running():
            if idx >= len(traj_t):
                print("✅ Trajectory completed.")
                break

            data.ctrl[:n_act] = traj_q[idx]
            mujoco.mj_step(model, data)
            viewer.sync()
            idx += 1
            time.sleep(dt)

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")
