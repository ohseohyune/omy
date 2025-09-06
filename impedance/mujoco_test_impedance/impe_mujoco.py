import mujoco
import mujoco.viewer as mj_view
import numpy as np
import time
import pinocchio as pin
import os

# ----------------------
# Simulation Configuration
# ----------------------

MODEL_XML_PATH = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/impedance/mujoco_test_impedance/scene.xml"
URDF_PATH = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/omy.urdf"  # 변환된 URDF 경로

model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print("Actuators:", [model.actuator(i).name for i in range(model.nu)])

dt = model.opt.timestep

# ----------------------
# Pinocchio Model Setup (for gravity compensation)
# ----------------------

if os.path.exists(URDF_PATH):
    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    pin_data = pin_model.createData()
    pin_model.gravity.linear = np.array([0.0, 0.0, -9.81])
    print(f"Pinocchio model loaded: nq={pin_model.nq}, nv={pin_model.nv}")
    GRAVITY_COMP = True
else:
    print(f"[Warning] URDF not found at {URDF_PATH}, running without gravity compensation")
    pin_model, pin_data = None, None
    GRAVITY_COMP = False

# ----------------------
# Impedance Control Parameters
# ----------------------

Q_DES = np.copy(data.qpos[:6])  # 목표 위치 (초기 qpos)
DQ_DES = np.zeros(6)

K = np.array([4.0, 4.0, 4.0, 3.0, 3.0, 3.0])  # [Nm/rad]
D = np.array([3.0, 3.0, 3.0, 2.0, 2.0, 2.0])  # [Nm/(rad/s)]

VEL_ALPHA = 0.25
dq_filt = np.zeros(6)

# ----------------------
# Helper Functions
# ----------------------

def get_q_dq():
    """현재 MuJoCo 상태에서 q, dq 반환"""
    q = np.copy(data.qpos[:6])
    dq = np.copy(data.qvel[:6])
    return q, dq

def clamp(val, lo, hi):
    return np.maximum(lo, np.minimum(hi, val))

def pinocchio_gravity_torque(q):
    """Pinocchio로 중력 토크 계산"""
    if not GRAVITY_COMP:
        return np.zeros(6)
    try:
        q_np = np.asarray(q).reshape(pin_model.nq)
        tau_g = pin.computeGeneralizedGravity(pin_model, pin_data, q_np)
        return tau_g[:6]  # 앞 6개 관절만
    except Exception as e:
        print(f"[Warning] Gravity computation failed: {e}")
        return np.zeros(6)

# ----------------------
# Simulation Loop
# ----------------------

try:
    with mj_view.launch_passive(model, data) as viewer:
        print("Start OMY impedance + gravity compensation simulation. Press [ESC] to exit.")

        while viewer.is_running():
            q, dq = get_q_dq()

            # 1) 속도 필터링
            dq_filt[:] = (1.0 - VEL_ALPHA) * dq_filt + VEL_ALPHA * dq

            # 2) 임피던스 제어 law
            u = []
            tau_g = pinocchio_gravity_torque(q)  # 중력 토크
            for i in range(6):
                e = Q_DES[i] - q[i]
                ed = DQ_DES[i] - dq_filt[i]
                ui = K[i] * e + D[i] * ed + tau_g[i]  # 중력보상 추가
                u.append(ui)

            # 3) MuJoCo actuator에 제어 입력 적용 (torque actuator 전제)
            for i in range(min(model.nu, 6)):
                data.ctrl[i] = u[i]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

except KeyboardInterrupt:
    print("\n[Ctrl+C] Stopping simulation...")
