## unified_admittance_trajectory_control.py

import mujoco
import mujoco.viewer as mj_view
import numpy as np
import time
import math
import os
#import matplotlib.pyplot as plt # 얘네 호출하면 시뮬레이션 더 느려짐
import pinocchio as pin
#from matplotlib.ticker import MultipleLocator
from pinocchio import buildModelFromUrdf
from trajectory_generator import generate_trajectory


## 1. 모델 & 데이터 로드
model_path = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/"
urdf_filename = "robot.urdf"
urdf_model_path = os.path.join(model_path, urdf_filename)

pin_model = buildModelFromUrdf(urdf_model_path)
pin_data = pin_model.createData()

mjc_model = mujoco.MjModel.from_xml_path(model_path + "scene.xml")
mjc_data = mujoco.MjData(mjc_model)


## 2. 파라미터 정의 & 상태 변수 초기화

# admittance control
M = np.diag([5, 5, 5, 0.5, 0.5, 0.5]) * 0.5 #6 by 6 대각행렬
D = np.diag([10, 10, 10, 1, 1, 1]) * 0.5
K = np.zeros((6, 6))

x_e = np.zeros(6)
dx_e = np.zeros(6)
ddx_e = np.zeros(6)
target_wrench = np.zeros(6) # 로봇의 엔드이펙터에 작용하기를 원하는 힘과 모멘트의 값 
# target_wrench = np.array([0,0,80,0,0,0])
# 보통 [Fx, Fy, Fz, Tx, Ty, Tz] 형태로 구성된 6차원 벡터

# trajectory
q0 = np.zeros(10)
q1 = np.array([0.5, 0.5, 0.4, 0.6, 0.5, 0.3, 0.2, 0.1, 0.2, 0.1])

v0 = np.zeros(10)
v1 = np.zeros(10)
V = np.full(10, 150.0)  
A = np.full(10, 150.0) 

t0 = 0.001
n_steps = 10000

# both
dq_g = np.zeros(pin_model.nv)  # 중력 보상용 속도 변수
gravity_gain = 3.5
dt = mjc_model.opt.timestep
n_act = mjc_model.nu


## 3. 함수 정의 

def get_sensor_wrench():
    if mjc_model.nsensor >= 2:
        return mjc_data.sensordata[:6]
    else:
        return np.zeros(6)

def apply_deadband(wrench, force_band=0.5, torque_band=0.05):
    if np.linalg.norm(wrench[:3]) < force_band and np.linalg.norm(wrench[3:]) < torque_band:
        return np.zeros(6)
    return wrench

def get_zeroed_sensor_wrench():
    return get_sensor_wrench() - sensor_zero


## 4. 이것저것(?)

frame_name = "tcp_link"  # 엔드 이펙터 링크 이름
frame_id = pin_model.getFrameId(frame_name)

# 반드시 존재하는지 확인
if frame_id == len(pin_model.frames):
    raise ValueError(f"[ERROR] Frame '{frame_name}' not found in URDF.")

# 경로 생성하기 
traj_t, traj_q, traj_qd = generate_trajectory(q0, q1, v0, v1, V, A, t0, n_steps)

# 시뮬레이션 워밍업
for _ in range(20):
    mujoco.mj_step(mjc_model, mjc_data)


# 센서 제로화
sensor_zero = get_sensor_wrench()

# 초기 관절 위치 목표값
qpos_target = np.copy(mjc_data.qpos[:pin_model.nq])


# 생성한 관절궤적을 시간 축에 맞춰 각 관절마다 위치 변화를 그려
# 1) Joint position trajectory 그래프
# 1) Joint position trajectory → C 방법 (서브플롯)
#fig, axes = plt.subplots(5, 2, figsize=(12, 10))  # 5행×2열
#axes = axes.ravel()  # (10,)로 펼치기

#for j, ax in enumerate(axes):
#    ax.plot(traj_t, traj_q[:, j], label=f'joint {j+1}')
#    ax.set_title(f'Joint {j+1}')
#    ax.set_xlabel("time [s]")
#    ax.set_ylabel("position [rad]")
#    ax.grid(True)
#    ax.yaxis.set_major_locator(MultipleLocator(0.1))
#    ax.set_xlim(0,0.2)
#    ax.set_ylim(0, 1)

#plt.tight_layout()


# 2) Joint velocity trajectory 그래프
#plt.figure()
#ax = plt.gca()
#for j in range(traj_qd.shape[1]):
#    ax.plot(traj_t, traj_qd[:, j], label=f'joint {j+1}')
#ax.set_title("Joint-space Velocity Trajectories")
#ax.set_xlabel("time [s]")
#ax.set_ylabel("velocity [rad/s]")
#ax.legend()
#ax.grid(True)

#plt.show()

# 시뮬레이션 시작
start_time = time.time()
sim_start = time.perf_counter()

## 5. 시뮬 루프 

try:
    with mj_view.launch_passive(mjc_model, mjc_data) as viewer:
        idx = 0
        while viewer.is_running():
            if idx >= len(traj_t):
                print("Trajectory Ended. Exiting loop.")
                break
             
            t = time.time() - start_time

            # MuJoCo 현재 상태 → Pinocchio에 적용
            q_mjc = np.copy(mjc_data.qpos[:pin_model.nq])
            dq_mjc = np.copy(mjc_data.qvel[:pin_model.nv])
            
            # Pinocchio 동역학 데이터 업데이트 
            pin.forwardKinematics(pin_model, pin_data, q_mjc, dq_mjc)
            pin.computeAllTerms(pin_model, pin_data, q_mjc, dq_mjc)

            # ㅈㅋㅂㅇ 구하기
            J = pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.ReferenceFrame.LOCAL)

            # 외력 측정 & 어드미턴스 계산
            wrench = apply_deadband(get_zeroed_sensor_wrench())

            ddx_e_new = np.linalg.inv(M) @ (wrench - target_wrench - D @ dx_e - K @ x_e)
            dx_e += dt * 0.5 * (ddx_e + ddx_e_new)
            x_e += dt * 0.5 * (dx_e + dx_e)
            ddx_e = ddx_e_new

            # Jacobian Pseudo inverse로 Δq 계산
            dq_adm = np.linalg.pinv(J) @ dx_e

            # 중력 항 계산 
            mujoco.mj_rnePostConstraint(mjc_model, mjc_data)

            # 중력 보상 항 
            gravity_comp = mjc_data.qfrc_bias[:n_act] / mjc_model.actuator_gear[:, 0]
            gravity_comp[6] = gravity_comp[8] = 0.0  # rh_r1, rh_l1 인덱스

            if idx >= len(traj_t):
                traj_q = traj_q[-1]
            
            # 제어 입력 = 어드미턴스 속도 + 중력 보상 속도
            qvel_cmd = dq_adm + traj_qd[idx] + gravity_comp 
            #qvel_cmd = dq_adm + gravity_comp 


            # torque actuators 개수를 구해서
            #n_torque = mjc_model.nu_total - mjc_model.nu
            # ctrl[n_act:] 영역에 원하는 토크 신호를 넣어줄 수 있습니다.
            #mjc_data.ctrl[n_act:] = torque_command_for_fingers

            for i in range(pin_model.nv): # nv로 해야됨
                mjc_data.ctrl[i] = qvel_cmd[i]
            
            mujoco.mj_step(mjc_model, mjc_data)
            viewer.sync()
            idx+=1
            #time.sleep(dt)

except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")

sim_end = time.perf_counter()              
print(f"Simulation loop time: {sim_end - sim_start:.3f} s")
print("Total simulated trajectory duration:", traj_t[-1], "s")
