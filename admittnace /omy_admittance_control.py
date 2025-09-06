#omy_admittance_control.py

import mujoco
import mujoco.viewer as mj_view
import time
import math
import os
import numpy as np
import pinocchio as pin
from pinocchio import buildModelFromUrdf

#=======
# step 1: 피노키오로 URDF 로드
#=======

# URDF 경로 
model_path = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/"
urdf_filename = "robot.urdf"
urdf_model_path = os.path.join(model_path, urdf_filename)

# 모델과 데이터 로드
pin_model = buildModelFromUrdf(urdf_model_path)
#pin_model.gravity.linear = np.array([0, 0, -9.81])  # ⬅️ 이거 중요!
pin_data = pin_model.createData()

#=======
# step 2: 무조코 모델 로드 + 데이터 준비 
#=======

mjc_model = mujoco.MjModel.from_xml_path(model_path + "scene.xml")
mjc_data = mujoco.MjData(mjc_model)

# ====================================
# Step 3: 어드미턴스 파라미터 정의 & 상태 변수 초기화
# ====================================

M = np.diag([5, 5, 5, 0.5, 0.5, 0.5]) * 0.5 #6 by 6 대각행렬
D = np.diag([10, 10, 10, 1, 1, 1]) * 0.5
K = np.zeros((6, 6))


dt = mjc_model.opt.timestep
x_e = np.zeros(6)
dx_e = np.zeros(6)
ddx_e = np.zeros(6)
target_wrench = np.zeros(6) # 로봇의 엔드이펙터에 작용하기를 원하는 힘과 모멘트의 값 
#target_wrench = np.array([0,0,80,0,0,0])
# 보통 [Fx, Fy, Fz, Tx, Ty, Tz] 형태로 구성된 6차원 벡터

dq_g = np.zeros(pin_model.nv)  # 중력 보상용 속도 누적 변수
#dq_g = np.zeros(6)

# 중력 보상 비례 이득 설정 
gravity_gain = 2.4

# ====================================
# Step 4: 센서 관련 함수 정의
# ====================================

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
    
# ====================================
# Step 5: 시뮬레이션 워밍업 & 센서 제로화 & 초기 목표값 설정
# ====================================    
    
# 시뮬레이션 워밍업
for _ in range(100):
    mujoco.mj_step(mjc_model, mjc_data)

# 센서 제로화
sensor_zero = get_sensor_wrench()

# 초기 관절 위치 목표값
qpos_target = np.copy(mjc_data.qpos[:pin_model.nq])

# 시뮬레이션 시작
start_time = time.time()

try:
    with mj_view.launch_passive(mjc_model, mjc_data) as viewer:
        while viewer.is_running():
            t = time.time() - start_time

            # Step 2 (루프 안): MuJoCo 현재 상태 → Pinocchio에 적용
            q_mjc = np.copy(mjc_data.qpos[:pin_model.nq])
            dq_mjc = np.copy(mjc_data.qvel[:pin_model.nv])
            
            # Pinocchio 동역학 데이터 업데이트 (반드시 먼저)
            pin.forwardKinematics(pin_model, pin_data, q_mjc, dq_mjc)
            pin.computeAllTerms(pin_model, pin_data, q_mjc, dq_mjc)

            # Step 3 (루프 안): Jacobian으로 Δq 계산 (어드미턴스 출력 변환)
            frame_name = "tcp_link"  # 엔드 이펙터 링크 이름
            frame_id = pin_model.getFrameId(frame_name)
            
            # 반드시 존재하는지 확인
            if frame_id == len(pin_model.frames):
                raise ValueError(f"[ERROR] Frame '{frame_name}' not found in URDF.")
                
        
            J = pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.ReferenceFrame.LOCAL)

            # 외력 측정 & 어드미턴스 계산
            wrench = apply_deadband(get_zeroed_sensor_wrench())

            ddx_e_new = np.linalg.inv(M) @ (wrench - target_wrench - D @ dx_e - K @ x_e)
            dx_e += dt * 0.5 * (ddx_e + ddx_e_new)
            x_e += dt * 0.5 * (dx_e + dx_e)
            ddx_e = ddx_e_new
             
            # Jacobian Pseudo-inverse로 Δq 계산
            delta_q = np.linalg.pinv(J) @ x_e
          

            # 중력 보상용 관절 가속도 계산 (중력만으로)
            tau_zero = np.zeros(pin_model.nv)
            #print(f"tau_zero shape = {tau_zero.shape}, pin_model.nv = {pin_model.nv}")


            #ddq_gravity = pin.forwardDynamics(pin_model, pin_data, q_mjc, dq_mjc, tau_zero)

            # 중력 속도 누적 (적분)
            #dq_g += dt * ddq_gravity
            
            # 중력 토크 → velocity actuator 용 보상 속도
            tau_g = pin.computeGeneralizedGravity(pin_model, pin_data, q_mjc)  

            dq_g = gravity_gain * tau_g
            qvel_cmd = delta_q + dq_g

            for i in range(pin_model.nv): #nv로 바꿔야함
                mjc_data.ctrl[i] = qvel_cmd[i]


            mujoco.mj_step(mjc_model, mjc_data)
            viewer.sync()
            time.sleep(0.001)

except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")
