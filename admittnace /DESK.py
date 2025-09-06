# DESK.py

import logging
import mujoco 
import mujoco.viewer as mj_view
import time
import os 
import pinocchio as pin
import numpy as np


from scipy.spatial.transform import Rotation as R
# 로컬(자체) 모듈
from trajectory_generator_cartesian import (
    generate_wiping_trajectory,
    compute_ik,
    generate_quintic_trajectory
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



model_path = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/"
urdf_model = os.path.join(model_path, "robot.urdf")
scene_xml  = os.path.join(model_path, "scene.xml")

# Pinocchio 초기화
pin_model = pin.buildModelFromUrdf(urdf_model)
pin_data  = pin_model.createData()
frame_id  = pin_model.getFrameId("tcp_link")

# MuJoCo 초기화
mjc_model = mujoco.MjModel.from_xml_path(scene_xml)
mjc_data  = mujoco.MjData(mjc_model)

target_site_id = mujoco.mj_name2id(mjc_model, mujoco.mjtObj.mjOBJ_SITE, "target_marker")

# Admittance 파라미터 (타입 3: 무거운 볼링공)
M_trans = 20.0 # 이동 파트 질량 (무겁게)
M_rot   = 2.0  # 회전 파트 질량 (무겁게)
D_trans = 100.0 # 큰 질량을 제어하기 위한 높은 댐핑
D_rot   = 10.0  # 큰 질량을 제어하기 위한 높은 댐핑

M = np.diag([M_trans, M_trans, M_trans, M_rot, M_rot, M_rot])
D = np.diag([D_trans, D_trans, D_trans, D_rot, D_rot, D_rot])
#K = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
K = np.diag([98.7,98.7, 500, 7.9,  7.9,  7.9]) 

Kp = 10.0
Kd = 1.0
Ki = 0.0 

def get_sensor_wrench(mjc_data: mujoco.MjData) -> np.ndarray:
    return mjc_data.sensordata[:6].copy()

x_e = np.zeros(6)
dx_e = np.zeros(6)
ddx_e = np.zeros(6)
target_wrench = np.zeros(6)
wrench_f = None
force_mask = np.array([1,1,1,1,1,1])

T0 = pin_data.oMf[frame_id].copy()
T1 = T0.copy()
T1.translation = np.array([0.2, -0.35, 0.4])


rpy = R.from_matrix(T1.rotation).as_euler('XYZ', degrees=True)
delta = np.array([0.0, 0.0, 0.0]);  rpy_new = rpy + delta
R_new = R.from_euler('XYZ', rpy_new, degrees=True).as_matrix()

T1.rotation = R_new


cartesian_poses, traj_times = generate_quintic_trajectory(
    T0, T1, duration=1.0, n_steps=500
)


# 시뮬레이션 워밍업
for _ in range(30):
    mujoco.mj_step(mjc_model, mjc_data)

sensor_zero = get_sensor_wrench(mjc_data)

q0 = np.zeros(pin_model.nq)
q_ref_prev  = q0.copy()
dq_ref_prev = np.zeros_like(q0)
q_err_integral = np.zeros_like(q0)
idx = 0
dt = mjc_model.opt.timestep

task_finished = False

try:
    with mj_view.launch_passive(mjc_model, mjc_data) as viewer:
        while viewer.is_running():

            mjc_data.site_xpos[target_site_id] = T1.translation

            if not task_finished:
                if idx < len(cartesian_poses) - 1:
                    idx += 1
                else:
                    task_finished = True
                    print("\n>>Trajectory finished.")
           
            # 현재 joint 상태
            q_mjc  = mjc_data.qpos[:pin_model.nq].copy()
            dq_mjc = mjc_data.qvel[:pin_model.nv].copy()
            

           
            pin.forwardKinematics(pin_model, pin_data, q_mjc, dq_mjc)
            pin.computeAllTerms(pin_model, pin_data, q_mjc, dq_mjc)



            gravity_comp_torque = pin.computeGeneralizedGravity(pin_model, pin_data, q_mjc)
           
            
            wrench = get_sensor_wrench(mjc_data) - sensor_zero

            ddx_e_new = np.linalg.inv(M) @ (
                (wrench - target_wrench)*force_mask - D @ dx_e - K @ x_e
            )
            dx_e  += dt * 0.5 * (ddx_e + ddx_e_new)
            x_e   += dt * dx_e
            ddx_e = ddx_e_new

            T_des = cartesian_poses[idx]

            try:
                success, q_ref = compute_ik(
                    pin_model, pin_data, frame_id, T_des, q_ref_prev
                )
                if not success:
                    logger.warning("IK failed to converge at step %d", idx)
                    q_ref = q_ref_prev
            except Exception as e:
                logger.error("IK exception at step %d: %s", idx, e)
                q_ref = q_ref_prev

            dq_ref = (q_ref - q_ref_prev) / dt

            q_err  = q_ref  - q_mjc
            q_err_integral += q_err * dt
            dq_err = dq_ref - dq_mjc

            J = pin.getFrameJacobian(
                pin_model, pin_data, frame_id, pin.ReferenceFrame.LOCAL
            )
            dq_adm = np.linalg.pinv(J) @ dx_e
            
            #gravity_comp_vel = gravity_comp_torque*2.435  # <--- 경험적 튜닝 값 (0.05 ~ 0.2)
            qvel_cmd = dq_adm + Kp * q_err + Ki * q_err_integral + Kd * dq_err #+ gravity_comp_vel

            mjc_data.ctrl[:pin_model.nv] = qvel_cmd 
           
            # 현재값 목표값 비교 
            T_now = pin_data.oMf[frame_id]
            pos_now = T_now.translation
            rpy_now_deg = R.from_matrix(T_now.rotation).as_euler('xyz', degrees=True)

            # 목표 위치 및 방향(오일러 각) 추출
            pos_des = T_des.translation
            rpy_des_deg = R.from_matrix(T_des.rotation).as_euler('xyz', degrees=True)
            
            print(f"--- Step {idx} ---")
            print(f" CURR POSE: Pos=[{pos_now[0]:.3f}, {pos_now[1]:.3f}, {pos_now[2]:.3f}], RPY=[{rpy_now_deg[0]:.1f}, {rpy_now_deg[1]:.1f}, {rpy_now_deg[2]:.1f}]")
            print(f" TARG POSE: Pos=[{pos_des[0]:.3f}, {pos_des[1]:.3f}, {pos_des[2]:.3f}], RPY=[{rpy_des_deg[0]:.1f}, {rpy_des_deg[1]:.1f}, {rpy_des_deg[2]:.1f}]")


            mujoco.mj_step(mjc_model, mjc_data) 
            viewer.sync()
            q_ref_prev  = q_ref
            dq_ref_prev = dq_ref

            time.sleep(0.001) 
except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")