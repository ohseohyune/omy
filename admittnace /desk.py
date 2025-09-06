# desk.py

import logging # 로깅을 위한 모듈(INFO, DEBUG, WARNING 등 로그 레벨 관리) Q
import time # 시간 측정, sleep() 등 시간 관련 유틸리티 제공 
import os # 파일 경로 조작, 환경 변수 읽기 등 운영체제 인터페이스 
from typing import List, Tuple # python 3 타입 힌드용 : 리스트 타입, 튜플 타입 등을 명시할 때 사용

import numpy as np # 수치 계산의 표준, 벡터, 행렬 연산에 필수 
import mujoco # 무조코 시뮬레이터의 핵심 API
import mujoco.viewer as mj_view # 무조코 시뮬레이션 뷰어(GUI) 모듈을 mj_view로 열 수 있음. 
import pinocchio as pin # 로봇 동역학 / 기구학 라이브러리 pinocchio
from pinocchio import SE3, Quaternion
from scipy.spatial.transform import Rotation as R


# SE3 : 3D 동차 변환 클래스 (회전 + 병진)
# Quaternion : 회전 표현용 쿼터니언 클래스 

# 로컬(자체) 모듈
from trajectory_generator_cartesian import (
    generate_wiping_trajectory,
    compute_ik,
    generate_quintic_trajectory
)

# 로깅(로그 출력)을 설정하는 부분
logging.basicConfig(level=logging.INFO) 
# 로깅의 기본 설정
# 여기서는 로그 레벨을 INFO로 지정해서, INFO 이상(WARNING,ERROR, CRITICAL) 수준의 메시지만 화면(표준 출력)으로 보이도록 함.  
logger = logging.getLogger(__name__)
# 로거 객체를 가져옴

def get_sensor_wrench(mjc_data: mujoco.MjData) -> np.ndarray:
    """첫 6개 센서 데이터를 [fx, fy, fz, tx, ty, tz]로 반환."""
    return mjc_data.sensordata[:6].copy()


def apply_deadband(
    wrench: np.ndarray,
    force_band: float = 0.1,
    torque_band: float = 0.05
) -> np.ndarray:
    """작은 힘/토크는 deadband 내에서 0으로 처리."""
    if np.linalg.norm(wrench[:3]) < force_band \
       and np.linalg.norm(wrench[3:]) < torque_band:
        return np.zeros(6)
    return wrench


def main() -> None:
    # Path 설정
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

    # 시뮬레이션 워밍업
    for _ in range(30):
        mujoco.mj_step(mjc_model, mjc_data)

    # Admittance 파라미터
    # 예: 1 Hz, ζ≈0.8 목표
    Kz = 500.0
    Mz = Kz / (2*np.pi*1.0)**2      # ≈ 75
    Dz = 2*0.8*np.sqrt(Mz*Kz)       # ≈ 700
    
    M = np.diag([2.5, 2.5, Mz, 0.2,  0.2,  0.2])
    D = np.diag([15.7,15.7,Dz , 2.5,  2.5,  2.5])
    K = np.diag([98.7,98.7, Kz, 7.9,  7.9,  7.9]) 

    # 상태 변수
    x_e = np.zeros(6)
    dx_e = np.zeros(6)
    ddx_e = np.zeros(6)
    target_wrench = np.zeros(6)
    wrench_f = None

    # 타이밍
    dt = mjc_model.opt.timestep
    #duration     = 1.5
    #n_steps      = 3000

    # 초기 EE 포즈
    q0 = np.zeros(pin_model.nq)
    pin.forwardKinematics(pin_model, pin_data, q0)
    pin.updateFramePlacements(pin_model, pin_data)
    
    # 현재 ee 자세 -> T0
    T0 = pin_data.oMf[frame_id].copy()

    # 초기 목표 EE 포즈 T1 (예시 오프셋 + 회전)
    #코드에서 frame_id를 "tcp_link"로 지정했기 때문에, T1은 tcp_link의 목표 자세(위치+방향)를 의미

    T1 = T0.copy()

    
    #T1.translation += np.array([0.12, -0.05, -0.45]) 0.2, -0.35, 0.4
    T1.translation = np.array([0.2, -0.35, 0.3])
    # ① 현재 회전 → Euler (degrees)
    rpy = R.from_matrix(T1.rotation).as_euler('XYZ', degrees=True)

    # ② 원하는 delta 각도 더하기
    delta = np.array([50.0, 0.0, 0.0])  # 이정도 각도가 ik가 풀리는 최대 각도인듯....
    rpy_new = rpy + delta

    # ③ Euler → 회전행렬
    R_new = R.from_euler('XYZ', rpy_new, degrees=True).as_matrix()

    # ④ T1.rotation 에 갱신
    T1.rotation = R_new
    
    #T_vertical = SE3(T1.rotation, T0.translation)

    # 초기 Cartesian 궤적 (도달용)
    cartesian_poses, traj_times = generate_quintic_trajectory(
        T0, T1, duration=3.0, n_steps=500
    )
    # traj_times는 0 -> duration 구간을 균일 분할한 1D 시간 배열
    # 이제 cartesian_poses를 따라가면서 실시한 제어를 수행함. 

    # PD, gravity compensation 등
    gear = mjc_model.actuator_gear[:, 0];  gear[gear==0] = 1.0
    Kp = 40.0
    Kd = 0.5
    Ki = 1.0
    q_err_integral = np.zeros(pin_model.nq)
    sensor_zero = get_sensor_wrench(mjc_data)

    # wiping 모션 파라미터
    force_threshold = -20   # 접촉 감지 임계값 (N)
    Fz_wipe         = 30.0    # 유지할 접촉력 (N)
    wipe_duration   = 0.6     # 한 세그먼트 지속 시간 (s)
    wipe_steps      = 500    # 세그먼트별 스텝 수
    wipe_loops      = 3       # 사각형 반복 횟수
    x_amp = 0.5
    y_amp = 0.0

    # 참조 joint 상태
    q_ref_prev  = q0.copy()
    dq_ref_prev = np.zeros_like(q0)
    trajectory_frozen = False
    idx = 0
    start_time = time.time()
    force_mask = np.array([1,1,1,1,1,1])
    fc = 20.0
    beta = np.exp(-2*np.pi*fc*dt)   # 0<beta<1
    phase = "APPROACH"

    # ========================== 준비 단계 코드 시작 ==========================
    logger.info("Moving to the initial ready pose...")
    ready_duration = 2.0  # 2초 동안 준비 자세로 이동
    ready_start_time = mjc_data.time
    q_ready_target = np.zeros(pin_model.nq) # 목표는 모든 관절 0도

    # 뷰어 없이 준비 자세로 이동 (더 빠름)
    while mjc_data.time - ready_start_time < ready_duration:
        q_mjc  = mjc_data.qpos[:pin_model.nq].copy()
        dq_mjc = mjc_data.qvel[:pin_model.nv].copy()
        
        # 간단한 PD 제어로 목표 자세 유지
        q_err = q_ready_target - q_mjc
        # 목표 속도는 0
        dq_err = 0 - dq_mjc

        # 여기서 사용하는 Kp, Kd는 준비 단계용 (원래 값과 같아도 무방)
        Kp_ready = 20.0
        Kd_ready = 2.0
        
        qvel_cmd = Kp_ready * q_err + Kd_ready * dq_err
        mjc_data.ctrl[:pin_model.nv] = qvel_cmd
        mujoco.mj_step(mjc_model, mjc_data)

    logger.info("Ready pose reached. Calibrating sensor.")
    # 로봇이 완벽하게 정지한 상태에서 센서 영점 다시 측정
    sensor_zero = get_sensor_wrench(mjc_data)

    try:
        with mj_view.launch_passive(mjc_model, mjc_data) as viewer:
            while viewer.is_running() :

               
                if idx < len(cartesian_poses) - 1:
                    idx += 1
                else:
                    if phase == "APPROACH":
                        idx = len(cartesian_poses) - 1
                    else:
                        idx = 0

                t_real = time.time() - start_time

                # 현재 joint 상태
                q_mjc  = mjc_data.qpos[:pin_model.nq].copy()
                dq_mjc = mjc_data.qvel[:pin_model.nv].copy()

                # Pinocchio 업데이트
                pin.forwardKinematics(pin_model, pin_data, q_mjc, dq_mjc)
                pin.computeAllTerms(pin_model, pin_data, q_mjc, dq_mjc)

                # 센서 wrench & deadband
                wrench = apply_deadband(get_sensor_wrench(mjc_data) - sensor_zero)

                #print("\n",wrench[2])
                
                R_ee = pin_data.oMf[frame_id].rotation
                wrench_w = np.hstack([R_ee@wrench[:3],R_ee@wrench[3:]])
                # 1차 저역통과
                if wrench_f is None:
                    wrench_f = wrench_w.copy()                 # ← 첫 스텝 평활 초기화
                else:
                    wrench_f = beta*wrench_f + (1-beta)*wrench_w

                # 1차 저역통과, 컷 20Hz
                
                wrench_f = beta* wrench_f + (1-beta)*wrench_w
                # 이후 wrench_w 대신 wrench_f 사용

                # 접촉 감지 시 wiping 궤적으로 전환
                if (phase == "APPROACH") and (wrench_f[2] < force_threshold) :
                    logger.info("Desk contact detected (fz=%.2f N). Start wiping.", wrench[2])
                    T_now = pin_data.oMf[frame_id].copy()
                    #T_des.translation[2] = T_now.translation[2] 
                    # Cartesian wiping 궤적 생성 (pure SE3)
                    wiping_poses, wiping_times = generate_wiping_trajectory(
                        pin_model, pin_data, frame_id,
                        q_ref_prev, T_now,
                        x_amp, y_amp,
                        wipe_duration, wipe_steps, wipe_loops
                    )
                    
                    cartesian_poses   = wiping_poses
                    #traj_times        = wiping_times
                    idx               = 0
                    phase = "WIPE" 
                    trajectory_frozen = True
                    target_wrench[2]  = Fz_wipe
                    q_err_integral[:] = 0.0

                    force_mask = np.array([0,0,1,0,0,0])
                    Kp = 20.0
                    Kd= 0.5
                    Ki = 0.3
                    
                # Admittance 식 
                ddx_e_new = np.linalg.inv(M) @ (
                    (wrench_f - target_wrench)*force_mask - D @ dx_e - K @ x_e
                )
                dx_e  += dt * 0.5 * (ddx_e + ddx_e_new)
                x_e   += dt * dx_e
                ddx_e = ddx_e_new

                # 원하는 Cartesian pose
                T_des = cartesian_poses[idx]
                if trajectory_frozen:
                    T_now = pin_data.oMf[frame_id].copy()
                    T_des.translation[2] = T_now.translation[2]

                # IK: q_ref 계산
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
                
                # Joint‐space admittance vel.
                J = pin.getFrameJacobian(
                    pin_model, pin_data, frame_id, pin.ReferenceFrame.WORLD
                )
                dq_adm = np.linalg.pinv(J) @ dx_e

                # Gravity compensation
                #mujoco.mj_rnePostConstraint(mjc_model, mjc_data)
                #gravity_comp = mjc_data.qfrc_bias[:pin_model.nv] / gear

                # Joint PD
                q_err  = q_ref  - q_mjc
                q_err_integral += q_err * dt
                dq_err = dq_ref - dq_mjc

                # 최종 토크/vel 커맨드
                alpha = 1.0 # PID 블랜딩 => 경로 추종 이득이 어드미턴스를 이기지 않게 조정
                qvel_cmd = (
                    dq_adm
                    + (Kp * q_err # 목표 지점과의 거리에 비례하여 속도를 조절한다 .
                    + Ki * q_err_integral
                    + Kd * dq_err) * alpha
                )
                mjc_data.ctrl[:pin_model.nv] = qvel_cmd

                mujoco.mj_step(mjc_model, mjc_data)
                viewer.sync()

                # 다음 스텝 준비
                q_ref_prev  = q_ref
                dq_ref_prev = dq_ref

                # real-time pacing
                #time.sleep(max(0.0, dt - (time.time() - (start_time + t_real))))

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    finally:
        logger.info("Simulation ended.")


if __name__ == "__main__":
    main()
