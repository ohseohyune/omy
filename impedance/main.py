# main.py

import time
from . import config as C
from .mujoco_test import MujocoTest

from .io_dxl import DxlIO
from .model_pin import GravityModel
from .control import ImpedanceController
from .estimate import KtEstimator

from .utils import (
    wait_until_next_cycle,
    update_kt_estimator,
    print_kt_summary,
)

def main():
    """
    dxl: dxl 객체 생성 + def __init__(self): 실행 

    1) PortHandler와 PacketHandler 준비
    2) Current Mode ON
    3) GroupSyncRead와 GroupSyncWrite 객체 생성 및 ID 등록

    """
    # === Initialize modules / 객체들 ===
    dxl =  DxlIO() #MujocoTest() #               
    pin = GravityModel()            # Pinocchio 기반 중력 보상 모델(URDF 모델이 있나 check , 없으면 MJCF => URDF)
    ctrl = ImpedanceController()    # 임피던스 제어기
    est  = KtEstimator()            # Kt 토크상수 온라인 추정기

    # === Set initial targets to avoid sudden jumps ===
    q_init, dq_init = dxl.read_q_dq()
    ctrl.init_targets(q_init, dq_init) # 현재 위치와 속도를 desired로 만들어줌

    print("▶ Start OMY Impedance Control (+gravity, Kt online). Press Ctrl+C to stop.")

    #nstep = 4 # for mujoco

    kt_estimates = [C.TORQUE_CONSTANT_KT] * len(C.MOTOR_IDS)
    t_prev = time.perf_counter() # 프로그램이 실행된 이후 경과된 시간을 초 단위(float) 로 반환합니다.
    loop_counter = 0
    print_interval = int(3.0 / C.CONTROL_PERIOD)  # 5초 주기 로그 출력용


    try:
        while True:
            # === Timing control ===
            t_prev = wait_until_next_cycle(t_prev, C.CONTROL_PERIOD)

            # === 1) Sensing ===
            joint_positions, joint_velocities = dxl.read_q_dq()

            # === 2) Gravity torque (from model) ===
            tau_gravity = pin.gravity_comp_torque(joint_positions)   

            if loop_counter % 250 == 0:
                print(f"\ntau_gravity : {tau_gravity}")         

            # === 3) Impedance + Gravity compensation → Current commands ===
            current_cmd_raw, current_imp_raw = ctrl.compute_impedance_with_gravity(
                joint_positions, joint_velocities, tau_gravity, kt_estimates
            )

            # 5) Update Kt estimator every 10 loops
            if loop_counter % 10 == 0:
                kt_estimates = update_kt_estimator(est, dxl, tau_gravity, current_imp_raw)

            # === 4) Send current commands ===
            dxl.send_goal_currents(current_cmd_raw)
            
            #if loop_counter % 250 == 0: #1초마다 출력 
            #    print(f"\ncur_cmd_raw :{current_cmd_raw} ")
            #    print(f"cur_imp_raw : {current_imp_raw}")

            # MuJoCo 한 주기 진행 (CONTROL_PERIOD 동안 시뮬레이션 시간 흘리기) / for mujoco
            #nstep = max(1, int(round(C.CONTROL_PERIOD / dxl.model.opt.timestep)))
            #dxl.step(nstep)            

            # 6) Print Kt summary
            print_kt_summary(est, print_interval, loop_counter)

            loop_counter += 1

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")
    finally:
        dxl.close()

if __name__ == "__main__":
    main()
