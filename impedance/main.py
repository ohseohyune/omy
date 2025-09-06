# main.py

import time
from . import config as C

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
    # === Initialize modules ===
    dxl = DxlIO()                   
    pin = GravityModel()            # Pinocchio 기반 중력 보상 모델
    ctrl = ImpedanceController()    # 임피던스 제어기
    est  = KtEstimator()            # Kt 토크상수 온라인 추정기

    # === Set initial targets to avoid sudden jumps ===
    q_init, dq_init = dxl.read_q_dq()
    ctrl.init_targets(q_init, dq_init)

    print("▶ Start OMY Impedance Control (+gravity, Kt online). Press Ctrl+C to stop.")

    t_prev = time.perf_counter()
    loop_counter = 0
    print_interval = int(30.0 / C.CONTROL_PERIOD)  # 30초 주기 출력용

    try:
        while True:
            # === Timing control ===
            t_prev = wait_until_next_cycle(t_prev, C.CONTROL_PERIOD)

            # === 1) Sensing ===
            joint_positions, joint_velocities = dxl.read_q_dq()

            # === 2) Gravity torque (from model) ===
            tau_gravity = pin.tau_g(joint_positions)

            # === 3) Impedance + Gravity compensation → Current commands ===
            current_cmd_01A, current_imp_01A = ctrl.step(
                joint_positions, joint_velocities, tau_gravity
            )

            # === 4) Send current commands ===
            dxl.send_goal_currents(current_cmd_01A)

            # 5) Update Kt estimator every 10 loops
            if loop_counter % 10 == 0:
                update_kt_estimator(est, dxl, tau_gravity, current_imp_01A)

            # 6) Print Kt summary every 30 seconds
            print_kt_summary(est, print_interval, loop_counter)


            loop_counter += 1

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")
    finally:
        dxl.close()

if __name__ == "__main__":
    main()
