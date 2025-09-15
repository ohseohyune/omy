#utils.py

import math, time
from .config import ENCODER_PULSES_PER_REV, MOTOR_GEAR_RATIO
from . import config as C


 
def unsigned_to_signed(val: int, bits: int) -> int:
    """Convert unsigned int (as returned by SDK) to signed two's complement."""
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val

def encoder_pulse_to_rad(pulse: int) -> float:
    """
    엔코더에서 읽은 펄스 값을 출력축 라디안 값으로 변환해줌. 
    pulse / ENCODER_PULSES_PER_REV : 모터축 회전 비율 => 몇 바퀴 돎? => 2pi 곱해서 라디안으로 변환 
    Gear ratio 고려해서 출력축 각도(rad)로 변환 
    """

    rad_motor = (pulse / ENCODER_PULSES_PER_REV) * (2.0 * math.pi)
    joint_angle_rad = rad_motor / MOTOR_GEAR_RATIO
    return joint_angle_rad

def dynamixel_vel_to_rads(vel_raw: int) -> float:
    """
    다이나믹셸의 present velocity 주소(548)에서 읽은 값은 단위가 0.01rev/min(모터축)임. 
    이걸 rad/s(출력축)으로 바꿔줌
    """
    rpm_motor = vel_raw * 0.01      
    motor_vel_rads = rpm_motor * (2.0 * math.pi / 60.0)
    return motor_vel_rads / MOTOR_GEAR_RATIO # 출력축 속도 = 모터축 속도 / 기어비  

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def wait_until_next_cycle(t_prev: float, CONTROL_PERIOD: float) -> float:
    """주기 CONTROL_PERIOD에 맞춰 루프를 쉬게 하고, 다음 기준 시각을 반환"""
    t_now = time.perf_counter()
    CONTROL_PERIOD_elapsed = t_now - t_prev  # 루프 시작한지 얼마나 됨?
    if CONTROL_PERIOD_elapsed < CONTROL_PERIOD:
        time.sleep(CONTROL_PERIOD - CONTROL_PERIOD_elapsed)
    return time.perf_counter()


def update_kt_estimator(estimator, dxl, tau_gravity, current_imp_raw):
    """
    Update Kt estimator with new sensor data.
    """
    measured_currents_A = dxl.read_currents_A() #[A]
    return estimator.update_online(tau_gravity, measured_currents_A, current_imp_raw)

def print_kt_summary(estimator, interval, loop_counter):
    """
    Print Kt estimation summary every `interval` loops.
    """
    if loop_counter % interval != 0:
        return
    summary = estimator.summary()
    print("\n=== Kt Estimation (avg ± std, N) ===")
    for motor_id in range(len(C.MOTOR_IDS)):
        stats = summary[motor_id]
        if stats["n"] > 0:
            print(
                f"Joint {motor_id+1}: "
                f"{stats['avg']:.4f} ± {stats['std']:.4f} Nm/A ({stats['n']})"
            )
        else:
            print(f"Joint {motor_id+1}: No data")
    print(f"TORQUE_CONSTANT_KT (configured): {C.TORQUE_CONSTANT_KT:.4f} Nm/A")
    print("====================================\n")
