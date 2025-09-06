#config.py 

from typing import List 

SERIAL_PORT = "/dev/ttyUSB0"   
DXL_BAUDRATE    = 6250000          
DXL_PROTOCOL_VERSION = 2.0
MOTOR_IDS: List[int] = [1, 2, 3, 4, 5, 6]

CONTROL_PERIOD = 0.004 # 샘플링 타임 : 250Hz [ Q 질문 ]
VEL_FILTER_ALPHA = 0.25

MOTOR_GEAR_RATIO = 99.0
ENCODER_PULSES_PER_REV = 524_288  # 19-bit

IMPEDANCE_KP = [10.0, 10.0, 10.0, 6.0, 6.0, 6.0]   # [0.01A / rad] -> 1rad의 오차 -> 전류 0.4A
IMPEDANCE_KD = [ 3.0,  3.0,  3.0, 2.0, 2.0, 2.0]   # [0.01A / (rad/s)]

# Current limits per joint [LSB = 0.01A]
MOTOR_CURRENT_LIMITS = [200, 200, 200, 200, 150, 120]

XML_PATH = "" # MJCF 파일 경로 
URDF_PATH = "" # 변환된 URDF 파일 저장 경로 

# // PIN: 전류 변환 파라미터
TORQUE_CONSTANT_KT = 1.33   # [N·m/A]  (Y-series YM070 근사값; 현장 값으로 보정 권장) [Q2] 
MECH_EFFICIENCY  = 0.90   # 감속/전달 효율(0.8~0.95), 보수적 가정

# Kt 추정을 위한 변수들
KT_ESTIMATION_WINDOW_SIZE = 100  # 추정에 사용할 데이터 윈도우 크기

