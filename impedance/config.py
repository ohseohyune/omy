#config.py 

from typing import List 

SERIAL_PORT = "/dev/ttyUSB0"   
DXL_BAUDRATE    = 6250000          
DXL_PROTOCOL_VERSION = 2.0
MOTOR_IDS: List[int] = [1, 2, 3, 4, 5, 6]

CONTROL_PERIOD = 0.004 # 샘플링 타임 : 250Hz : 초당 250번 : 센서 데이터를 읽고 → 제어기 계산 → 액추에이터로 신호 보내는 주기 [Q]
VEL_FILTER_ALPHA = 0.1 # 내부적으로 필터링된 값을 읽어오니까, 일단 0으로 해두고 노이즈가 큰 것 같으면(?) 수정해주기 ! 
PRESENT_CURRENT_SCALE_A_PER_LSB = 0.01 # [A]

MOTOR_GEAR_RATIO = 99.0
ENCODER_PULSES_PER_REV = 524_288  # 19-bit

IMPEDANCE_KP = [10.0, 10.0, 10.0, 6.0, 6.0, 6.0]   # [Nm / rad] -> 1rad의 오차 -> 전류 0.4A
IMPEDANCE_KD = [ 3.0,  3.0,  3.0, 2.0, 2.0, 2.0]   # [Nm / (rad/s)]

# Current limits per joint [LSB = 0.01A]
MOTOR_CURRENT_LIMITS = [800, 800, 800, 800, 800, 800] # 최대는 2240(080) / 2080(070)

XML_PATH = "scene.xml" # MJCF 파일 경로  
URDF_PATH = "omy.urdf" # 변환된 URDF 파일 저장 경로 

# // PIN: 전류 변환 파라미터
TORQUE_CONSTANT_KT = 2.3   # [N·m/A]  
MECH_EFFICIENCY  = 0.70   # 감속/전달 효율(0.8~0.95), 보수적 가정 / torque output = torque motor * MECH_EFFICIENCY (열손실등의 고려)

# Kt 추정을 위한 변수들
KT_ESTIMATION_WINDOW_SIZE = 100  # 추정에 사용할 데이터 윈도우 크기 

