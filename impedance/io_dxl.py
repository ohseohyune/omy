#io_dxl.py
# Dynamixel 입출력 래퍼 (SyncRead/SyncWrite, 현재값 읽기)

import struct, sys
from typing import List, Tuple
from . import config as C

from dynamixel_sdk import (
    PortHandler, 
    PacketHandler, 
    GroupSyncRead, 
    GroupSyncWrite 
)

from .utils import (
    unsigned_to_signed, 
    encoder_pulse_to_rad, 
    dynamixel_vel_to_rads, 
    clamp
)


# ===== Control Table =====
"""
Operating Mode Setting 

0 : Current Control Mode
1 : Velocity Control Mode
3(Default) : Position Control Mode
"""
ADDR_OPERATING_MODE    = 33
ADDR_TORQUE_ENABLE     = 512
ADDR_GOAL_CURRENT      = 526
ADDR_PRESENT_CURRENT   = 546
ADDR_PRESENT_VELOCITY  = 548
ADDR_PRESENT_POSITION  = 552

LEN_TORQUE_ENABLE      = 1
LEN_OPERATING_MODE     = 1
LEN_GOAL_CURRENT       = 2
LEN_SYNC_READ_LEN      = 8   # vel(4byte) + pos(4)

OPERATING_MODE_CURRENT = 0   # Current Control Mode
TORQUE_ENABLE          = 1
TORQUE_DISABLE         = 0


class DxlIO:

    def __init__(self):
        
        """
        1) Open Port -> Set Baudrate
        2) Torque OFF → OperatingMode(Current) → Torque ON
        
        GroupSyncRead/Write 객체 Initialized to read and write several motor data simultaneously.
        """
        
        self.port = PortHandler(C.SERIAL_PORT)
        if self.port.openPort():
            print("Succeeded to open the port")
        else:
            raise RuntimeError(f"Failed to open port: {C.SERIAL_PORT}")
        
        if self.port.setBaudRate(C.DXL_BAUDRATE):
            print("Succeeded to change the baudrate!")
        else:
            raise RuntimeError(f"Failed to set baudrate: {C.DXL_BAUDRATE}")
            
        
        self.pkt  = PacketHandler(C.DXL_PROTOCOL_VERSION) 
        # pkt 객체는 실제 데이터를 직접 보내는 게 아니라, **통신 규칙(프로토콜)**을 맞춰주는 번역기 같은 역할

        for dxl_id in C.MOTOR_IDS:
            self._write(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, LEN_TORQUE_ENABLE)
            self._write(dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_CURRENT, LEN_OPERATING_MODE)
        for dxl_id in C.MOTOR_IDS:
            self._write(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, LEN_TORQUE_ENABLE)

        # Read motor's vel, pos, cur
        """
        포트(self.port)를 통해 데이터를 보내고
        패킷 처리(self.pkt)를 이용해서 요청/응답을 올바른 형식으로 주고받음
        """

        self.gsRead = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_VELOCITY, LEN_SYNC_READ_LEN)
        # GroupSyncRead 클래스의 instance 이고, 이 instance를 self.gsRead라는 이름(멤버 변수)에 저장
        # 이후에는 self.gsRead를 통해서 그 객체의 기능(메서드)를 계속 쓸 수 있음. 
        for dxl_id in C.MOTOR_IDS:
            if not self.gsRead.addParam(dxl_id):
                raise RuntimeError(f"GroupSyncRead addParam failed for ID {dxl_id}")

        # Write command to motors
        self.gsWrite_current = GroupSyncWrite(self.port, self.pkt, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)

    def _write(self, id_, addr, value, byte_len):
        if byte_len == 1:
            r, e = self.pkt.write1ByteTxRx(self.port, id_, addr, value)
        elif byte_len == 2:
            r, e = self.pkt.write2ByteTxRx(self.port, id_, addr, value)
        elif byte_len == 4:
            r, e = self.pkt.write4ByteTxRx(self.port, id_, addr, value)
        else:
            raise ValueError("Unsupported byte length")
        if r != 0 or e != 0:
            raise RuntimeError(f"Write error: id={id_}, addr={addr}, comm={r}, err={e}")

    def read_q_dq(self) -> Tuple[List[float], List[float]]:
        if self.gsRead.txRxPacket() != 0: 
            # GroupSyncRead 객체(gsRead)를 통해 모터에서 데이터를 읽어옴.0이여야 성공
            # GroupSyncRead라는 클래스 안에 있는 txRxPacket 매서드를 사용
            raise RuntimeError("GroupSyncRead txRxPacket failed")
        qs, dqs = [], []
        for dxl_id in C.MOTOR_IDS:
            """
            txRxPacket()으로 모터들에게 데이터를 요청하고 나면, 모터별 응답이 GroupSyncRead 객체 내부 버퍼에 저장
            getData(dxl_id, address, length)는 그 버퍼에서 특정 모터(dxl_id)의 특정 주소(address)의 데이터를 꺼내오는 메서드"""
           
            vel_u32 = self.gsRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)
            pos_u32 = self.gsRead.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
            
            vel_i32 = unsigned_to_signed(vel_u32, 32)
            pos_i32 = unsigned_to_signed(pos_u32, 32)
            
            dq = dynamixel_vel_to_rads(vel_i32)
            q  = encoder_pulse_to_rad(pos_i32)
            
            qs.append(q); dqs.append(dq)
        
        return qs, dqs

    def read_currents_A(self) -> List[float]:
        motor_current_scale = C.PRESENT_CURRENT_SCALE_A_PER_LSB # 1 raw == 0.01 A 
        motor_current = []
        for dxl_id in C.MOTOR_IDS:
            raw_cur_value, packet_success_result, error = self.pkt.read2ByteTxRx(self.port, dxl_id, ADDR_PRESENT_CURRENT)
            if packet_success_result != 0 or error != 0:
                raise RuntimeError(f"Read current error ID {dxl_id}")
            motor_current.append(unsigned_to_signed(raw_cur_value, 16) * motor_current_scale) # raw -> A , present current : 2byte -> 16bit 
        return motor_current # [A]

    def send_goal_currents(self, goal_currents_raw: List[float]): #[raw]
        for idx, motor_id in enumerate(C.MOTOR_IDS):
            clamped_current_goal = int(clamp(goal_currents_raw[idx], -C.MOTOR_CURRENT_LIMITS[idx], C.MOTOR_CURRENT_LIMITS[idx]))
            
            """
            Goal Current 주소는 2byte 크기(int16)
            모터가 기다리는 raw 데이터 형태는 [0x78 , 0x00] 처럼 120을 little-endian 16비트로 표현한 값
            struct.pack(format, value)은 파이썬 값을 원하는 이진 데이터로 바꿔 줘.

            긍게 우리가 보는 전류 값을 
            모터가 이해하는 raw값으로 변환해주는겨 

            raw == 1이면 0.01[A]
            """
            param_bytes = struct.pack('<h', clamped_current_goal)  # little-endian int16
            
            """
            [ addParam ]
            특정 모터(motor_id)에 쓸 값을 추가(버퍼에 등록)
            param_bytes는 실제로 쓰고 싶은 값(전류 목표치)을 바이트 배열 형태로 변환한 
            여러 모터에 대해 addParam을 반복하면, 내부 버퍼에 모두 저장됨
            """
            if not self.gsWrite_current.addParam(motor_id, param_bytes):
                raise RuntimeError(f"GroupSyncWrite addParam failed for ID {motor_id}")

        # Transmit packet to all motors / txPacket() → 등록된 모든 모터에 한 번에 전송
        tx_result = self.gsWrite_current.txPacket()
        if tx_result != 0:
            raise RuntimeError(f"GroupSyncWrite txPacket failed: {tx_result}")
        
        self.gsWrite_current.clearParam() # clearParam() → 버퍼 초기화

    def close(self):
        try:
            self.send_goal_currents([0]*len(C.MOTOR_IDS))
        except Exception:
            pass
        for dxl_id in C.MOTOR_IDS:
            try:
                self._write(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, LEN_TORQUE_ENABLE)
            except Exception:
                pass
        self.port.closePort()


