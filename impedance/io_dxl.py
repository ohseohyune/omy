#io_dxl.py
# Dynamixel 입출력 래퍼 (SyncRead/SyncWrite, 현재값 읽기)

import struct
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
LEN_SYNC_READ_LEN      = 8   # vel(4) + pos(4)
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
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port: {C.SERIAL_PORT}")
        if not self.port.setBaudRate(C.DXL_BAUDRATE):
            raise RuntimeError(f"Failed to set baudrate: {C.DXL_BAUDRATE}")
        self.pkt  = PacketHandler(C.DXL_PROTOCOL_VERSION)

        for dxl_id in C.MOTOR_IDS:
            self._write(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, LEN_TORQUE_ENABLE)
            self._write(dxl_id, ADDR_OPERATING_MODE, OPERATING_MODE_CURRENT, LEN_OPERATING_MODE)
        for dxl_id in C.MOTOR_IDS:
            self._write(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, LEN_TORQUE_ENABLE)

        # Read motor's vel, pos, cur
        self.gsRead = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_VELOCITY, LEN_SYNC_READ_LEN)
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
            raise RuntimeError("GroupSyncRead txRxPacket failed")
        qs, dqs = [], []
        for dxl_id in C.MOTOR_IDS:
            vel_u32 = self.gsRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)
            pos_u32 = self.gsRead.getData(dxl_id, ADDR_PRESENT_POSITION, 4)
            vel_i32 = unsigned_to_signed(vel_u32, 32)
            pos_i32 = unsigned_to_signed(pos_u32, 32)
            dq = dynamixel_vel_to_rads(vel_i32, C.MOTOR_GEAR_RATIO)
            q  = encoder_pulse_to_rad(pos_i32, C.ENCODER_PULSES_PER_REV, C.MOTOR_GEAR_RATIO)
            qs.append(q); dqs.append(dq)
        return qs, dqs

    def read_currents_A(self) -> List[float]:
        scale = C.PRESENT_CURRENT_SCALE_A_PER_LSB()
        I = []
        for dxl_id in C.MOTOR_IDS:
            val, r, e = self.pkt.read2ByteTxRx(self.port, dxl_id, ADDR_PRESENT_CURRENT)
            if r != 0 or e != 0:
                raise RuntimeError(f"Read current error ID {dxl_id}")
            I.append(unsigned_to_signed(val, 16) * scale)
        return I

    def send_goal_currents(self, cur_list_01A: List[float]):
        for i, dxl_id in enumerate(C.MOTOR_IDS):
            cmd = int(clamp(cur_list_01A[i], -C.MOTOR_CURRENT_LIMITS[i], C.MOTOR_CURRENT_LIMITS[i]))
            param = struct.pack('<h', cmd)  # little-endian int16
            if not self.gsWrite_current.addParam(dxl_id, param):
                raise RuntimeError(f"GroupSyncWrite addParam failed for ID {dxl_id}")

        r = self.gsWrite_current.txPacket()
        if r != 0:
            raise RuntimeError(f"GroupSyncWrite txPacket failed: {r}")
        self.gsWrite_current.clearParam()

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


