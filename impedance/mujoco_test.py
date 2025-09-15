# mujoco_test.py

import numpy as np
import mujoco
import mujoco.viewer as mjv
from typing import List, Tuple
from . import config as C

class MujocoTest:


    def __init__(self, model_xml: str = C.XML_PATH):
        # MuJoCo model 로드
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)

        # MuJoCo viewer 실행 (원하는 경우)
        self.viewer = mjv.launch_passive(self.model, self.data)

        self.motor_ids = C.MOTOR_IDS

    def step(self, n : int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def read_q_dq(self) -> Tuple[List[float], List[float]]:
        """MuJoCo에서 q, dq 읽기"""
        qs = self.data.qpos[:len(self.motor_ids)].tolist()
        dqs = self.data.qvel[:len(self.motor_ids)].tolist()
        return qs, dqs

    def read_currents_A(self) -> List[float]:
        """
        Dynamixel에서는 센서에서 읽지만,
        여기서는 MuJoCo actuator force를 단순히 '전류에 비례한다'고 가정.
        """
        # MuJoCo에서 actuator force를 가져옴
        forces = self.data.qfrc_actuator[:len(self.motor_ids)]
        # 단순히 Kt = C.TORQUE_CONSTANT_KT로 역변환해서 전류로 가정
        currents = (forces / (C.TORQUE_CONSTANT_KT * C.MOTOR_GEAR_RATIO * C.MECH_EFFICIENCY)).tolist()
        return currents

    def send_goal_currents(self, goal_currents_raw: List[float]):
        """
        raw (0.01A 단위)를 받아 MuJoCo actuator torque로 변환해줌.
        """
        for idx, motor_id in enumerate(self.motor_ids):
            current_A = goal_currents_raw[idx] * 0.01
            torque = current_A * C.TORQUE_CONSTANT_KT * C.MOTOR_GEAR_RATIO * C.MECH_EFFICIENCY
            self.data.ctrl[idx] = torque

    def close(self):
        print("[SIM] Closing MuJoCo fake DxlIO.")
        self.viewer.close()
