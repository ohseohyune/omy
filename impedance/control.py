# control.py
from typing import List, Tuple, Optional
import numpy as np
from . import config as C # . : 현재 디렉토리 
from model_pin import GravityModel

class ImpedanceController:
    """ 
    임피던스 제어 + 중력보상 전류 생성
    """
    
    def __init__(self):
        self.IMPEDANCE_KP = C.IMPEDANCE_KP
        self.IMPEDANCE_KD = C.IMPEDANCE_KD
        self.Q_DES: List[float]  = [0.0]*len(C.MOTOR_IDS) 
        self.DQ_DES: List[float] = [0.0]*len(C.MOTOR_IDS) 
        self.DDQ_DES: List[float] = [0.0]*len(C.MOTOR_IDS)
        self.dq_lpf_filtered: List[float]= [0.0]*len(C.MOTOR_IDS)

    def init_targets(self, q_init: List[float], dq_init: List[float]):
        """로봇을 몄을 때의 자세와 속도를 초기 목표로 삼아서 
        로봇이 켜진 순간 그 위치에서 멈춰있도록 제어""" 
        self.Q_DES  = list(q_init)
        self.dq_lpf_filtered= list(dq_init) 

    @staticmethod
    def tau_to_current_raw(tau_out_list: List[float],kt_estimates:List[float]) -> List[float]:
        
        """입력 : 각 관절의 중력 보상 토크(tau_out_list) 
        
        1. 토크 -> 모터 전류(A) 변환(Kt * 기어비 * eff로 나눔)
        2. Dynamixel Goal Current 단위로 스케일링
        
        최종 출력 : 0.01A 단위 전류 리스트"""
        current_raw_list = []
        for i,tau in enumerate(tau_out_list):
            kt = kt_estimates[i]

            if kt is None:
                kt = C.TORQUE_CONSTANT_KT

            denom = kt * C.MOTOR_GEAR_RATIO * C.MECH_EFFICIENCY
            current_raw = (tau / denom) / 0.01
            current_raw_list.append(current_raw)
        return current_raw_list

    def compute_impedance_with_gravity(self, q: List[float], dq: List[float], tau_g: List[float], kt_estimates: List[Optional[float]]) -> Tuple[List[float], List[float]]:
        
        dyn = GravityModel()
        # 속도 저역통과
        for i in range(len(C.MOTOR_IDS)):
            self.dq_lpf_filtered[i] = (1.0 - C.VEL_FILTER_ALPHA) * self.dq_lpf_filtered[i] + C.VEL_FILTER_ALPHA * dq[i]
        
        q_error     = np.array(self.Q_DES)  - np.array(q)
        q_dot_error = np.array(self.DQ_DES) - np.array(self.dq_lpf_filtered)

        q_dot_dot_ref = (
            np.array(self.DDQ_DES)
            + np.array(self.IMPEDANCE_KD) * q_dot_error
            + np.array(self.IMPEDANCE_KP) * q_error
        )        

        # 만약 gravity_comp_torque가 주어지지 않았다면 → PD 사용
        if tau_g is None:
            Kp = 4.0
            Kd = 0.1
            q_home = self.Q_DES  # 초기 목표 자세를 홈 포즈로 사용
            tau_g = [-Kp * (qi - qh) - Kd * dqi for qi, qh, dqi in zip(q, q_home, dq)]

        # 중력보상 전류(0.01A) => 피노키오에서 얻은 중력 토크를 전류 명령으로 변환

        # 임피던스 + 중력
        current_raw, current_imp_raw = [], []

        MassMatrix = dyn.mass_matrix(q)
        Coriolis = dyn.Coriolis(q, dq)

        # 임피던스 제어 항 (Kp*e + Kd*ed)
        computed_torque = (
            MassMatrix @ q_dot_dot_ref
            + Coriolis 
        )

        # τ = M(q)qdd_ref + C(q,q̇)q̇ + g(q)
        tau_cmd = computed_torque + np.array(tau_g)
        # 토크 → 전류
        current_raw = self.tau_to_current_raw(tau_cmd.tolist(), kt_estimates)
        current_imp_raw = self.tau_to_current_raw(computed_torque.tolist(),kt_estimates)

        
        return current_raw, current_imp_raw #[raw_current] 모터에 바로 들어갈 수 있는 전류 명령 값

