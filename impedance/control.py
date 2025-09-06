# control.py
from typing import List, Tuple
from . import config as C # . : 현재 디렉토리 

class ImpedanceController:
    """ 임피던스 제어 + 중력보상 전류 생성 """
    def __init__(self):
        self.IMPEDANCE_KP = C.IMPEDANCE_KP
        self.IMPEDANCE_KD = C.IMPEDANCE_KD
        self.Q_DES: List[float]  = []
        self.DQ_DES: List[float] = [0.0]*len(C.MOTOR_IDS)
        self.dq_filt: List[float]= []

    def init_targets(self, q_init: List[float], dq_init: List[float]):
        """로봇을 몄을 때의 자세와 속도를 초기 목표로 삼아서 
        로봇이 켜진 순간 그 위치에서 멈춰있도록 제어""" 
        self.Q_DES  = list(q_init)
        self.dq_filt= list(dq_init)

    @staticmethod
    def tau_to_current01A_list(tau_out_list: List[float]) -> List[float]:
        
        """입력 : 각 관절의 중력 보상 토크(tau_out_list) 
        
        1. 토크 -> 모터 전류(A) 변환(Kt * 기어비 * eff로 나눔)
        2. Dynamixel Goal Current 단위로 스케일링
        
        최종 출력 : 0.01A 단위 전류 리스트"""

        denom = C.KT_NM_PER_A * C.MOTOR_GEAR_RATIO * C.EFFICIENCY
        return [ (tau/denom)/0.01 for tau in tau_out_list ]

    def step(self, q: List[float], dq: List[float], tau_g_out: List[float]) -> Tuple[List[float], List[float]]:
        # 속도 저역통과
        for i in range(len(C.MOTOR_IDS)):
            self.dq_filt[i] = (1.0 - C.VEL_FILTER_ALPHA) * self.dq_filt[i] + C.VEL_FILTER_ALPHA * dq[i]

        # 만약 tau_g가 주어지지 않았다면 → PD fallback 사용
        if tau_g is None:
            Kp = 2.0
            Kd = 0.1
            q_home = self.q_DES  # 초기 목표 자세를 홈 포즈로 사용
            tau_g = [-Kp * (qi - qh) - Kd * dqi for qi, qh, dqi in zip(q, q_home, dq)]

        # 중력보상 전류(0.01A) => 피노키오에서 얻은 중력 토크를 전류 명령으로 변환
        u_g = self.tau_to_current01A_list(tau_g_out)

        # 임피던스 + 중력
        u, u_imp = [], []
        for i in range(len(C.MOTOR_IDS)):
            e  = self.Q_DES[i]  - q[i]
            ed = self.DQ_DES[i] - self.dq_filt[i]
            u_imp_i = self.IMPEDANCE_KP[i]*e + self.IMPEDANCE_KD[i]*ed
            u_i     = u_imp_i + u_g[i]
            u.append(u_i); u_imp.append(u_imp_i)
        return u, u_imp

