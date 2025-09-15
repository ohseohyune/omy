# estimate.py

import numpy as np
from typing import List, Dict, Optional
from . import config as C

class KtEstimator:
    
    """
    온라인 Kt(모터에 1[A]의 전류를 흘렸을 때, 몇 [Nm]의 토크가 발생하냐 ) 추정
      
    제어 루프가 돌아가는 동안 중력 토크(tau_g)와 모터 전류를 비교해서 K(t) 추정 

    좀 더 정확하게 추정하기 위해서, 윈도우(sliding buffer)에 여러 samples 모아서 평균 / 표준편차로 구함 
      """
    
    def __init__(self, window_size: int = C.KT_ESTIMATION_WINDOW_SIZE):
        self.window_size = window_size # 추정에 사용할 샘플 개수 
        self.kt_buffers = {i: [] for i in range(len(C.MOTOR_IDS))} # 각 관절별로 추정된 Kt 값을 저장할 버퍼 

    def update_online(self, tau_gravity: List[float], currents_measured_A: List[float], currents_impedance_raw: List[float]) -> List[Optional[float]]:
        
        """
        [ input ]

          tau_gravity : Pinocchio에서 계산한 중력 토크 [Nm]
          currents_measured_A : 실제 모터에서 읽어온 전류 [A]
          currents_impedance_01A : 임피던스 제어 전류[0.01A 단위]
          ====
          def update_kt_estimator(estimator, dxl, tau_gravity, current_imp_01A):
    
          Update Kt estimator with new sensor data.
    
          measured_currents_A = dxl.read_currents_A()
          estimator.update_online(tau_gravity, measured_currents_A, current_imp_01A)
        
        """
        
        kt_estimates = []

        for i in range(len(C.MOTOR_IDS)):
            imp_current_A = currents_impedance_raw[i] * 0.01  # [A]로 변환
            gravity_current_A = currents_measured_A[i] - imp_current_A # 실제 측정된 전류에서 임피던스 제어 전류를 빼서 중력보상 전류만 추출 [Q] 나중에는 경로에 의한 토크도 빼줘야 할듯 
            
           

            # 임계치: 100 mA, 0.01 Nm 이상일 때만 업데이트
            if abs(gravity_current_A) > 0.1 and abs(tau_gravity[i]) > 0.01: 
                """무조코 시뮬레이터에서 임계값 줄이기"""
                kt = abs(tau_gravity[i]) / (abs(gravity_current_A) * C.MOTOR_GEAR_RATIO * C.MECH_EFFICIENCY)
            else:
                kt = None

            # 추정값을 버퍼에 추가
            # 버퍼 크기가 window_size를 초과하면 가장 오래된 샘플 제거 

            if kt is not None:
                self.kt_buffers[i].append(kt)
                if len(self.kt_buffers[i]) > self.window_size:
                    self.kt_buffers[i].pop(0) 
            kt_estimates.append(kt)
            
        return kt_estimates # 각 관절마다 상이한 Kt 리스트 반환 
  
    def summary(self) -> Dict[int, dict]: 
        """
        각 관절별로 버퍼에 쌓인 데이터를 평균, 표준편차, 샘플 수로 요약해서 반환
        
        실시간으로 추정된 K(t) 샘플들을 여러개 쌓아서, 평균적으로 K(t)가 얼마쯤 되는지를 안정적으로 판단하기 위함.
        """
        state = {}
        for i in range(len(C.MOTOR_IDS)):
            arr = np.array(self.kt_buffers[i], float)
            if arr.size:
                state[i] = dict(avg=float(arr.mean()), std=float(arr.std()), n=int(arr.size))
            else:
                state[i] = dict(avg=None, std=None, n=0)
        return state

