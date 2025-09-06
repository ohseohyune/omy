# model_pin.py

import os, numpy as np, pinocchio as pin
from mjcf2urdf import convert
from . import config as C

class GravityModel:
    """
    
    MJCF→URDF + Pinocchio 모델 로드 + g(q) 계산
    
    """

    def __init__(self):
        self.urdf_model_loaded = False 

        # MJCF → URDF (입력 XML이 있고, URDF가 없으면 변환)

        xml_exists  = C.XML_PATH and os.path.exists(C.XML_PATH)
        urdf_exists = C.URDF_PATH and os.path.exists(C.URDF_PATH)

        if xml_exists and not urdf_exists:
            print(f"[PIN] Converting MJCF → URDF: {C.XML_PATH} → {C.URDF_PATH}")
            convert(C.XML_PATH, C.URDF_PATH)  

        if urdf_exists:
            print(f"[PIN] Loading URDF: {C.URDF_PATH}")
            self.model = pin.buildModelFromUrdf(C.URDF_PATH, pin.JointModelFixed())
            self.data  = self.model.createData()
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81])

            assert self.model.nv == 6, f"Expected 6 DoF, but got nv={self.model.nv}"
            self.urdf_model_loaded = True
        else:
            print(f"[PIN] URDF not found ({C.URDF_PATH}). Gravity compensation disabled.")

    def tau_g(self, q_current_rad):
        
        if not self.urdf_model_loaded:
            print("[WARN] Gravity compensation disabled. Motors will not be controlled.")
            return None
    
        q_cur_numpyArray = np.asarray(q_current_rad, dtype=float).reshape(self.model.nq) # 입력받은 q값을 pinocchio의 내부 모델 차원에 맞게 reshape
        tau = pin.computeGeneralizedGravity(self.model, self.data, q_cur_numpyArray)  # shape (nv,)
        return tau.tolist() # Numpy배열인 tau를 파이썬 리스트로 변환해줌. 
