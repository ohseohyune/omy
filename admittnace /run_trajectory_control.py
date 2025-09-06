#run_trajectory_control.py

import mujoco
import mujoco.viewer as mj_view
import numpy as np
import time
import matplotlib.pyplot as plt
from trajectory_generator import generate_trajectory 

MODEL_XML_PATH = "/home/ohseohyun/colcon_ws/src/robotis_mujoco_menagerie/robotis_omy/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

dt = model.opt.timestep
n_act = model.nu

q0 = np.zeros(10)
q1 = np.array([0.5, 0.5, 0.4, 0.6, 0.5, 0.3, 0.2, 0.1, 0.2, 0.1])

v0 = np.zeros(10)
v1 = np.zeros(10)
V = np.full(10, 150.0)  
A = np.full(10, 150.0) 

t0 = 0.001
n_steps = 10000


traj_t, traj_q, traj_qd = generate_trajectory(q0, q1, v0, v1, V, A, t0, n_steps)
#print(np.max(np.abs(traj_qd)))
#print("총 실행 시간:", traj_t[-1], "초")

for _ in range(100):
    mujoco.mj_step(model, data)

try:
    with mj_view.launch_passive(model, data) as viewer:
        idx = 0  

        while viewer.is_running():
            #if idx >= len(traj_t):
            #    print("Trajectory completed.")
            #    break
            
            # 중력 항 계산 
            mujoco.mj_rnePostConstraint(model, data)

            # 중력 보상 항 
            gravity_comp = data.qfrc_bias[:n_act] / model.actuator_gear[:, 0]

            # 제어 명령: 목표 위치를 data.ctrl 넣기 
            #data.ctrl[:n_act] = traj_q[idx] #xml의 액추에이터가 position으로 

            # 제어 명령: 목표 속도를 data.ctrl 넣기 
            data.ctrl[:n_act] = traj_qd[idx] + gravity_comp # xml의 액추에이터가 velocity

            mujoco.mj_step(model, data)
            viewer.sync()
            idx += 1 
            time.sleep(dt)  # 실시간 동기화

except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")
