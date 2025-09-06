# trajectory_generator_cartesian.py

"""
Module: trajectory_generator_cartesian.py

Generate purely Cartesian (SE3) trajectories and provide an on‐demand IK solver.
"""

import logging # python 표준 로깅 프레임워크를 쓰기 위함 
from typing import List, Tuple # 타입 힌트용. 읽는 사람이나 정적 분석 도구 등에게 '이 함수는 리스트를 반환한다'는 의도를 명확히 전달해줌. 
import numpy as np # 수치 계산 표준 라이브러리인 Numpy를 np라는 별칭으로 가져와 다양한 기능을 사용
import pinocchio as pin # 로봇 동역학, 기구학 라이브러리인 Pinocchio를 pin이라는 별칭으로 가져옴
from pinocchio import SE3, Quaternion, exp3
"""pin.SE3, pin.Quaternion 대신 직접 SE3, Quaternion 클래스를 네임스페이스로 불러와서,
코드 내에서 짧게 사용하도록 편의성을 높임. SE3은 3X3 회전 + 3X1 병진을 함께 다루는 동차 변환 클래스
Quaternion은 회전만 쿼터니언 형태로 다루는 클래스임"""

logger = logging.getLogger(__name__) #이 모듈의 이름을 로거 이름으로 갖는 인스터늣 생성
"""로거(logger)”는 파이썬의 logging 모듈에서 제공하는 “로그 메시지를 찍는 주체(객체)”입니다. 쉽게 말해, 프로그램이 실행되는 동안 
발생하는 정보(디버그 메시지, 경고, 에러 등)를 기록하고 관리하기 위한 도구"""

def interpolate_transform(T0: SE3, T1: SE3, s: float) -> SE3:
    """
    SE3 포즈 T0와 T1 사이를 비율 s만큼 매끄럽게 보간해서 중간 포즈를 반환 
    s: float -> 0이면 T0, 1이면 T1, 그 사이 값이면 중간 위치 및 자세
    반환 : SE3 -> 보간된 포즈 객체  
    """
    # Linear interpolation of translation
    t_interp = (1 - s) * T0.translation + s * T1.translation
    # T0.translation과 T1.translation은 둘 다 3차원 벡터 [x, y, z]
    # (1 - s) * T0.translation + s * T1.translation 형태의 선형 보간으로 위치를 중간 값으로 만듦
    # s=0.5면 두 위치의 중간점이 되고, s=0.2면 20% 지점이 됨. 

    # Quaternion SLERP for rotation
    q0 = Quaternion(T0.rotation) # 초기 EE의 회전을 쿼터니언으로 변환
    q1 = Quaternion(T1.rotation) # desired EE의 회전 정보를 쿼터니언으로 변환 
    q_interp = q0.slerp(s, q1).normalized() 
    # 구면 선형 보간 : 두 쿼터니언 사이를 구면 경로를 따라 등속도로 보간.
    # 회전 행렬을 선형 보간하면 중간에 스케일이 틀어지거나 부드럽지 않은 반면, SLERP는 항상 단위 쿼터니언을 유지하며 부드러운 회전 궤적을 보장함. 
 
    return SE3(q_interp.toRotationMatrix(), t_interp)
    """
    q_interp.toRotationMatrix() 로 다시 3×3 회전행렬로 변환하고,

    앞서 계산한 t_interp 3D 위치를 함께 넣어 SE3(R, t) 구성자로 중간 포즈를 반환"""

def generate_quintic_trajectory(
    T0: SE3,
    T1: SE3,
    duration: float,
    n_steps: int
) -> Tuple[List[SE3], np.ndarray]:
    
    traj_t = np.linspace(0.0, duration, n_steps)
    poses: List[SE3] = []

    """poses랑 traj_t 생성을 qubic로 만들어보자 """
   
    # 1) translation: cubic spline -> 3D np.ndarray
    p0, p1 = T0.translation, T1.translation
    v0 = v1 = np.zeros(3) # rest-to-rest
    a0 = a1 = np.zeros(3)

    # 2) rotatio : SLERP(쿼터니언 보간)
    q0 = Quaternion(T0.rotation)
    q1 = Quaternion(T1.rotation)

    # 3) 합쳐서 SE3 생성 
    for t in traj_t:
        # a) 병진: 5차 다항 보간 → [pos, vel, acc]
        pos_vel_acc = quintic_spline_multi(
            t, 0.0, duration,
            p0, v0, a0,
            p1, v1, a1
        )
        p = pos_vel_acc[0]  # position vector

        # b) 회전: SLERP으로 normalized quaternion
        s = t / duration
        q_interp = q0.slerp(s, q1).normalized()

        # c) 최종 SE3 저장
        poses.append(SE3(q_interp.toRotationMatrix(), p))

    return poses, traj_t

def quintic_spline(t : float, t_0 : float, t_f : float,
                   x_0 : float,
                   x_dot_0 : float,
                   x_ddot_0 : float,
                   x_f : float,
                   x_dot_f : float,
                   x_ddot_f : float):

    if t < t_0:
        return np.array([x_0, x_dot_0, x_ddot_0])
    
    if t > t_f:
        return np.array([x_f, x_dot_f, x_ddot_f])
    
    t_s = t_f - t_0
    a1 = x_0
    a2 = x_dot_0
    a3 = x_ddot_0 / 2.0
    mat = np.array([
        [t_s**3, t_s**4, t_s**5],
        [3 * t_s**2, 4 * t_s**3, 5 * t_s**4],
        [6 * t_s, 12 * t_s**2, 20 * t_s**3]
    ])
    v = np.array([
        x_f - x_0 - x_dot_0 * t_s - 0.5 * x_ddot_0 * t_s**2,
        x_dot_f - x_dot_0 - x_ddot_0 * t_s,
        x_ddot_f - x_ddot_0
    ])
    res = np.linalg.solve(mat, v)
    a4, a5, a6 = res
    t_e = t - t_0
    position = a1 + a2 * t_e + a3 * t_e**2 + a4 * t_e**3 + a5 * t_e**4 + a6 * t_e**5
    velocity = a2 + 2 * a3 * t_e + 3 * a4 * t_e**2 + 4 * a5 * t_e**3 + 5 * a6 * t_e**4
    acceleration = 2 * a3 + 6 * a4 * t_e + 12 * a5 * t_e**2 + 20 * a6 * t_e**3

    return np.array([position, velocity, acceleration])

def quintic_spline_multi(t, t_0, t_f,
                        x_0 : np.ndarray, 
                        x_dot_0 : np.ndarray, 
                        x_ddot_0 : np.ndarray, 
                        x_f : np.ndarray, 
                        x_dot_f : np.ndarray, 
                        x_ddot_f : np.ndarray):
    if t < t_0:
        return np.array([x_0, x_dot_0, x_ddot_0])
    
    if t > t_f:
        return np.array([x_f, x_dot_f, x_ddot_f])
    
    dim = x_0.shape[0]

    results = np.zeros((3, dim))

    for d in range(dim):
        res = quintic_spline(t, t_0, t_f, x_0[d], x_dot_0[d], x_ddot_0[d], x_f[d], x_dot_f[d], x_ddot_f[d])
        results[:, d] = res

    return results


def generate_wiping_trajectory(
    pin_model: pin.Model,
    pin_data: pin.Data,
    frame_id: int,
    q_start: np.ndarray,
    T_start: SE3,
    x_amplitude: float,
    y_amplitude: float,
    duration: float,
    n_steps: int,
    loops: int
) -> Tuple[List[SE3], np.ndarray]:

    all_poses: List[SE3] = []
    all_times: List[float] = []
    t_accum = 0.0
    q_prev = q_start.copy()
    T_prev = T_start.copy()

    """
    T_descend = T_prev.copy()
    T_descend.translation[2] -=0.01

    poses, times = generate_quintic_trajectory(
        pin_model, pin_data, frame_id, q_prev,T_prev, T_descend, duration=0.3, n_steps=500
    )
    # 시간 누적
    times = times + t_accum
    t_accum = times[-1]
    # 결과에 추가
    all_poses.extend(poses)
    all_times.extend(times.tolist())
    # 다음 세그먼트 시작점을 하강된 위치로 갱신
    T_prev = T_descend
    # q_prev 는 IK seed로 그대로 유지
    logger.info("Added initial descent: 3cm down in %d steps", len(poses))
"""
    
    # Define rectangle offsets
    offsets = [
        np.array([-x_amplitude, 0.0, 0.0]),
        np.array([0.0, -y_amplitude, 0.0]),
        np.array([x_amplitude, 0.0, 0.0]),
        np.array([0.0, y_amplitude, 0.0]),
    ]

    for loop in range(loops):
        for offset in offsets:
            T_next = T_prev.copy()
            T_next.translation += offset
            # maintain Z height

            poses, times = generate_quintic_trajectory(
                T_prev, T_next, duration, n_steps
            )

            # shift times by accumulated offset
            times = times + t_accum
            t_accum = times[-1]

            all_poses.extend(poses)
            all_times.extend(times.tolist())

            # prepare for next segment
            T_prev = T_next
            q_prev = q_prev  # IK seed stays the same; actual IK done later

    logger.info(
        "Generated wiping trajectory: %d loops, total steps=%d",
        loops, len(all_poses)
    )
    return all_poses, np.array(all_times)

# Damped Least-Squares 방식으로 목표 EE 포즈 T_target에 수렴하는 관절값을 찾는 ik
def compute_ik(
    pin_model: pin.Model,
    pin_data: pin.Data,
    frame_id: int,
    T_target: SE3, # 도달하고자 하는 EE의 SE3(회전 + 병진)
    q_init: np.ndarray, # ik 반복의 초기 관절값 시드
    max_iter: int = 500, # 최대 반복 횟수 
    tol: float = 3e-2 # 수렴 허용 오차
) -> Tuple[bool, np.ndarray]: # 수렴 여부와 최종 관절값
    
    damping_base: float = 1e-2
    
    q = q_init.copy()
    for i in range(max_iter): # 최대 max_iter 회수만큼 IK 업데이트를 시도
        
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        
        T_curr = pin_data.oMf[frame_id]  # .homogeneous 없이 SE3 객체를 바로 사용
        err6 = pin.log6(T_curr.inverse() * T_target).vector
        err_norm = np.linalg.norm(err6)
        
        if np.linalg.norm(err6) < tol:
            logger.debug("IK converged in %d iterations", i)
            return True, q

        J = pin.computeFrameJacobian(
            pin_model, pin_data, q, frame_id, pin.ReferenceFrame.LOCAL
        )
        
        lam = damping_base * err_norm
        dq = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6)) @ err6
        q += dq # 작은 보정량 dq를 더해 q에 점근적으로 목표 관절값으로 수렴시키기 

    logger.warning("IK did not converge after %d iterations", max_iter)
    return False, q # max_iter 횟수만큼 돌았는데 tol 내로 수렴되지 않으면, False 플래그와 마지막까지 계산된 q 를 반환

# 이 알고리즘은 Singular configuration(자코비안 행렬이 저차원)에서도 Large joint updates 방지를 위해 λ 댐핑항을 넣어 안정적인 IK 수렴을 도모
