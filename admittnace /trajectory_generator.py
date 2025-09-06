#trajectory_generator.py

import numpy as np

tol=1e-8

def decide_vp(q0, q1, v0, v1, A): 
    dq   = q1 - q0 
    term = 0.5 * (v0 + v1) #
    disc = term**2 + A * np.abs(dq)
    sqrt_disc = np.sqrt(disc)
    denom = term + np.copysign(sqrt_disc, dq)

    # 움직이는 관절에만 공식을 적용, 나머지는 v0로 채움
    moving = np.abs(dq) > tol
    vp = np.zeros_like(dq)
    vp[moving] = A[moving] * dq[moving] / denom[moving]
    # (원래 속도가 0이 아니면 v0[moving] 대신 v0 사용)-
    return vp


def set_vc(vp, V):
    over = np.abs(vp) > V 
    vc = vp.copy()
    vc[over] = np.copysign(V[over], vp[over])
    return vc

def calc_times(vc, v0, v1, A): 
    Ta = 3 * np.abs(vc - v0) / (2 * A)
    Td = 3 * np.abs(v1 - vc) / (2 * A)
    return Ta, Td

def calc_Tc(q0, q1, v0, vc, v1, Ta, Td): # 등속 구간의 지속 시간 구하기

    # 1) 가속/감속 거리
    dq_acc = 0.5 * (v0 + vc) * Ta
    dq_dec = 0.5 * (vc + v1) * Td
    dq_tot = q1 - q0

    # 2) 분모 vc 가 0 에 가까운 곳(=static joint) 마스크
    moving = np.abs(vc) > tol

    # 3) Tc 계산 (moving 한 joint 에만 나눗셈 수행)
    Tc = np.zeros_like(vc)
    Tc[moving] = (dq_tot[moving] - dq_acc[moving] - dq_dec[moving]) / vc[moving]

    # 4) 혹시 음수가 나오면 0 으로 클램핑 (삼각형 프로파일 처리)
    Tc = np.clip(Tc, 0.0, None)

    return Tc

def calc_coeffs_acc(q0, v0, vc, Ta): 
    a0 = q0.copy()
    a1 = v0.copy()
    a2 = np.zeros_like(q0)
    a3 = np.zeros_like(q0)
    a4 = np.zeros_like(q0)
    a5 = np.zeros_like(q0)
    
    # 자칫 Ta=0일 때 1/Ta**2 등이 nan이 되니까 safe 마스크 사용
    safe = np.abs(Ta) > tol
    a3[safe] = (vc[safe] - v0[safe]) / Ta[safe]**2
    a4[safe] = (v0[safe] - vc[safe]) / (2 * Ta[safe]**3)

    # (6,6) 배열 반환
    return np.stack([a0, a1, a2, a3, a4, a5], axis=0)

def calc_coeffs_dec(q1, v1, vc, Td): 
    d0 = q1 - Td * (vc + v1) / 2
    d1 = vc.copy()
    d2 = np.zeros_like(q1)
    d3 = np.zeros_like(q1)
    d4 = np.zeros_like(q1)
    d5 = np.zeros_like(q1)

    safe = np.abs(Td) > tol   # Td가 0에 가까운 관절은 제외
    d3[safe] = (v1[safe] - vc[safe]) / Td[safe]**2
    d4[safe] = (vc[safe] - v1[safe]) / (2 * Td[safe]**3)
    
    return np.stack([d0, d1, d2, d3, d4, d5], axis=0)  # shape=(6,6)


# 3) 전체 궤적 생성 함수

def generate_trajectory(q0, q1, v0, v1, V, A, t0, n_steps=100):
    # 3.1 정적(static) 관절 마스크
    dq     = q1 - q0
    static = np.isclose(dq, 0.0)
    #0인 관절을 True로 표현해서,0으로 나누는 오류를 방지하고, 관절값을 고정하기 위함. 
    # 3.2 vp → vc
    vp = decide_vp(q0, q1, v0, v1, A)
    vc = set_vc(vp, V)

    # 3.3 Ta, Td, Tc
    Ta, Td = calc_times(vc, v0, v1, A)
    Tc      = calc_Tc(q0, q1, v0, vc, v1, Ta, Td) #constant time -> 사다리꼴 프로파일인 관절만 값이 존재. 

    # 3.4 구간 경계시각
    t0_arr = np.full_like(q0, t0)
    ta     = t0_arr + Ta
    
    #ta = ta.astype(np.float64)
    #Tc = Tc.astype(np.float64)

    td     = ta + Tc
    t1     = td + Td

    # 3.5 전체 종료 시간 (static 관절은 t0로 대체)
    t_end = np.nanmax(np.where(static, t0, t1))
    #전체 시간 벡터를 모든 관절 중 가장 늦게 끝나는 시간으로 설정했음
    t     = np.linspace(t0, t_end, n_steps)

    # 3.6 5차 다항식 계수
    A_coefs = calc_coeffs_acc(q0, v0, vc, Ta)
    D_coefs = calc_coeffs_dec(q1, v1, vc, Td)

    # 3.7 도함수 계수 (k·a_k)
    k        = np.arange(6)
    A_dcoefs = A_coefs * k[:,None]
    D_dcoefs = D_coefs * k[:,None]

    # 3.8 시간 메쉬 & 마스크
    T    = t[:,None]       # (n_steps,1)
    taB  = ta[None,:]      # (1,6)
    tdB  = td[None,:]

    m_acc   = T <  taB
    m_const = (T>= taB) & (T< tdB)
    m_dec   = T>= tdB

    N = q0.shape[0]  # 관절 수
    dt_acc   = T - t0
    dt_acc = np.tile(dt_acc, (1, N))  # → shape: (n_steps, N)

    dt_const = T - taB
    dt_dec   = T - tdB
    #dt_dec = np.tile(dt_dec, (1, N))

    # 3.9 다항식 평가 준비
    powers      = np.arange(6)[:,None,None]    # (6,1,1)
    dt_acc_pows = dt_acc[None,:,:]   ** powers  # (6,n_steps,6)
    dt_dec_pows = dt_dec[None,:,:]   ** powers

   # 위치 계산
    q_acc = np.einsum('kn,tnk->tn', A_coefs, dt_acc_pows.transpose(1, 2, 0))
    q_dec = np.einsum('kn,tnk->tn', D_coefs, dt_dec_pows.transpose(1, 2, 0))

    # 속도 계산
    qd_acc = np.einsum('kn,tnk->tn', A_dcoefs, dt_acc_pows.transpose(1, 2, 0))
    qd_dec = np.einsum('kn,tnk->tn', D_dcoefs, dt_dec_pows.transpose(1, 2, 0))

    # 3.12 등속 구간
   
    powers = np.arange(6)[:, None]     # (6,1)
    dtTa_pows = Ta[None, :] ** powers  # (6,10)

    # 다항식 평가: qa_Ta = Σ (a_k * Ta^k)
    qa_Ta = np.sum(A_coefs * dtTa_pows, axis=0)  # (10,)
    
    q_const   = qa_Ta[None,:] + vc[None,:] * dt_const
    qd_const  = np.tile(vc, (n_steps,1))

    # 3.13 최종 합성
    q  = np.where(m_acc,    q_acc,
         np.where(m_const,  q_const, q_dec))
    qd = np.where(m_acc,    qd_acc,
         np.where(m_const,  qd_const, qd_dec))
    
    # dq가 작은 관절 (예: 7,8,9,10) 은 T₁ₖ 이 훨씬 작아서,
    # 꽤 긴 시간 동안 “감속 폴리노미얼”이 유효 구간을 벗어나 적용되어
    # 심하게 발산해 버립니다.generate_trajectory 의 “감속(polynomial) 구간 이후”에 궤적이 계속 발산하는 문제는,
    # q_dec 다항식이 원래 유효 구간(각 관절의 종료시점) 이후에도 그대로 적용되기 때문입니다.
    # 이를 해결하려면 “t > t1[k] (감속 종료 시점)” 이후에는 강제로 목표위치 q1[k] 를 고정해 주면 됩니다.
    # 이제 오차나 발산이 없다 ! 오예 !!!
    times = t  # shape (n_steps,)
    for k in range(q0.shape[0]):
        # 감속 종료 시점 이후 마스크
        after = times >= t1[k]
        q[after, k]  = q1[k]    # 위치 고정
        qd[after, k] = 0.0      # 속도 0

    # 3.14 정적 관절 보정
    if np.any(static):
        q[:, static]  = q0[static]
        qd[:, static] = 0.0

    return t, q, qd

