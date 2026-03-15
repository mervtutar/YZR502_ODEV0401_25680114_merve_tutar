"""
2-DOF Planar RR Robot - Dynamic Model and Inverse Dynamics Controller
Dynamics derived using the Lagrange method.
Robot Parameters:
    L1 = 0.5 m,  L2 = 0.5 m
    m1 = 1.0 kg, m2 = 0.5 kg
    lc1 = L1/2,  lc2 = L2/2  (CoM at midpoint)
    I_i = (1/12) * m_i * L_i^2  (thin rod approximation)
"""

import numpy as np

#Robot Parameters
L1, L2 = 0.5, 0.5          # link lengths [m]
m1, m2 = 1.0, 0.5          # link masses [kg]
lc1, lc2 = L1 / 2, L2 / 2 # CoM distances from proximal joint [m]
I1 = (1 / 12) * m1 * L1**2 # moment of inertia link 1 [kg·m²]
I2 = (1 / 12) * m2 * L2**2 # moment of inertia link 2 [kg·m²]
g  = 9.81                   # gravitational acceleration [m/s²]


#Mass Matrix M(q) 
def mass_matrix(q):
    
    q2 = q[1]
    c2 = np.cos(q2)

    M11 = (m1 * lc1**2 + I1
           + m2 * (L1**2 + lc2**2 + 2 * L1 * lc2 * c2)
           + I2)
    M12 = m2 * (lc2**2 + L1 * lc2 * c2) + I2
    M22 = m2 * lc2**2 + I2

    return np.array([[M11, M12],
                     [M12, M22]])


#Coriolis / Centrifugal Matrix C(q, dq)
def coriolis_matrix(q, dq):
    
    q2 = q[1]
    dq1, dq2 = dq
    h = m2 * L1 * lc2 * np.sin(q2)

    C = np.array([[-h * dq2,        -h * (dq1 + dq2)],
                  [ h * dq1,         0.0             ]])
    return C


# Gravity Vector G(q)
def gravity_vector(q):
  
    q1, q2 = q
    G1 = ((m1 * lc1 + m2 * L1) * g * np.cos(q1)
          + m2 * lc2 * g * np.cos(q1 + q2))
    G2 = m2 * lc2 * g * np.cos(q1 + q2)
    return np.array([G1, G2])


# Inverse Dynamics Controller 
def inverse_dynamics_control(q, dq, qd, dqd, ddqd, Kp, Kv):
    """
    Computed-torque (inverse dynamics) controller with PD outer loop.

    τ = M(q)·[q̈_d + Kv·(q̇_d − q̇) + Kp·(q_d − q)] + C(q,q̇)·q̇ + G(q)

    Parameters
    ----------
    q, dq         : current joint positions and velocities, shape (2,)
    qd, dqd, ddqd : desired positions, velocities, accelerations, shape (2,)
    Kp, Kv        : gain matrices, shape (2, 2)

    Returns
    -------
    tau : ndarray, shape (2,)  — joint torques [N·m]
    """
    e   = qd  - q          # position error
    de  = dqd - dq         # velocity error
    M   = mass_matrix(q)
    C   = coriolis_matrix(q, dq)
    G   = gravity_vector(q)

    tau = M @ (ddqd + Kv @ de + Kp @ e) + C @ dq + G
    return tau


# Forward Kinematics
def forward_kinematics(q):
    """
    End-effector Cartesian position.

    Parameters
    ----------
    q : array-like, shape (2,)

    Returns
    -------
    x, y : float — end-effector coordinates [m]
    """
    q1, q2 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return x, y


# Inverse Kinematics
def inverse_kinematics(x, y, elbow_up=True):
    """
    Analytical inverse kinematics for the 2-DoF RR robot.

    Parameters
    ----------
    x, y     : desired end-effector position [m]
    elbow_up : bool — selects elbow-up (True) or elbow-down (False) solution

    Returns
    -------
    q : ndarray, shape (2,)  — [q1, q2] in radians
    """
    cos_q2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)         # numerical safety
    sin_q2 = np.sqrt(1.0 - cos_q2**2)
    if not elbow_up:
        sin_q2 = -sin_q2

    q2 = np.arctan2(sin_q2, cos_q2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * sin_q2, L1 + L2 * cos_q2)
    return np.array([q1, q2])


# Jacobian
def jacobian(q):
    """
    Geometric Jacobian of the 2-DoF robot end-effector.

    Parameters
    ----------
    q : array-like, shape (2,)

    Returns
    -------
    J : ndarray, shape (2, 2)
    """
    q1, q2 = q
    J = np.array([
        [-L1 * np.sin(q1) - L2 * np.sin(q1 + q2),  -L2 * np.sin(q1 + q2)],
        [ L1 * np.cos(q1) + L2 * np.cos(q1 + q2),   L2 * np.cos(q1 + q2)]
    ])
    return J


# Jacobian Time Derivative
def jacobian_dot(q, dq):
    
    q1, q2 = q
    dq1, dq2 = dq
    Jd = np.array([
        [-(L1 * np.cos(q1) + L2 * np.cos(q1 + q2)) * dq1
          - L2 * np.cos(q1 + q2) * dq2,
         -L2 * np.cos(q1 + q2) * (dq1 + dq2)],
        [-(L1 * np.sin(q1) + L2 * np.sin(q1 + q2)) * dq1
          - L2 * np.sin(q1 + q2) * dq2,
         -L2 * np.sin(q1 + q2) * (dq1 + dq2)]
    ])
    return Jd


# ODE Right-Hand Side
def robot_ode(t, state, trajectory_fn, Kp, Kv):

    q  = state[:2]
    dq = state[2:]

    qd, dqd, ddqd = trajectory_fn(t)
    tau = inverse_dynamics_control(q, dq, qd, dqd, ddqd, Kp, Kv)

    M   = mass_matrix(q)
    C   = coriolis_matrix(q, dq)
    G   = gravity_vector(q)

    # Forward dynamics: q̈ = M⁻¹(τ − C·q̇ − G)
    ddq = np.linalg.solve(M, tau - C @ dq - G)

    return np.concatenate([dq, ddq])
