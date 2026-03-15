"""
trajectory.py

Circular and square trajectory generators for the 2-DOF robot end-effector.
Desired Cartesian trajectories are converted to joint space via inverse kinematics
and the Jacobian.
"""

import numpy as np
from robot_dynamics import inverse_kinematics, jacobian, jacobian_dot, L1, L2


def circular_trajectory(t, cx=0.5, cy=0.3, radius=0.15, omega=np.pi / 3):

    # Desired Cartesian state 
    xd   = cx + radius * np.cos(omega * t)
    yd   = cy + radius * np.sin(omega * t)

    dxd  = -radius * omega * np.sin(omega * t)
    dyd  =  radius * omega * np.cos(omega * t)

    ddxd = -radius * omega**2 * np.cos(omega * t)
    ddyd = -radius * omega**2 * np.sin(omega * t)

    # Inverse kinematics → joint position 
    qd  = inverse_kinematics(xd, yd, elbow_up=True)

    # Jacobian → joint velocity
    J   = jacobian(qd)
    J_inv = np.linalg.inv(J)
    dqd = J_inv @ np.array([dxd, dyd])

    # Joint acceleration via dq̈ = J⁻¹(ẍ − J̇·q̇)
    Jd   = jacobian_dot(qd, dqd)
    ddqd = J_inv @ (np.array([ddxd, ddyd]) - Jd @ dqd)

    return qd, dqd, ddqd


def square_trajectory(t, cx=0.5, cy=0.3, side=0.20, period=4.0):
   
    omega = 2 * np.pi / period
    A = side / 2

    #Fourier approximation of square wave (first 3 harmonics) 
    def sq(phi):
        return (4 / np.pi) * (
            np.sin(phi)
            + np.sin(3 * phi) / 3
            + np.sin(5 * phi) / 5
        )

    def dsq(phi):
        return (4 / np.pi) * (
            np.cos(phi)
            + np.cos(3 * phi)
            + np.cos(5 * phi)
        )

    def ddsq(phi):
        return -(4 / np.pi) * (
            np.sin(phi)
            + 3 * np.sin(3 * phi)
            + 5 * np.sin(5 * phi)
        )

    phi_x = omega * t
    phi_y = omega * t - np.pi / 2   # 90° phase shift → square path

    xd   = cx + A * sq(phi_x)
    yd   = cy + A * sq(phi_y)
    dxd  = A * omega * dsq(phi_x)
    dyd  = A * omega * dsq(phi_y)
    ddxd = A * omega**2 * ddsq(phi_x)
    ddyd = A * omega**2 * ddsq(phi_y)

    qd   = inverse_kinematics(xd, yd, elbow_up=True)
    J    = jacobian(qd)
    J_inv = np.linalg.inv(J)
    dqd  = J_inv @ np.array([dxd, dyd])
    Jd   = jacobian_dot(qd, dqd)
    ddqd = J_inv @ (np.array([ddxd, ddyd]) - Jd @ dqd)

    return qd, dqd, ddqd
