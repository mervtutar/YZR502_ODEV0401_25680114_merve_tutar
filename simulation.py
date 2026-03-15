"""
Main simulation script for the 2-DOF planar robot inverse dynamics control.

Usage:
    python simulation.py 
Outputs:
    figures/01_joint_positions.png
    figures/02_joint_errors.png
    figures/03_joint_torques.png
    figures/04_cartesian_trajectory.png
    figures/05_gain_analysis.png
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

from robot_dynamics import robot_ode, forward_kinematics
from trajectory import circular_trajectory, square_trajectory

os.makedirs("figures", exist_ok=True)

STYLE = {
    "desired":  {"color": "#2563EB", "lw": 1.8, "ls": "--", "label": "Desired"},
    "actual":   {"color": "#DC2626", "lw": 1.5, "ls": "-",  "label": "Actual"},
    "error":    {"color": "#16A34A", "lw": 1.5, "ls": "-"},
    "torque1":  {"color": "#7C3AED", "lw": 1.5, "ls": "-",  "label": r"$\tau_1$"},
    "torque2":  {"color": "#DB2777", "lw": 1.5, "ls": "-",  "label": r"$\tau_2$"},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


#Simulation
def run_simulation(traj_name="circular", kp_val=100.0, kv_val=20.0,
                   duration=12.0, dt=0.005):
   
    traj_fn = circular_trajectory if traj_name == "circular" else square_trajectory

    Kp = np.diag([kp_val, kp_val])
    Kv = np.diag([kv_val, kv_val])

    # Initial conditions: start at the first desired pose
    qd0, _, _ = traj_fn(0.0)
    state0 = np.concatenate([qd0, np.zeros(2)])

    t_span = (0.0, duration)
    t_eval = np.arange(0.0, duration, dt)

    sol = solve_ivp(
        fun=lambda t, s: robot_ode(t, s, traj_fn, Kp, Kv),
        t_span=t_span,
        y0=state0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    t  = sol.t
    q  = sol.y[:2, :].T   # (N, 2)
    dq = sol.y[2:, :].T   # (N, 2)

    # Recompute desired trajectories and torques along the solution 
    N = len(t)
    qd   = np.zeros((N, 2))
    dqd  = np.zeros((N, 2))
    ddqd = np.zeros((N, 2))
    tau  = np.zeros((N, 2))

    from robot_dynamics import (inverse_dynamics_control, mass_matrix,
                                 coriolis_matrix, gravity_vector)

    for i, ti in enumerate(t):
        qd_i, dqd_i, ddqd_i = traj_fn(ti)
        qd[i]   = qd_i
        dqd[i]  = dqd_i
        ddqd[i] = ddqd_i
        tau[i]  = inverse_dynamics_control(q[i], dq[i], qd_i, dqd_i, ddqd_i,
                                            Kp, Kv)

    #End-effector Cartesian positions 
    xy  = np.array([forward_kinematics(q[i])  for i in range(N)])
    xyd = np.array([forward_kinematics(qd[i]) for i in range(N)])

    return dict(t=t, q=q, dq=dq, qd=qd, dqd=dqd, ddqd=ddqd,
                tau=tau, xy=xy, xyd=xyd)


#Plotting 
def plot_joint_positions(res, traj_name, save=True):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.suptitle(f"Joint Positions – {traj_name.capitalize()} Trajectory", fontweight="bold")

    for i, ax in enumerate(axes):
        ax.plot(res["t"], np.rad2deg(res["qd"][:, i]),
                **{**STYLE["desired"], "label": r"$q_{" + f"{i+1}" + r",d}$"})
        ax.plot(res["t"], np.rad2deg(res["q"][:, i]),
                **{**STYLE["actual"],  "label": r"$q_{" + f"{i+1}" + r"}$"})
        ax.set_ylabel(f"Joint {i+1} [°]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    if save:
        fig.savefig("figures/01_joint_positions.png", bbox_inches="tight")
    plt.close(fig)


def plot_joint_errors(res, traj_name, save=True):
    e = np.rad2deg(res["qd"] - res["q"])      # tracking error

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.suptitle(f"Tracking Error – {traj_name.capitalize()} Trajectory", fontweight="bold")

    for i, ax in enumerate(axes):
        ax.plot(res["t"], e[:, i], **STYLE["error"],
                label=r"$e_{" + f"{i+1}" + r"} = q_{d} - q$")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_ylabel(f"Error Joint {i+1} [°]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    if save:
        fig.savefig("figures/02_joint_errors.png", bbox_inches="tight")
    plt.close(fig)


def plot_torques(res, traj_name, save=True):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    fig.suptitle(f"Joint Torques – {traj_name.capitalize()} Trajectory", fontweight="bold")

    labels = [r"$\tau_1$ [N·m]", r"$\tau_2$ [N·m]"]
    colors = [STYLE["torque1"]["color"], STYLE["torque2"]["color"]]

    for i, ax in enumerate(axes):
        ax.plot(res["t"], res["tau"][:, i], color=colors[i], lw=1.5,
                label=labels[i])
        ax.set_ylabel(labels[i])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    if save:
        fig.savefig("figures/03_joint_torques.png", bbox_inches="tight")
    plt.close(fig)


def plot_cartesian(res, traj_name, save=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    sd = {k: v for k, v in STYLE["desired"].items() if k != "label"}
    sa = {k: v for k, v in STYLE["actual"].items()  if k != "label"}
    ax.plot(res["xyd"][:, 0], res["xyd"][:, 1], **sd, label="Desired")
    ax.plot(res["xy"][:, 0],  res["xy"][:, 1],  **sa, label="Actual")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"End-Effector Trajectory – {traj_name.capitalize()}", fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig("figures/04_cartesian_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def plot_gain_analysis(traj_name="circular", duration=12.0, save=True):
   
    gain_sets = [
        (50,  10,  "Kp=50, Kv=10"),
        (100, 20,  "Kp=100, Kv=20"),
        (200, 40,  "Kp=200, Kv=40"),
        (500, 60,  "Kp=500, Kv=60"),
    ]

    rms1_list, rms2_list, labels = [], [], []

    for kp, kv, lbl in gain_sets:
        print(f"  Gain analysis: {lbl} …")
        r = run_simulation(traj_name, kp_val=kp, kv_val=kv, duration=duration)
        e = np.rad2deg(r["qd"] - r["q"])
        rms1_list.append(np.sqrt(np.mean(e[:, 0]**2)))
        rms2_list.append(np.sqrt(np.mean(e[:, 1]**2)))
        labels.append(lbl)

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, rms1_list, w, label="Joint 1", color="#2563EB", alpha=0.85)
    ax.bar(x + w/2, rms2_list, w, label="Joint 2", color="#DC2626", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMS Tracking Error [°]")
    ax.set_title(f"Gain Analysis – {traj_name.capitalize()} Trajectory", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig("figures/05_gain_analysis.png", bbox_inches="tight")
    plt.close(fig)

    return labels, rms1_list, rms2_list


def main():
    parser = argparse.ArgumentParser(description="2-DOF Robot Inverse Dynamics Simulation")
    parser.add_argument("--traj",     default="circular", choices=["circular", "square"])
    parser.add_argument("--Kp",       type=float, default=100.0)
    parser.add_argument("--Kv",       type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--save",     action="store_true", default=True)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  2-DOF Robot Inverse Dynamics Simulation")
    print(f"  Trajectory : {args.traj}")
    print(f"  Kp = {args.Kp},  Kv = {args.Kv}")
    print(f"  Duration   : {args.duration} s")
    print(f"{'='*60}\n")

    print("Running simulation …")
    res = run_simulation(args.traj, kp_val=args.Kp, kv_val=args.Kv,
                         duration=args.duration)

    e = np.rad2deg(res["qd"] - res["q"])
    print(f"RMS error  J1: {np.sqrt(np.mean(e[:,0]**2)):.4f}°")
    print(f"RMS error  J2: {np.sqrt(np.mean(e[:,1]**2)):.4f}°")
    print(f"Max |error| J1: {np.max(np.abs(e[:,0])):.4f}°")
    print(f"Max |error| J2: {np.max(np.abs(e[:,1])):.4f}°")
    print(f"Max |τ1|: {np.max(np.abs(res['tau'][:,0])):.3f} N·m")
    print(f"Max |τ2|: {np.max(np.abs(res['tau'][:,1])):.3f} N·m")

    print("\nGenerating plots …")
    plot_joint_positions(res, args.traj, save=args.save)
    plot_joint_errors(res,    args.traj, save=args.save)
    plot_torques(res,         args.traj, save=args.save)
    plot_cartesian(res,       args.traj, save=args.save)

    print("Running gain analysis …")
    plot_gain_analysis(args.traj, args.duration, save=args.save)

    print(f"\nAll figures saved to: figures/")
    print("Done.\n")


if __name__ == "__main__":
    main()
