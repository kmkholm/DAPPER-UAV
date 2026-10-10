"""Flight-phase ground truth from the drone-side 5 Hz state log (gt_<sysid>.csv).

The rules use the autopilot mode, the mission command being executed and the
vehicle kinematics; they are fixed before any model is trained.
GLOBAL_POSITION_INT vz is positive downwards (NED).
"""
import numpy as np
import pandas as pd

COPTER_CLASSES = ["Ground", "Ascent", "Hover", "Cruise", "Orbit", "Teleop", "Descent"]
PLANE_CLASSES = ["Ground", "VTOL climb", "VTOL descent", "Cruise", "Loiter", "Teleop"]

# ArduCopter / ArduPlane custom modes and MAV_CMD ids used below
C_AUTO, C_CIRCLE = 3, 7
P_CIRCLE, P_AUTO, P_RTL, P_LOITER, P_QHOVER, P_QLOITER, P_QLAND, P_QRTL = 1, 10, 11, 12, 18, 19, 20, 21
CMD_LOITER_UNLIM, CMD_LOITER_TURNS, CMD_LOITER_TIME, CMD_LOITER_TO_ALT = 17, 18, 19, 31
CMD_VTOL_TAKEOFF, CMD_VTOL_LAND = 84, 85


def load_gt(path):
    g = pd.read_csv(path)
    g = g.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return g


def label_copter(g):
    hs = np.hypot(g.vx.values, g.vy.values)
    vz, alt = g.vz.values, g.alt.values
    lab = np.full(len(g), 3, dtype=np.int8)                       # Cruise by default
    lab[hs < 0.7] = 2                                              # Hover
    lab[(vz > 0.4) & (hs < 1.5)] = 6                               # Descent
    lab[(-vz > 0.4) & (hs < 1.5)] = 1                              # Ascent
    orbit = (g["mode"].values == C_CIRCLE) | ((g["mode"].values == C_AUTO) & (g.cmd.values == CMD_LOITER_TURNS))
    lab[orbit & (hs >= 0.7)] = 4
    lab[(g.setpoint.values == 1) | (g.override.values == 1)] = 5   # Teleop
    ground = (g.armed.values == 0) | ((alt < 0.5) & (hs < 0.3) & (np.abs(vz) < 0.3))
    lab[ground] = 0
    return lab


def label_plane(g):
    hs = np.hypot(g.vx.values, g.vy.values)
    vz, alt, mode, cmd = g.vz.values, g.alt.values, g["mode"].values, g.cmd.values
    lab = np.full(len(g), 3, dtype=np.int8)                       # Cruise
    loiter = np.isin(mode, [P_LOITER, P_CIRCLE]) | ((mode == P_AUTO) & np.isin(cmd, [CMD_LOITER_UNLIM, CMD_LOITER_TURNS,
                                                                                     CMD_LOITER_TIME, CMD_LOITER_TO_ALT]))
    lab[loiter] = 4
    vtol = (hs < 7.0) | np.isin(mode, [P_QHOVER, P_QLOITER, P_QLAND, P_QRTL]) | ((mode == P_AUTO) & np.isin(cmd, [CMD_VTOL_TAKEOFF, CMD_VTOL_LAND]))
    down = np.isin(mode, [P_QLAND, P_QRTL]) | ((mode == P_AUTO) & (cmd == CMD_VTOL_LAND)) | ((mode == P_RTL) & (hs < 7.0)) | (vz > 0.3)
    lab[vtol & ~down] = 1
    lab[vtol & down] = 2
    lab[g.override.values == 1] = 5
    ground = (g.armed.values == 0) | ((alt < 1.0) & (hs < 0.5) & (np.abs(vz) < 0.3))
    lab[ground] = 0
    return lab


def label(g, vehicle):
    return label_copter(g) if vehicle == "copter" else label_plane(g)
