import sys
import numpy as np
import modernrobotics as mr
import ur5sim as sim


def my_params():
    """Return screw-axis and home config. in space frame for UR5 scene

    :return S_list: The joint screw axis in the space frame when the
                    manipulator is at home position, cols as screw-axis vector
    :return M: The home configuration of the end-effector
    """
    # UR5 link parameters (in meters)
    L1, L2 = (0.425, 0.392)
    W1, W2 = (0.110, 0.0814)
    H1, H2 = (0.089, 0.080)

    # Screw-axis spacial form
    S_list = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, -H1, 0, 0],
        [0, 1, 0, -(H1 + L1), 0, 0],
        [0, 1, 0, -(H1 + L1 + L2), 0, 0],
        [0, 0, -1, -W1, 0, 0],
        [0, 1, 0, -(H1 + L1 + L2 + H2), 0, 0]
    ]).T

    # Home position
    M = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, W1 + W2],
        [0, 1, 0, H1 + L1 + L2 + H2],
        [0, 0, 0, 1]
    ])
    return S_list, M


def mr_params():
    """Return screw-axis and home config. in space frame for UR5_mr scene

    :return S_list: The joint screw axis in the space frame when the
                    manipulator is at home position, cols as screw-axis vector
    :return M: The home configuration of the end-effector"""
    # UR5 link parameters
    L1, L2 = (0.425, 0.392)
    W1, W2 = (0.109, 0.082)
    H1, H2 = (0.089, 0.095)

    # End-effector frame at zero position
    M = np.array([
        [-1, 0, 0, L1 + L2],
        [0, 0, 1, W1 + W2],
        [0, 1, 0, H1 - H2],
        [0, 0, 0, 1]
    ])

    # Screw axes
    S_list = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, -H1, 0, 0],
        [0, 1, 0, -H1, 0, L1],
        [0, 1, 0, -H1, 0, L1 + L2],
        [0, 0, -1, -W1, L1 + L2, 0],
        [0, 1, 0, H2 - H1, 0, L1 + L2]
    ]).T
    return S_list, M


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)  # Human printing

    # 0. Get the geometric parameters for PoE
    S_list, M = mr_params()

    # 1. Forward kinematics with PoE
    theta_d = np.array([0, -np.pi / 2, 0, 0, np.pi / 2, 0])
    T_d = mr.FKinSpace(M, S_list, theta_d)
    print('Desired config: \n{}\n'.format(T_d))

    # 2. Compute inverse kinematics with PoE and Newton-Raphson
    theta0 = np.array([0, 0, np.pi / 8, 0, 0, np.pi / 8])
    eps_omg = 1e-2
    eps_v = 1e-3
    theta_sol, ok = mr.IKinSpace(S_list, M, T_d, theta0, eps_omg, eps_v)
    if ok:
        print('IK solution: {}\n'.format(theta_sol))
    else:
        print('Not found solution within desired config and tolerances')
        sys.exit()

    # 3. Connect to sim, move robot and compare
    ur5 = sim.UR5Sim()
    ur5.connect()  # Connect to simulation-server
    ur5.set_joints(theta_sol)  # Move robot to ik-solution
    T_nw = ur5.get_SE3()  # Get the current config
    print('New config: \n{}\n'.format(T_nw))
