import sim
import time
import numpy as np


def euler2so3(angles):
    """Rotation matrix in so3 of Euler-XYZ angles"""
    c1, s1 = np.cos(angles[0]), np.sin(angles[0])
    c2, s2 = np.cos(angles[1]), np.sin(angles[1])
    c3, s3 = np.cos(angles[2]), np.sin(angles[2])
    return np.block([
        [c2 * c3, -c2 * s3, s2],
        [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
        [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]
    ])


class UR5Sim(object):
    def __init__(self):
        self.clientId = -1  # not connected
        self.joints = []  # robot joints
        self.frames = {'s': None, 'b': None}  # Robot frames
        pass

    def is_connected(self):
        if self.clientId != -1:
            return True
        else:
            print('Error client not connected')
            return False

    def get_obj(self, name: str):
        if self.is_connected():
            res, obj = sim.simxGetObjectHandle(
                self.clientId, name, sim.simx_opmode_blocking)
            if res == sim.simx_return_ok:
                return obj
            else:
                print('Error to get {}'.format(name))
                return None
        else:
            return None

    def connect(self, host='127.0.0.1', port=19999):
        sim.simxFinish(-1)  # close all connections
        self.clientId = sim.simxStart(host, port, True, True, 5000, 5)  # connect
        if self.clientId != -1:
            # Get robot frames {s},{b}
            self.frames['s'] = self.get_obj('Base_Frame')  # spatial frame
            self.frames['b'] = self.get_obj('EE_Frame')  # body frame
            for i in range(6):
                # Get robot joints [j1,...,j6]
                self.joints.append(self.get_obj('UR5_joint' + str(i + 1)))
            # print('Connection success!')
            return True
        else:
            print('Error to connect to {}:{}'.format(host, port))
            return False

    def set_joints(self, theta):
        if self.is_connected():
            for i in range(6):
                sim.simxSetJointPosition(
                    self.clientId,
                    self.joints[i],
                    theta[i],
                    sim.simx_opmode_oneshot
                )
            time.sleep(0.5)

    def get_joints(self):
        theta = []
        if self.is_connected():
            for i in range(6):
                _, val = sim.simxGetJointPosition(
                    self.clientId,
                    self.joints[i],
                    sim.simx_opmode_oneshot_wait
                )
                theta.append(val)
        return np.array(theta)

    def get_SE3(self):
        if self.is_connected():
            _, pos = sim.simxGetObjectPosition(
                self.clientId, self.frames['b'], self.frames['s'], sim.simx_opmode_oneshot_wait)
            _, ori = sim.simxGetObjectOrientation(
                self.clientId, self.frames['b'], self.frames['s'], sim.simx_opmode_oneshot_wait)
            R = euler2so3(ori)
            p = np.array(pos).reshape((3, 1))
            return np.block([[R, p], [0, 0, 0, 1]])
        else:
            return np.zeros((4, 4))
