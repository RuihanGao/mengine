import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

import mengine as m

np.set_printoptions(precision=3, suppress=True)

# NOTE: This assignment asks you to Implement FK using screw coordinates.
# Create environment and ground plane
env = m.Env()

# Robot configuration
l1 = 0.5
l2 = 0.4
l3 = 0.3


def reset_sim():
    env.reset()
    ground = m.Ground([0, 0, -0.5])
    env.set_gui_camera(look_at_pos=[0, 0, 0.5], distance=1.5)
    # Create example robot
    robot = m.URDF(
        filename=os.path.join(m.directory, "assignments", "example_arm.urdf"), static=True, position=[0, 0, 0]
    )
    robot.controllable_joints = [0, 1, 2]
    robot.end_effector = 3
    robot.update_joint_limits()
    return robot


def sample_configuration():
    # Sample a random configuration for the robot.
    # NOTE: Be conscious of joint angle limits
    # output: q: joint angles of the robot
    return np.random.uniform(low=[-np.pi, -np.pi / 2, -np.pi / 2], high=[np.pi / 2, np.pi / 2, np.pi / 2])


def get_exp_coordinates(omega, v, theta):
    # NOTE It can be helpful to implement and use this function,
    # but it is not required. You can also perform all calculations in calculate_FK.
    # You should use expm(w*theta) to compute the matrix exponential
    #
    # Calculate the exponential coordinates (exp([S]theta)) of a screw
    # input: omega: angular part
    #        v: linear part
    #        theta: angle of rotation
    # output: E: exponential coordinates of the screw (4x4 matrix)
    # ------ TODO Student answer below -------
    # print(f"In function get_exp_coordinates, w shape is {omega.shape}, v shape is {v.shape}") # (3, ), (3, )
    omega_matrix = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    exp_omega_theta = expm(omega_matrix * theta) # shape [3,3]
    rightmost_col = (
        theta * np.eye(3)
        + (1 - np.cos(theta)) * omega_matrix
        + (theta - np.sin(theta)) * omega_matrix @ omega_matrix
    )
    rightmost_col = np.dot(rightmost_col, v)
    rightmost_col = rightmost_col[:, np.newaxis]
    E = np.concatenate((exp_omega_theta, rightmost_col), axis=1)
    E = np.concatenate((E, np.array([[0, 0, 0, 1]])), axis=0)
    return E
    # ------ Student answer above -------


def calculate_FK(q, joint=3):
    # Calculate the forward kinematics of the robot
    # NOTE: Use screw coordinate representation and the product of exponentials formulation
    # You should use expm(w*theta) to compute the matrix exponential
    # input: q: joint angles of the robot
    #        joint: index of the joint to calculate the FK for. 0 is the base joint, and 3 is the end effector
    # output: ee_position: position of the end effector
    #         ee_orientation: orientation of the end effector as a quaternion
    # ------ TODO Student answer below -------
    M4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, l1 + l2 + l3], [0, 0, 0, 1]])
    M3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, l1 + l2], [0, 0, 0, 1]])
    M2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, l1], [0, 0, 0, 1]])
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # # S1, S2, S3 in vector form
    # S3 = np.array([1, 0, 0, 0, l1 + l2, 0])  # w1 = 1 rad/s about the x axis
    # S2 = np.array([1, 0, 0, 0, l1, 0])  # w2 = 1 rad/s about the x axis
    # S1 = np.array([0, 0, 1, 0, 0, 0])  # w3 = 1 rad/s about the z axis

    w1 = np.array([0, 0, 1])
    w2 = np.array([1, 0, 0])
    w3 = np.array([1, 0, 0])
    v1 = np.array([0, 0, 0])
    v2 = np.array([0, l1, 0])
    v3 = np.array([0, l1 + l2, 0])

    # # write down the 4x4 matrix form for S1, S2, S3
    # S1 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    # S2 = np.array([[0, 0, 0, 0], [0, 0, -1, l1], [0, 1, 0, 0], [0, 0, 0, 0]])
    # S3 = np.array([0, 0, 0, 0], [0, 0, -1, l1], [0, 1, 0, 0], [0, 0, 0, 0])

    # use the function get_exp_coordinates
    E1 = get_exp_coordinates(w1, v1, q[0])
    E2 = get_exp_coordinates(w2, v2, q[1])
    E3 = get_exp_coordinates(w3, v3, q[2])

    if joint == 3:
        pose = E1 @ E2 @ E3 @ M4
    elif joint == 2:
        pose = E1 @ E2 @ M3
    elif joint == 1:
        pose = E1 @ M2
    elif joint == 0:
        pose = M1

    ee_position = pose[:3, 3]
    ee_orientation = pose[:3, :3]

    ee_orientation = m.get_quaternion(
        ee_orientation
    )  # NOTE: If you used transformation matrices, call this function to get a quaternion
    return ee_position, ee_orientation
    # ------ Student answer above -------


def compare_FK(ee_positions, ee_positions_pb, ee_orientations, ee_orientations_pb):
    # Compare the FK implementation to the built-in one
    # input: ee_positions: list of positions of the end effector
    #        ee_positions_pb: list of positions of the end effector from pybullet
    #        ee_orientations: list of orientations of the end effector
    #        ee_orientations_pb: list of orientations of the end effector from pybullet
    distance_error_sum = 0
    orientation_error_sum = 0
    for p1, p2 in zip(ee_positions, ee_positions_pb):
        distance_error_sum += np.linalg.norm(p1 - p2)
    for q1, q2 in zip(ee_orientations, ee_orientations_pb):
        error = np.arccos(2 * np.square(q1.dot(q2)) - 1)
        orientation_error_sum += 0 if np.isnan(error) else error
    print("Average FK distance error:", distance_error_sum / len(ee_positions))
    print("Average FK orientation error:", orientation_error_sum / len(ee_orientations))


# ##########################################
# Problem 2:
# Forward Kinematics using screw coordinates
# ##########################################
robot = reset_sim()
# test cases
q_test = np.array([[0, 0, 0], [-0.3, 0.7, 0.9], [0.8, 1.4, 1.2]])
for q_i, idx in zip(q_test, range(3)):
    ee_pos, ee_orient = calculate_FK(q_i, joint=3)
    print("ee position and orientation for testcase ", idx, ": ", ee_pos, ee_orient)

ee_positions = []
ee_orientations = []

ee_positions_pb = []
ee_orientations_pb = []

for i in range(1000):
    # sample a random configuration q
    q = sample_configuration()
    # move robot into configuration q
    robot.control(q, set_instantly=True)
    m.step_simulation(realtime=True)
    # calculate ee_position, ee_orientation using calculate_FK
    ee_position, ee_orientation = calculate_FK(q, joint=3)

    ee_positions.append(ee_position)
    ee_orientations.append(ee_orientation)
    # calculate ee position, orientation using pybullet's FK
    ee_position_pb, ee_orientation_pb = robot.get_link_pos_orient(robot.end_effector)
    ee_positions_pb.append(ee_position_pb)

    ee_orientations_pb.append(ee_orientation_pb)
    # print(ee_position, ee_position_pb, ee_orientation, ee_orientation_pb)
    # m.Shape(m.Sphere(radius=0.02), static=True, position=ee_position, collision=False, rgba=[1, 0, 0, 1])
    # m.Shape(m.Sphere(radius=0.02), static=True, position=ee_position_pb, collision=False, rgba=[0, 1, 0, 1])

# compare your implementation and pybullet's FK
compare_FK(ee_positions, ee_positions_pb, ee_orientations, ee_orientations_pb)

# ##########################################
