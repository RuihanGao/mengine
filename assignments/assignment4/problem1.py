import os
import numpy as np
import mengine as m
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

env = m.Env(gravity=[0, 0, 0])


def moveto(robot, robot_marker, pos):
    # moves robot and robot reference frame ('robot_marker') to position pos
    robot.set_base_pos_orient(
        pos+np.array([0.1, -0.1, 0.]), m.get_quaternion([np.radians(90), 0, 0]))
    robot_marker.set_base_pos_orient(
        pos, [0, 0, 0, 1])


def reset():
    # Create environment and ground plane
    env.reset()
    ground = m.Ground(position=[0, 0, -0.02])
    env.set_gui_camera(look_at_pos=[0.5, 0.5, 0], distance=0.7, pitch=-89.99)

    robot_init_position = np.array([0.0, 0, 0.0])
    robot = m.Shape(m.Mesh(filename=os.path.join(m.directory, 'triangle.obj'), scale=[
        1, 0.1, 1]), static=False, position=robot_init_position, orientation=m.get_quaternion([np.radians(90), 0, 0]), rgba=[0, 1, 0, 0.5])
    # mark robot reference frame
    robot_marker = m.Shape(m.Sphere(radius=0.02), static=False, collision=False,
                           position=robot_init_position+np.array([-0.1, 0.1, 0.]), rgba=[1, 1, 1, 1])

    l1 = 0.3; h1 = 0.48; l2 = 0.4; h2 = 0.36
    c1 = [0.5, 0.5]; c2 = (0.9, 0.75)
    obstacle1 = m.Shape(m.Box(half_extents=[l1/2, h1/2, 0.01]), static=True, position=[
                        c1[0], c1[1], 0.0], rgba=[1, 1, 0, 1])
    obstacle2 = m.Shape(m.Box(half_extents=[l2/2, h2/2, 0.01]), static=True, position=[
                        c2[0], c2[1], 0.0], rgba=[1, 1, 0, 1])

    m.step_simulation(realtime=True)


# ------ TODO Student answer below -------
    # Compute C-space obstacle for robot and obstacles using Minkowski difference
    # rewrite the obstacle in 2D space by marking the bottom-left and top-right corners

    rectangle1 = np.array([[c1-l1/2, c1-h1/2], [c1+l1/2, c1+h1/2]])
    triangle = np.array([[-0.1, 0.1], [0.1, 0.1], [0, -0.1]])

    # Compute the Minkowski sum
    minkowski_sum = []
    for t in triangle:
        for r in rectangle1:
            minkowski_sum.append(t+r)

    # Compute Minkowski difference by finding the convex hull of the Minkowski sum
    minkowski_difference_hull = ConvexHull(minkowski_sum)
    # Extract vertices of the Minkowski difference
    minkowski_difference_vertices = minkowski_sum[minkowski_difference_hull.vertices]


    # Plot the triangle
    plt.fill(triangle[:, 0], triangle[:, 1], color='blue', alpha=0.5, label='Triangle')

    # Plot the rectangle
    x, y = zip(*rectangle1)
    plt.fill(x, y, color='green', alpha=0.5, label='Rectangle')

    # Plot the Minkowski difference
    x, y = zip(*minkowski_difference_vertices)
    plt.fill(x, y, color='red', alpha=0.5, label='Minkowski Difference')

    # Set axis limits and labels
    plt.xlim(min(triangle[:, 0].min(), rectangle1[:, 0].min(), minkowski_difference_vertices[:, 0].min()) - 1,
            max(triangle[:, 0].max(), rectangle1[:, 0].max(), minkowski_difference_vertices[:, 0].max()) + 1)
    plt.ylim(min(triangle[:, 1].min(), rectangle1[:, 1].min(), minkowski_difference_vertices[:, 1].min()) - 1,
            max(triangle[:, 1].max(), rectangle1[:, 1].max(), minkowski_difference_vertices[:, 1].max()) + 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()




    # hints: if robot.get_contact_points() is not None, then robot is in collision.

    # you can draw points to the simulation scene at a position by calling
    # m.Shape(m.Sphere(radius=0.01), static=True, collision=False,
    # position=position, rgba=[1, 0, 0, 1])

    # you can get the robot's current position using
    # pos = robot.get_base_pos_orient()[0]

    while True:
        m.step_simulation(realtime=True)

# ------ Student answer above -------


reset()
