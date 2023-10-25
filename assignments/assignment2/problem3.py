import os
import sys
import time

import numpy as np

import mengine as m


def invertQ(q):
    """
    Invert a quaternion
    """
    # ------ TODO Student answer below -------
    qx, qy, qz, qw = q
    q_conjugate = np.array([-qx, -qy, -qz, qw])
    q_sq = qx**2 + qy**2 + qz**2 + qw**2
    q_inverse = q_conjugate / q_sq
    return q_inverse
    # ------ Student answer above -------


def line_intersection(p1, p2, q1, q2):
    """
    Find the intersection of two 3D line segments p1-p2 and q1-q2.
    If there is an intersection, returns the point. Otherwise, returns None.
    """
    # ------ TODO Student answer below -------
    # assume the intersection (x, y, z) = p1 + t1 * (p2 - p1)
    # and (x, y, z) = q1 + t2 * (q2 - q1)
    # then we solve linear system
    # print(f"p1 is {p1}")
    # print(f"p2 is {p2}")
    # print(f"q1 is {q1}")
    # print(f"q2 is {q2}")

    p1x, p1y, p1z = p1
    p2x, p2y, p2z = p2
    q1x, q1y, q1z = q1
    q2x, q2y, q2z = q2
    A = np.array([[p2x - p1x, q1x - q2x], [p2z - p1z, q1z - q2z]])
    b = np.array([q1x - p1x, q1z - p1z])

    # print(f"A is \n {A}")
    # print(f"b is {b}")
    t1, t2 = np.linalg.solve(A, b)
    # substitute t1 to get line on p1-p2
    x1 = p1x + t1 * (p2x - p1x)
    y1 = p1y + t1 * (p2y - p1y)
    z1 = p1z + t1 * (p2z - p1z)

    # substitute t2 to get line on q1-q2
    x2 = q1x + t2 * (q2x - q1x)
    y2 = q1y + t2 * (q2y - q1y)
    z2 = q1z + t2 * (q2z - q1z)
    # check if the lines intersect
    if np.isclose(x1, x2, atol=0.01) and np.isclose(y1, y2, atol=0.01) and np.isclose(z1, z2, atol=0.01):
        if t1 >= 0 and t1 <= 1 and t2 >= 0 and t2 <= 1:
            return np.array([x1, y1, z1])
        else:
            return None
    else:
        return None
    # ------ Student answer above -------


def quaternion_to_rotation_matrix(q):
    # input: q: quaternion
    # output: R: rotation matrix shape (3, 3)
    qx, qy, qz, qw = q
    R = np.array(
        [
            [qw**2 + qx**2 - qy**2 - qz**2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), qw**2 - qx**2 + qy**2 - qz**2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw**2 - qx**2 - qy**2 + qz**2],
        ]
    )
    return R


# Create environment and ground plane
env = m.Env()
# ground = m.Ground()
env.set_gui_camera(look_at_pos=[0, 0, 0], yaw=30)

fbl = m.URDF(
    filename=os.path.join(m.directory, "fourbarlinkage.urdf"),
    static=True,
    position=[0, 0, 0.3],
    orientation=[0, 0, 0, 1],
)
fbl.controllable_joints = [0, 1, 2]
# Create a constraint for the 4th joint to create a closed loop
fbl.create_constraint(
    parent_link=1,
    child=fbl,
    child_link=4,
    joint_type=m.p.JOINT_POINT2POINT,
    joint_axis=[0, 0, 0],
    parent_pos=[0, 0, 0],
    child_pos=[0, 0, 0],
)
m.step_simulation(steps=20, realtime=False)

coupler_links = [1, 3, 5]

links = [1, 3]
global_points = []
previous_global_points = []
lines = [None, None]
lines_start_end = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]

for link in links:
    global_points.append(fbl.get_link_pos_orient(link)[0])
    previous_global_points.append(global_points[-1])
    point = m.Shape(m.Sphere(radius=0.02), static=True, position=global_points[-1], rgba=[0, 0, 1, 1])

intersect_points_local = []
intersect_points_local_bodies = []

for i in range(10000):
    fbl.control([np.radians(i)] * 3)

    if i > 3:
        for j, (link, global_position, previous_global_position) in enumerate(
            zip(links, global_points, previous_global_points)
        ):
            p_new = fbl.get_link_pos_orient(link)[0]
            ic_vector_of_motion = p_new - previous_global_position
            ic_bisector = np.cross(ic_vector_of_motion, [0, 1, 0])
            ic_bisector = ic_bisector / np.linalg.norm(ic_bisector)
            previous_global_points[j] = p_new

            lines[j] = m.Line(
                p_new - ic_bisector, p_new + ic_bisector, radius=0.005, rgba=[0, 0, 1, 0.5], replace_line=lines[j]
            )
            lines_start_end[j] = (p_new - ic_bisector, p_new + ic_bisector)

        if len(intersect_points_local) < 400:
            # stop drawing if we have drawn 500 points
            try:
                intersect_point = line_intersection(
                    lines_start_end[0][0], lines_start_end[0][1], lines_start_end[1][0], lines_start_end[1][1]
                )
            except np.linalg.LinAlgError:
                print(f"LinAlgError at i={i}")
                sys.exit(1)

            if intersect_point is not None:
                # print(f"We find intersection point at i={i}, {intersect_point}")
                m.Shape(
                    m.Sphere(radius=0.005), static=True, position=intersect_point, collision=False, rgba=[1, 0, 0, 1]
                )
                # ------ TODO Student answer below -------
                # draw moving centrode
                # get intersection point in local frame w.r.t. link 4
                # To compute the moving centrode, we need to transform the coordinates of the intersection point from the global frame to the local frame of link 4.

                # compute the coordinate transfrom from the base link to link 4
                # Hint: You can use Body.get_link_pos_orient(link) to get the position and orientation of a link
                # get the position and orientation of link 4
                link4_pos, link4_orient = fbl.get_link_pos_orient(4)
                # compute the inverse of the orientation of link 4
                link4_orient_inverse = invertQ(link4_orient)

                # compute the coordinate transform from the base link to link 4
                link4_R = quaternion_to_rotation_matrix(link4_orient_inverse)
                # compute the inverse of the position of link 4
                link4_pos_inverse = -link4_R @ link4_pos
                link4_transform = np.hstack((link4_R, link4_pos_inverse.reshape(3, 1)))
                link4_transform = np.vstack((link4_transform, np.array([0, 0, 0, 1])))
                # apply the transform to the intersection point
                local_intersect_point = np.dot(link4_transform, np.hstack((intersect_point, 1)))
                local_intersect_point = local_intersect_point[:3]
                # ------ Student answer above -------

                intersect_points_local.append(local_intersect_point)
                # get global coordinates of intersection point
                intersect_point_local_body = m.Shape(
                    m.Sphere(radius=0.005), static=True, position=intersect_point, collision=False, rgba=[0, 1, 0, 1]
                )
                intersect_points_local_bodies.append(intersect_point_local_body)

        # redraw intersection points of moving centrode
        # ------ TODO Student answer below -------
        # Hint: You can use Body.set_base_pos_orient(xyz) to update a body's position
        # Note: we redraw here to manually move the green points to the correct position (in global frame)
        # We put the redrawing outside the loop to accelerate the simulation
        for body, point_local in zip(intersect_points_local_bodies, intersect_points_local):
            link4_pos, link4_orient = fbl.get_link_pos_orient(4)
            link4_orient_R = quaternion_to_rotation_matrix(link4_orient)
            link4_transform_in_global_frame = np.hstack((link4_orient_R, link4_pos.reshape(3, 1)))
            link4_transform_in_global_frame = np.vstack((link4_transform_in_global_frame, np.array([0, 0, 0, 1])))
            # point_local is in link4 coord frame, we want to transform it to global frame
            point_local_in_global_frame = link4_transform_in_global_frame @ np.hstack((point_local, 1))
            body.set_base_pos_orient(point_local_in_global_frame[:3])
        # ------ Student answer above -------

    m.step_simulation(realtime=True)

    if i == 500 or i == 600 or i == 700:
        print("Please save screenshot and include in writeup")
        input("Press Enter to continue...")
