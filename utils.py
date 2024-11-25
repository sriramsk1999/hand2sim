import imageio
import numpy as np
import open3d as o3d


def vis_trajectory(traj):
    """
    A utility function to visualize trajectory
    """
    frames = []
    trajectory_points = []
    for pose in traj:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        frame.transform(pose)
        frames.append(frame)
        # Extract the translation (position) of each pose for the trajectory line
        trajectory_points.append(pose[:3, 3])

    # Create a LineSet to visualize the trajectory as a line between points
    trajectory_points = np.array(trajectory_points)
    lines = [
        [i, i + 1] for i in range(len(trajectory_points) - 1)
    ]  # Connect points with lines
    colors = [[0, 1, 0] for _ in lines]  # Green color for the lines

    # Create the LineSet object
    trajectory_line = o3d.geometry.LineSet()
    trajectory_line.points = o3d.utility.Vector3dVector(trajectory_points)
    trajectory_line.lines = o3d.utility.Vector2iVector(lines)
    trajectory_line.colors = o3d.utility.Vector3dVector(colors)

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_sphere.paint_uniform_color([1, 0, 0])  # Red for start
    start_sphere.translate(trajectory_points[0])

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    end_sphere.paint_uniform_color([0, 0, 1])  # Blue for end
    end_sphere.translate(trajectory_points[-1])

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Visualize everything
    o3d.visualization.draw_geometries(
        frames + [trajectory_line, start_sphere, end_sphere, origin_frame]
    )


def set_initial_ee_target(env, goal_sid, translation, rotation):
    env.sim.model.site_rgba[goal_sid][3] = 0.2  # make visible
    # place ee target in a more suitable location / orientation
    env.sim.model.site_pos[goal_sid] = translation
    env.sim.model.site_quat[goal_sid] = rotation
    env.sim.forward()
    return


def read_video_from_path(path):
    reader = imageio.get_reader(path)
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)
