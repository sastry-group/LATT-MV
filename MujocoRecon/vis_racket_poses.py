import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import pickle

FILE = "pre_gt.pkl"
with open(FILE, "rb") as f:
    data = pickle.load(f)

target_poses, poses_achieved, where_landed = data

poses_achieved_filtered = []
target_poses_filtered = []

position_error = 0.
n = 0
rot_error = 0.
for i in range(len(poses_achieved)):
    if str(poses_achieved[i]) == 'None' or str(target_poses[i]) == 'None':
        continue
    target_poses_filtered.append(target_poses[i])
    poses_achieved_filtered.append(poses_achieved[i])
    n += 1
    position_error += np.linalg.norm(poses_achieved[i][:3] - target_poses[i][:3])
    quat1 = R.from_quat(poses_achieved[i][3:])
    quat2 = R.from_quat(target_poses[i][3:])
    # Angle between quaternions
    # print(poses_achieved[i][3:], target_poses[i][3:])
    # exit(0)
    quat_error = quat1.inv() * quat2
    # print(quat_error.magnitude())
    # exit(0)
    rot_error += quat_error.magnitude() * 180 / np.pi
    # np.arc

avg_position_error = position_error / n
avg_rot_error = rot_error / n
poses_achieved_filtered = np.array(poses_achieved_filtered)
target_poses_filtered = np.array(target_poses_filtered)

def plot_cylinder(ax, position, quaternion, radius=0.15, height=0.02, resolution=20, color='c'):
    """
    Plots a closed cylinder in 3D space given a position and quaternion orientation.

    :param ax: The matplotlib 3D axis to plot on.
    :param position: (x, y, z) position of the cylinder center.
    :param quaternion: (x, y, z, w) quaternion for cylinder orientation.
    :param radius: Radius of the cylinder.
    :param height: Height of the cylinder.
    :param resolution: Number of points around the cylinder circumference.
    """
    # Create a circle in the XY plane (base of the cylinder)
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(-height / 2, height / 2, 2)  # Two points for cylinder height

    # Create the grid for cylinder sides
    X, Z = np.meshgrid(x, z)
    Y, Z = np.meshgrid(y, z)

    # Stack coordinates for transformation
    points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)

    # Rotate points with the quaternion
    rotation = R.from_quat(quaternion)
    points_rotated = rotation.apply(points)

    # Translate points
    points_rotated += position

    # Reshape back into grids for plotting
    X_rot = points_rotated[:, 0].reshape(X.shape)
    Y_rot = points_rotated[:, 1].reshape(Y.shape)
    Z_rot = points_rotated[:, 2].reshape(Z.shape)

    # Plot the surface (cylinder sides)
    ax.plot_surface(X_rot, Y_rot, Z_rot, color=color, alpha=0.6)

    # Create and plot circular end caps
    for z_pos in [-height / 2, height / 2]:  # Bottom and top of the cylinder
        # End cap grid in polar coordinates
        theta_cap = np.linspace(0, 2 * np.pi, resolution)
        r_cap = np.linspace(0, radius, resolution // 2)
        R_cap, Theta_cap = np.meshgrid(r_cap, theta_cap)
        
        X_cap = R_cap * np.cos(Theta_cap)
        Y_cap = R_cap * np.sin(Theta_cap)
        Z_cap = np.full_like(X_cap, z_pos)

        # Stack coordinates for transformation
        cap_points = np.stack((X_cap.flatten(), Y_cap.flatten(), Z_cap.flatten()), axis=1)
        cap_points_rotated = rotation.apply(cap_points) + position

        # Reshape back into grid for plotting
        X_cap_rot = cap_points_rotated[:, 0].reshape(X_cap.shape)
        Y_cap_rot = cap_points_rotated[:, 1].reshape(Y_cap.shape)
        Z_cap_rot = cap_points_rotated[:, 2].reshape(Z_cap.shape)

        # Plot the circular cap
        ax.plot_surface(X_cap_rot, Y_cap_rot, Z_cap_rot, color=color, alpha=0.6)

def plot_filled_cuboid(ax, corner1, corner2, color='orange', alpha=0.3):
    """
    Plots a filled cuboid given two opposite corners.

    :param ax: The matplotlib 3D axis to plot on.
    :param corner1: Coordinates of the first corner (x, y, z).
    :param corner2: Coordinates of the opposite corner (x, y, z).
    :param color: Color of the cuboid.
    :param alpha: Transparency of the cuboid.
    """
    # Generate the coordinates of the cuboid's 8 corners
    x_vals = [corner1[0], corner2[0]]
    y_vals = [corner1[1], corner2[1]]
    z_vals = [corner1[2], corner2[2]]
    corners = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals])

    # print(corners[2],corners[3],corners[6],corners[7])
    # Define 6 faces by selecting corners
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # Bottom face
        [corners[4], corners[5], corners[6], corners[7]],  # Top face
        [corners[0], corners[2], corners[4], corners[6]],  # Side face 1
        [corners[1], corners[3], corners[5], corners[7]],  # Side face 2
        [corners[0], corners[1], corners[4], corners[5]],  # Side face 3
        [corners[2], corners[3], corners[6], corners[7]],  # Side face 4
    ]

    # Plot each face as a filled surface
    for face in faces:
        face = np.array(face)
        ax.plot_surface(
            face[:, 0].reshape(2, 2),
            face[:, 1].reshape(2, 2),
            face[:, 2].reshape(2, 2),
            color=color,
            alpha=alpha,
            rstride=1,
            cstride=1
        )

poses_achieved_filtered[:,0] -= 1.5
poses_achieved_filtered[:,2] -= 0.66

target_poses_filtered[:,0] -= 1.5
target_poses_filtered[:,2] -= 0.66

shuffler = np.random.permutation(len(poses_achieved_filtered))[:10]
# Define cylinder parameters
positions = poses_achieved_filtered[shuffler, :3]  # Example positions
quaternions = poses_achieved_filtered[shuffler,3:]  # Example quaternions

positions_ = target_poses_filtered[shuffler, :3]  # Example positions
quaternions_ = target_poses_filtered[shuffler,3:]  # Example quaternions

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot each cylinder
for position, quaternion in zip(positions, quaternions):
    plot_cylinder(ax, position, quaternion, color='r')

for position_, quaternion_ in zip(positions_, quaternions_):
    plot_cylinder(ax, position_, quaternion_, color='g')

rot_mat = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
position = [[-1.5, 0., 0.14]]
quaternion = [R.from_matrix(rot_mat).as_quat()]
plot_cylinder(ax, position, quaternion, color='b')

corner1 = [-1.37, -0.76, -0.05]
corner2 = [1.37, 0.76, 0.05]
plot_filled_cuboid(ax, corner1, corner2)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
plt.show()

with open(FILE[:-4]+'.txt', "a") as f:
    f.write("Mean position error: "+str(avg_position_error)+'\n')
    f.write("Mean rot error in x: "+str(avg_rot_error)+'\n')
    
