import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

def create_circle(center, axis, R, num_points=100):
    # Generate points on a circle in the xy-plane
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle = R * np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    # Align the circle with the given axis
    if not np.allclose(axis, [0, 0, 1]):
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        c = np.dot(z_axis, axis)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
        circle = rotation_matrix.dot(circle)
    
    # Translate circle to the center
    circle = circle.T + center

    return circle.T

RESCALE_FACTOR = 0.003048
def render(processed_video, fps, show_feet=False, show_extended=False):
    min_frame, max_frame = 0, processed_video.num_frames if show_extended else len(processed_video)
    scene = {
        i: processed_video[i] for i in range(max_frame)
    }
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the vertices of the table
    table_dims = [900 * RESCALE_FACTOR, 500 * RESCALE_FACTOR, 250 * RESCALE_FACTOR]
    
    # Add the ping pong table to the scene
    def add_table(ax):
        table_x = [-table_dims[0] / 2, table_dims[0] / 2, table_dims[0] / 2, -table_dims[0] / 2, -table_dims[0] / 2]
        table_y = [-table_dims[1] / 2, -table_dims[1] / 2, table_dims[1] / 2, table_dims[1] / 2, -table_dims[1] / 2]
        table_z = [table_dims[2]] * 5
        
        # Draw the table outline
        ax.plot(table_x, table_y, table_z, color='b')
        
        # Add a colored surface for the table top
        X, Y = np.meshgrid([-table_dims[0] / 2, table_dims[0] / 2], [-table_dims[1] / 2, table_dims[1] / 2])
        Z = np.full_like(X, table_dims[2])
        ax.plot_surface(X, Y, Z, color='darkblue', alpha=0.5)

    bounds = 1300 * RESCALE_FACTOR
    ball_trajectory = []

    def update(frame):            
        ax.cla()  # Clear the current axes
        if frame in scene:
            ax.set_xlim(-bounds, bounds)
            ax.set_ylim(-bounds, bounds)
            ax.set_zlim(0, bounds)
            
            p1_keypoints, p2_keypoints, ball_pos = scene[frame]
            
            # Plot the players.
            points = np.concatenate((p1_keypoints, p2_keypoints), axis=0) * RESCALE_FACTOR
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5)
            
            if ball_pos is not None:
                ax.scatter(ball_pos[0] * RESCALE_FACTOR, ball_pos[1] * RESCALE_FACTOR, ball_pos[2] * RESCALE_FACTOR, s=4, color="green")
            if show_feet:
                p1_feet = processed_video.player1_feet_2d[frame]
                p2_feet = processed_video.player2_feet_2d[frame]
                ax.scatter(p1_feet[:, 0] * RESCALE_FACTOR, p1_feet[:, 1] * RESCALE_FACTOR, [0, 0], s=1.5, color="red")
                ax.scatter(p2_feet[:, 0] * RESCALE_FACTOR, p2_feet[:, 1] * RESCALE_FACTOR, [0, 0], s=1.5, color="red")
            if frame == 0:
                ball_trajectory.clear()
            if ball_pos is not None:
                ball_trajectory.append(ball_pos)
                # Keep only the last 3 positions
                if len(ball_trajectory) > 5:
                    ball_trajectory.pop(0)
                ball_trajectory_arr = np.array(ball_trajectory)
                ax.plot(ball_trajectory_arr[:, 0] * RESCALE_FACTOR, ball_trajectory_arr[:, 1] * RESCALE_FACTOR, ball_trajectory_arr[:, 2] * RESCALE_FACTOR, color="red")
            add_table(ax)
                
            # Add title and labels to the axes
            ax.set_title('3D Reconstruction of Ping Pong Rally')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')
            
        return ax,
    
    
    ani = FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=1000/fps, repeat_delay=2000, repeat=True)  # interval in milliseconds
    ani.save("rec.gif", writer='pillow')
    plt.show()