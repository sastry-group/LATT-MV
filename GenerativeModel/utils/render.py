import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.animation import FuncAnimation
import pickle
from .analysis import generate_reachable_box, correct_pred

RESCALE_FACTOR = 0.003048

class MultiSampleVideoRenderer:
    
    def __init__(self, data, pad1_data, pad2_data, preds_start_index):
        self.videos = []
        self.preds_start_index = preds_start_index
        for i in range(len(data)):
            if i > 0 and preds_start_index is not None:
                data[i][preds_start_index:, 52, :] = correct_pred(data[i][preds_start_index:, 52, :])
            v = VideoRenderer(data[i], pad1_data[i], pad2_data[i])
            self.videos.append(v)
        
        ball_positions = []
        for video in self.videos[1:]:
            ball_positions.append(video.ball)
        ball_positions = np.array(ball_positions)
        self.mean_ball_pos = ball_positions.mean(axis=0)
        self.std_ball_pos  = ball_positions.std(axis=0)
        
    def save(self):
        with open('rec.pkl', 'wb') as f:
            pickle.dump(self, f)
        
    def render(self):
        render(self.videos, fps=self.videos[0].fps, preds_start_index=self.preds_start_index, mean_ball_pos=self.mean_ball_pos, std_ball_pos=self.std_ball_pos)
    
class VideoRenderer:
    
    def __init__(self, data, pad1_data, pad2_data):
        self.load(data, pad1_data, pad2_data)
        
    def load(self, data, pad1_data, pad2_data):        
        # Load metadata
        self.fps = data[0, 0, 0]
        self.num_frames = int(data[0, 0, 1])
        self.num_frames_usable = int(data[0, 0, 2])
        
        # Load player and ball data
        self.player1 = data[:, 2:27, :]  # Assuming 44 keypoints for each player
        self.player2 = data[:, 27:52, :]
        self.ball    = data[:, 52, :]
        self.pad1_data = pad1_data
        self.pad2_data = pad2_data
        # print(data.shape, self.ball.shape)
        # Replace NaN values with None for ball positions
        self.ball = [b if not np.isnan(b).any() else None for b in self.ball]
        
    def __getitem__(self, i):
        return self.player1[i], self.player2[i], self.ball[i], self.pad1_data[i], self.pad2_data[i]
        
    def __len__(self):
        return self.num_frames_usable

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def render_paddle(ax, pose, radius, color, rescale_factor):
    center = pose[:3] * rescale_factor
    quat = pose[3:]
    
    # Create a circle of points
    theta = np.linspace(0, 2*np.pi, 20)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    
    # Create a line from the center to the edge of the paddle
    line1_x = np.array([0, 2*radius])
    line1_y = np.zeros(2)
    line1_z = np.zeros(2)
    line2_x = np.zeros(2)
    line2_y = np.zeros(2)
    line2_z = np.array([0, 2*radius])
    
    # Combine points into a single array
    circle_points = np.column_stack((x, y, z))
    line1_points = np.column_stack((line1_x, line1_y, line1_z))
    line2_points = np.column_stack((line2_x, line2_y, line2_z))
    
    # Create rotation matrix from quaternion
    rotation_matrix = quaternion_to_rotation_matrix(quat)
    
    # Apply rotation to points
    rotated_circle = np.dot(circle_points, rotation_matrix.T)
    rotated_line1 = np.dot(line1_points, rotation_matrix.T)
    rotated_line2 = np.dot(line2_points, rotation_matrix.T)
    
    # Translate points to center position
    translated_circle = rotated_circle + center
    translated_line1 = rotated_line1 + center
    translated_line2 = rotated_line2 + center
    
    # Create the paddle circle
    paddle = ax.plot(translated_circle[:, 0], translated_circle[:, 1], translated_circle[:, 2], color=color, alpha=0.7)[0]
    
    # Create the orientation line
    orientation_line1 = ax.plot(translated_line1[:, 0], translated_line1[:, 1], translated_line1[:, 2], color='black', linewidth=1)[0]
    orientation_line2 = ax.plot(translated_line2[:, 0], translated_line2[:, 1], translated_line2[:, 2], color='green', linewidth=1)[0]
    
    # Fill the paddle
    verts = list(map(tuple, translated_circle))
    paddle_fill = art3d.Poly3DCollection([verts], alpha=0.3)
    paddle_fill.set_color(color)
    ax.add_collection3d(paddle_fill)
    
    return paddle, paddle_fill, orientation_line1, orientation_line2

def render_table(ax, table_dims):
    table_x = [-table_dims[0] / 2, table_dims[0] / 2, table_dims[0] / 2, -table_dims[0] / 2, -table_dims[0] / 2]
    table_y = [-table_dims[1] / 2, -table_dims[1] / 2, table_dims[1] / 2, table_dims[1] / 2, -table_dims[1] / 2]
    table_z = [+table_dims[2]] * 5

    # Draw the table outline
    ax.plot(table_x, table_y, table_z, color='b')

    # Add a colored surface for the table top
    X, Y = np.meshgrid([-table_dims[0] / 2, table_dims[0] / 2], [-table_dims[1] / 2, table_dims[1] / 2])
    Z = np.full_like(X, table_dims[2])
    ax.plot_surface(X, Y, Z, color='darkblue', alpha=0.5)

    # Add table legs
    leg_positions = [
        (-table_dims[0] / 2, -table_dims[1] / 2),
        ( table_dims[0] / 2, -table_dims[1] / 2),
        ( table_dims[0] / 2,  table_dims[1] / 2),
        (-table_dims[0] / 2,  table_dims[1] / 2),
    ]

    for x, y in leg_positions:
        ax.plot([x, x], [y, y], [table_dims[2], 0], color='brown', linewidth=2)

def render_player(ax, p_keypoints, scene_idx):
    p_keypoints = p_keypoints * RESCALE_FACTOR
    ax.scatter(p_keypoints[:, 0], p_keypoints[:, 1], p_keypoints[:, 2], s=0.5, color="black" if scene_idx == 0 else "blue")

def render_ball(ax, ball_trajectory, scene_idx):
    head_ball_pos_scaled = ball_trajectory[-1] * RESCALE_FACTOR # the most recent value. the head (or rosh, as r. eva would say) of the trajectory
    ax.scatter(head_ball_pos_scaled[0], head_ball_pos_scaled[1], head_ball_pos_scaled[2], s=4, color="black" if scene_idx == 0 else "green")

    ball_trajectory_arr = np.array(ball_trajectory) * RESCALE_FACTOR
    ax.plot(ball_trajectory_arr[:, 0], ball_trajectory_arr[:, 1], ball_trajectory_arr[:, 2], color="black" if scene_idx == 0 else "red")

def render_box(ax, box_x, box_y, box_z, **illustration_kwargs):
    # Create the vertices for the box
    vertices = [
        [box_x[0], box_y[0], box_z[0]],
        [box_x[1], box_y[0], box_z[0]],
        [box_x[1], box_y[1], box_z[0]],
        [box_x[0], box_y[1], box_z[0]],
        [box_x[0], box_y[0], box_z[1]],
        [box_x[1], box_y[0], box_z[1]],
        [box_x[1], box_y[1], box_z[1]],
        [box_x[0], box_y[1], box_z[1]],
    ]

    # Define the faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side faces
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[3], vertices[0], vertices[4], vertices[7]],
    ]

    # Create the 3D polygon collection with darker edges
    box = art3d.Poly3DCollection(
        faces, **illustration_kwargs,
    )
    ax.add_collection3d(box)


def render(processed_videos, fps, paddle_radius=0.08, show_feet=False, show_extended=False, preds_start_index=None, mean_ball_pos=None, std_ball_pos=None):
    """
    i scene index
    j frame index
    """

    # Define the vertices of the table
    table_dims = [900 * RESCALE_FACTOR, 500 * RESCALE_FACTOR, 250 * RESCALE_FACTOR]
    bounds = 3

    # 85% coverage
    conformal_quantiles = np.array([
        [1.4413868749807062, 1.443891338487522, 1.4259729262843222, 1.4100970127319434, 1.3828740395916277, 1.707768393437788, 1.9745400133270832, 2.0078195309541114, 1.9083580096684092, 1.885500704390635, 1.8187640274543608, 1.7579935699143623, 1.7173908813365362, 1.7009101082154314, 1.6715816156605778, 1.61290752465446, 1.5194540362738713, 1.4291331107827643, 1.3674529670318802, 1.3274110607994705, 1.2862556207149027, 1.299595168469526, 1.2744079245414146, 1.281233416737543, 1.3308655676133445, 1.433287918193609, 1.5428261160740573, 1.6749918508298711, 1.7028837893065456, 1.8843216474370121, 2.322996183873063, 1.8486106968479856, 2.37990055172976, 1.836100258244134], 
        [0.7576080095591078, 0.8639516596483199, 0.9652562573597373, 1.0580350256311817, 1.1762157739739192, 1.336021632717487, 1.149952523596564, 1.2028722433040984, 1.315343109345028, 1.4077472281111212, 1.4752008390251188, 1.5188088051462412, 1.509195847671136, 1.496212899169868, 1.4317042592002547, 1.3549013708401423, 1.27512183735148, 1.226043374048814, 1.155042138199697, 1.1252106986841575, 1.0883252876750298, 1.08669096540102, 1.0768037666233718, 1.0846593069221806, 1.1010926821501654, 1.1420852566057051, 1.1138349761068183, 1.1868479912745065, 1.2847617539685012, 1.324059018581373, 1.1614976555063585, 1.2586661080100536, 1.1629843931677748, 1.0391272712086883], 
        [0.40550781640637296, 0.5742079440870748, 0.6812971397118697, 0.7661444710282662, 0.8415379903185687, 0.9402827848608202, 0.7080130071480693, 0.6057326442700078, 0.5746672670870012, 0.5615233440799146, 0.5710602636760211, 0.561112864746084, 0.5312603983613653, 0.5117987256840801, 0.5157448388330336, 0.5150762613446283, 0.5113235767753378, 0.48560122547970874, 0.4512521915857473, 0.4488816650213105, 0.4557234478722744, 0.45870182822009503, 0.5162635518891237, 0.565577248774426, 0.6239351552541345, 0.683423214824455, 0.7606941273692029, 0.8514389244445777, 0.8902068848783098, 0.9571187223630979, 1.0459959294356345, 0.918422908633598, 0.9791553810014163, 0.9666145126566881] 
    ])

    num_scenes = len(processed_videos)
    min_frame, max_frame = 0, processed_videos[0].num_frames if show_extended else len(processed_videos[0])
    scenes = [ { j: processed_videos[i][j] for j in range(max_frame) } for i in range(num_scenes) ]
    ball_trajectories = [[] for _ in range(num_scenes)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()  # Clear the current axes

        # for each of the scenes that we're overlaying on top of each other
        for i in range(num_scenes):
            scene = scenes[i]
            if not frame in scene:
                continue

            p1_keypoints, p2_keypoints, ball_pos, pad1_pose, pad2_pose = scene[frame]

            # Plot the ball.
            ball_trajectory = ball_trajectories[i]   
            if frame == 0:
                ball_trajectory.clear()
            if ball_pos is not None:
                ball_trajectory.append(ball_pos)
                # Keep only the last 5 positions
                if len(ball_trajectory) > 5:
                    ball_trajectory.pop(0)

                render_ball(ax, ball_trajectory, i)

            # Plot the players.
            render_player(ax, p1_keypoints, i)
            render_player(ax, p2_keypoints, i)

            # Render paddles (optional, if data is available)
            # render_paddle(ax, pad1_pose, paddle_radius, "red", RESCALE_FACTOR)
            # render_paddle(ax, pad2_pose, paddle_radius, "red", RESCALE_FACTOR)

            # # Plot the reachable box as it grows over time
            # if i == 0 and preds_start_index and frame >= preds_start_index:
            #     t = (frame - preds_start_index) / fps
            #     box_x, box_y, box_z = generate_reachable_box(t)
            #     render_box(ax, box_x, box_y, box_z, alpha=0.1, facecolor='red', edgecolor='darkred', linewidths=1.5)
                
            # # Plot the conformal box
            # if i == 0 and preds_start_index and frame >= preds_start_index and frame <= 25 + preds_start_index:
            #     t = round((frame - preds_start_index) * 30 / fps)
            #     r = conformal_quantiles[:, t]
            #     r = r * ((std_ball_pos[frame] * RESCALE_FACTOR) + 0.016 * (t + 1))
            #     mean_ball_pos_scaled = mean_ball_pos[frame] * RESCALE_FACTOR
            #     box_x = [mean_ball_pos_scaled[0] - r[0], mean_ball_pos_scaled[0] + r[0]]
            #     box_y = [mean_ball_pos_scaled[1] - r[1], mean_ball_pos_scaled[1] + r[1]]
            #     box_z = [mean_ball_pos_scaled[2] - r[2], mean_ball_pos_scaled[2] + r[2]]
            #     render_box(ax, box_x, box_y, box_z, alpha=0.2, facecolor='cyan', edgecolor='darkblue', linewidths=1.5)

        render_table(ax, table_dims)

        ax.set_title('Sampled "game states"')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')

        ax.set_xlim(-bounds, bounds)
        ax.set_ylim(-bounds, bounds)
        ax.set_zlim(0, bounds)

        return ax,

    ani = FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=2.25*1000/fps, repeat_delay=2000, repeat=True)  # interval in milliseconds
    ani.save("rec.gif", writer='pillow')  # Saving is optional now
    plt.show()
