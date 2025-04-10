import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import math
from gymnasium import spaces
# from gym.utils import seeding
import gymnasium as gym
import mujoco_viewer
from scipy.spatial.transform import Rotation as R

TABLE_SHIFT = 1.7
# MuJoCo XML definition with Franka Panda robot and table tennis setup
xml = """
<mujoco model="table_tennis">
    <include file="iiwa14.xml"/>
    <compiler angle="radian" />
    <option timestep="0.0167" gravity="0 0 -9.81" />
    <worldbody>
        <!-- Ground -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="vis" pos="0 0 1.26" quat="0 0.7068252 0 0.7073883">
            <geom name="cylinder" type="cylinder" pos="0.1 0 0" size="0.10 0.015" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
            <geom name="handle" type="cylinder" pos="-0.05 0 0" size="0.02 0.05" quat="0 0.7068252 0 0.7073883" rgba="0 0 1 0.3" contype="0" conaffinity="0"/>
        </body>
        <!-- Table -->
        <body name="table" pos="1.7 0 0.56">
            <!-- Table surface -->
            <geom name="table_top" type="box" size="1.37 0.7625 0.02" rgba="0 0 1 1" friction="0.2 0.2 0.1" solref="0.04 0.1" solimp="0.9 0.999 0.001" />
        </body>

        <body name="net" pos="1.7 0 0.57" euler="0 0 0"> <!-- Position and rotate net -->
            <!-- Net surface -->
            <geom name="net_geom" type="box" size="0.01 0.7625 0.1" rgba="1 1 1 1" friction="0 0 0" solref="0.00001 1" solimp="0.99 0.9999 0.00001" />
        </body>
            
        <!-- Ball -->
        <body name="ball" pos="2 0 2">
            <freejoint name="haha"/>
            <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027" rgba="1 0.5 0 1" 
                  friction="0.001 0.001 0.001" solref="0.04 0.05" solimp="0.9 0.999 0.001" />
        </body>
    </worldbody>
</mujoco>
"""

class KukaTennisEnv(gym.Env):
    def __init__(self,proc_id=0,history=4):
        super(KukaTennisEnv, self).__init__()
        self.history = history  
        # Load the MuJoCo model
        self.model = mj.MjModel.from_xml_string(xml)  # Use your actual MuJoCo XML path
        self.data = mj.MjData(self.model)

        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(7,), dtype=np.float32)  # Adjust based on your actuator count
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 7 + 7 + 7 + 7*history,), dtype=np.float32)

        # Simulation time step
        self.sim_dt = self.model.opt.timestep

        self.viewer = None

        self.max_episode_steps = 100
        self.current_step = 0
        self.orientation_K = 10.0
        self.dist_k = 10.0
        self.prev_reward = 0.
        self.tolerance_range = [2.5,1.0]
        self.tolerance_exp = 12_000_000/256
        self.total_steps = 0
        self.proc_id = proc_id
        self.prev_actions = np.zeros((history,7))
        self.last_qvel = np.zeros(3)
        self.last_qpos = np.zeros(3)
        
    def set_target_pose(self,pose):
        # print("Req:", pose)
        self.curr_target = pose
        self.update_vis_pose(self.curr_target)

    def update_vis_pose(self,pose):
        # Update the cylinder geom position
        target_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'vis')
        # self.data.geom_xpos[target_geom_id] = np.array(pose)
        self.model.body_pos[target_geom_id] = pose[:3]
        self.model.body_quat[target_geom_id] = pose[[6,3,4,5]]


    def reset_ball_throw(self):
        initial_velocity = np.array([-4.5, 0., 2., 0., 0., 0.])
        # print(self.data.body('ball').cvel)
        self.data.qvel[-6:] = initial_velocity
        self.data.qpos[-7:-4] = np.array([1.8+TABLE_SHIFT,np.random.uniform(-0.4,0.4),0.9])

    def step(self, action):
        self.prev_actions[:-1,:] = self.prev_actions[1:,:]
        self.prev_actions[-1,:] = action
        self.current_step += 1
        self.total_steps += 1
        # Apply action to actuators
        # self.data.ctrl[:] = np.array(action) + np.array(self.data.qpos[:7])
        self.data.ctrl[:] = np.array(action) + np.array(self.data.qpos[:7])
        # print(self.data.ctrl[:])
        # print(np.array(self.data.qvel[-6:-3]))

        # Check for collisions
        ncon = self.data.ncon  # Number of contacts
        if ncon > 0:
            for i in range(len(self.data.contact)):
                # print(i,len(self.data.contact))
                contact = self.data.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2
                # geom1 = self.model.geom_id2name(geom1)
                # geom2 = self.model.geom_id2name(geom2)
                # Retrieve the names of geom1 and geom2
                geom1 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom1)
                geom2 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom2)
                if geom2 is None or geom1 is None:
                    continue
                # print(f"Collision detected between geom {geom1} and geom {geom2}")

                if geom1 == "ball_geom" or geom2 == "ball_geom":
                    # print("Ball hit")
                    ball_vel = np.array(self.last_qvel)
                    # print(ball_vel)
                    ball_vel[2] = -ball_vel[2]
                    self.data.qpos[-7:-4] = self.last_qpos + 0.016*ball_vel
                    self.data.qvel[-6:-3] = ball_vel
                    # mj.mj_step(self.model, self.data)
                    # self.data.qvel[-6:-3] = ball_vel
                    # print(self.data.qvel[-6:])
                if (geom1 == "racket" and geom2=="ball_geom") or (geom2 == "racket" and geom1=="ball_geom"):
                    print("Racket hit the ball")
                    # Reflect ball perpendicular to racket
                    ball_vel = np.array(self.last_qvel)
                    racket_rot = R.from_quat(self.data.body('tennis_racket').xquat[[1,2,3,0]]).as_matrix()
                    # print(racket_rot)
                    racket_normal = racket_rot[:,2]
                    ball_vel = ball_vel - 2*np.dot(ball_vel,racket_normal)*racket_normal
                    self.data.qpos[-7:-4] = self.last_qpos + 0.016*ball_vel
                    self.data.qvel[-6:-3] = ball_vel
                    # mj.mj_step(self.model, self.data)
                    # self.data.qvel[-6:-3] = ball_vel
        self.last_qvel = np.array(self.data.qvel[-6:-3])
        self.last_qpos = np.array(self.data.qpos[-7:-4])
        mj.mj_step(self.model, self.data)
        self.calc_target_racket_pose()
        # print(np.linalg.norm(self.curr_target[3:7]), np.linalg.norm(self.data.body('tennis_racket').xquat))
        # print(self.curr_target[3:7],self.data.body('tennis_racket').xquat)
        end_effector_pos = self.data.body('tennis_racket').xpos
        # print(end_effector_pos)
        end_effector_quat = self.data.body('tennis_racket').xquat[[1,2,3,0]]
        # print(self.data.body('link7').xquat[[1,2,3,0]])
        # print(self.data.body('tennis_racket').xquat[[1,2,3,0]])
        # print(end_effector_quat,self.data.body('tennis_racket').xquat)
        diff_pos = self.curr_target[:3] - end_effector_pos

        r_current = R.from_quat(end_effector_quat)
        r_target = R.from_quat(self.curr_target[3:7])
        diff_quat = r_target*r_current.inv()
        diff_quat = diff_quat.as_quat()
        # Get observation (qpos: joint positions, qvel: joint velocities)
        obs = np.float32(np.concatenate([self.data.qpos[:7], self.data.qvel[:7],diff_pos,diff_quat,self.curr_target,self.prev_actions.flatten()]))
        # print(self.data.qpos)
        # Calculate reward and done
        reward = self._calculate_reward()
        curr_reward = reward - self.prev_reward - 150*np.sum(np.abs(self.prev_actions[-1,:]))/(7000*0.15)
        self.prev_reward = reward
        done = self._is_done()
        tol = self.tolerance_range[1] + (self.tolerance_range[0] - self.tolerance_range[1])*np.exp(-self.total_steps/self.tolerance_exp)
       
        if self.current_step >= self.max_episode_steps:
            self.current_step = 0
            done = True
        return obs, curr_reward, done, False, {}

    def reset_target(self):
        self.curr_target = np.array([0.,0.,0.,0.,0.,0.,0.])
        self.curr_target[0] = np.random.uniform(-0.2,0.2)
        self.curr_target[1] = np.random.uniform(-0.5,0.5)
        self.curr_target[2] = np.random.uniform(0.75,1.05)
        xr,yr,zr = np.random.uniform(-1,1,3)*0.5
        z_axis = np.array([1.,yr,zr])
        z_axis = z_axis/np.linalg.norm(z_axis)
        theta_z = np.arctan2(self.curr_target[2],self.curr_target[1])+xr*0.5
        x_axis = np.array([0.,np.cos(theta_z),np.sin(theta_z)])
        x_axis[0] = (-x_axis[1]*z_axis[1] - x_axis[2]*z_axis[2])/z_axis[0]
        x_axis = x_axis/np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis,x_axis)
        Rot = np.array([x_axis,y_axis,z_axis]).T
        r = R.from_matrix(Rot)
        q = r.as_quat()
        self.curr_target[3:7] = q
        self.update_vis_pose(self.curr_target)
        
    def calc_target_racket_pose(self,bounce_factor=1,table_z=0.56,g = -9.81,x_target=TABLE_SHIFT+1.37/2.,y_target=0.):
        # Get ball position and velocity
        ball_pos = self.data.body('ball').xpos
        ball_vel = np.zeros((3,1))
        ball_vel[:,0] = self.data.qvel[-6:-3]
        if ball_vel[0,0] > 0 or ball_pos[0] < 0.5 or ball_vel[2,0] > 15.:
            return
        print("vel:",ball_vel[:,0])
        x = 0.
        T = (x - ball_pos[0])/ball_vel[0,0]
        y = ball_pos[1] + ball_vel[1,0]*T
        
        z = ball_pos[2] + ball_vel[2,0]*T + 0.5*g*T*T - 0.1
        v_future = ball_vel + np.array([[0.],[0.],[g*T]])
        if z<table_z :
            vz_bounce = -np.sqrt(-2*g*max(ball_pos[2]-table_z,0)+ball_vel[2,0]*ball_vel[2,0])
            t_bounce = (vz_bounce - ball_vel[2,0])/g
            t_remaining = T - t_bounce
            v_future[2,0] = -vz_bounce + g*t_remaining
            z = table_z + 0.5*g*t_remaining*t_remaining - vz_bounce*t_remaining*bounce_factor -0.1
        pos = np.array([x,y,z])
        # print("Target:",pos)
        # Calculate racket orientation
        # All possible z_axis are of form [1,yr,zr]. Generate them with np.meshgrid on yr,zr from range -1,1
        yzr = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,0.1,100))
        yzr = np.array(yzr).reshape(2,-1).T
        xyzr = np.concatenate([np.ones((yzr.shape[0],1)),yzr],axis=1)
        xyzr = xyzr/np.linalg.norm(xyzr,axis=1)[:,None]
        # ball_vel[2,0] += g*T
        ball_vels = np.tile(v_future.T,(100*100,1)).T
        ball_reflected_vels = ball_vels - 2*np.sum(ball_vels*xyzr.T,axis=0,keepdims=True)*xyzr.T
        vz_hits = -np.sqrt(-2*g*max(pos[2]-table_z,0)+ball_reflected_vels[2]*ball_reflected_vels[2])
        t_hits = (vz_hits - ball_reflected_vels[2])/g
        x_hits = pos[0] + ball_reflected_vels[0]*t_hits
        y_hits = pos[1] + ball_reflected_vels[1]*t_hits
        costs = (x_hits-x_target)**2 + (y_hits-y_target)**2
        idx = np.argmin(costs)
        z_axis = xyzr[idx]
        # print("x:",x_hits[idx],"y:",y_hits[idx],"z:",z_axis,"t:",t_hits[idx])
        theta_z = np.arctan2(z,x)
        x_axis = np.array([0.,np.cos(theta_z),np.sin(theta_z)])
        x_axis[0] = (-x_axis[1]*z_axis[1] - x_axis[2]*z_axis[2])/z_axis[0]
        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis,x_axis)
        Rot = np.array([x_axis,y_axis,z_axis]).T
        r = R.from_matrix(Rot)
        q = r.as_quat()
        self.set_target_pose(np.concatenate([pos,q]))

    def reset(self,seed=None):
        self.current_step = 0
        self.prev_actions = np.zeros((self.history,7))
        prev_robot_pos = np.array(self.data.qpos[:7])
        mj.mj_resetData(self.model, self.data)
        self.reset_target()
        self.reset_ball_throw()
        self.data.qpos[:7] = prev_robot_pos
        # for i in range(7):
        #     self.data.qpos[i] = np.random.uniform(-1.,1.)*0.5
        # target_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, 'vis')
        # print(self.data.geom_xpos[target_geom_id])
        mj.mj_forward(self.model, self.data)
        # print(self.data.geom_xpos[target_geom_id])
        self.prev_reward = self._calculate_reward()
        end_effector_pos = self.data.body('tennis_racket').xpos
        # print(end_effector_pos)
        end_effector_quat = self.data.body('tennis_racket').xquat[[1,2,3,0]]
        diff_pos = self.curr_target[:3] - end_effector_pos
        r_current = R.from_quat(end_effector_quat)
        # print(r_current.as_matrix())
        r_target = R.from_quat(self.curr_target[3:7])
        diff_quat = r_target*r_current.inv()
        diff_quat = diff_quat.as_quat()
        
        # Return initial observation
        obs = np.float32(np.concatenate([self.data.qpos[:7], self.data.qvel[:7],diff_pos,diff_quat,self.curr_target,self.prev_actions.flatten()]))

        info = {}
        return obs, info

    def render(self, mode="human"):
        # return
        if not hasattr(self, 'viewer'):
            self.viewer = mj.MjViewer(self.model)
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _calculate_reward(self):
        racket_pos = self.data.body('tennis_racket').xpos
        # print(racket_pos)
        racket_orientation = self.data.body('tennis_racket').xquat[[1,2,3,0]]
        # print(racket_orientation)
        r_current = R.from_quat(racket_orientation)
        r_target = R.from_quat(self.curr_target[3:7])
        diff_quat_rel = r_target*r_current.inv()
        diff_quat = diff_quat_rel.as_quat()
        # print(diff_quat_rel)
        error = diff_quat_rel.magnitude()
        # print("angle error: ",error,2*np.arcsin(np.linalg.norm(diff_quat_rel.as_quat()[:3])))
        # Implement your reward calculation
        reward = - self.dist_k*np.linalg.norm(racket_pos - self.curr_target[:3]) - (self.orientation_K*error)
        
        # if np.linalg.norm(racket_pos - self.curr_target[:3]) < 0.2 and error < 0.2:
        #     reward += 250
        # if self.current_step < 1 or self.current_step >299 :
        #     print(self.current_step,reward,error,np.linalg.norm(racket_pos - self.curr_target[:3]))
            # print(racket_pos, racket_orientation, self.curr_target)
        return reward

    def get_expert_cmd(self) :
        jacp = np.zeros((3, self.model.nv))  # Jacobian for translational velocity (3D vector)
        jacr = np.zeros((3, self.model.nv))  # Jacobian for rotational velocity (3D vector)
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'tennis_racket')
        curr_pos = self.data.body('tennis_racket').xpos
        d_pos = np.array([0.75,0.,0.6]) - curr_pos
        # print(data.body('tennis_racket'),body_id)
        mj.mj_jac(self.model, self.data, jacp, jacr, self.data.body('tennis_racket').xpos, body_id)
        target_joint_vel = 0.5*np.dot(np.linalg.pinv(jacp),d_pos)
        target_joint_pos = np.array(self.data.qpos[:7]) + target_joint_vel[:7]
        return np.array(target_joint_pos)

    def _is_done(self):
        # Implement termination condition
        return False

# Initialize the GLFW window for rendering
def init_glfw():
    if not glfw.init():
        raise Exception("Unable to initialize GLFW")
    window = glfw.create_window(1280, 720, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Unable to create GLFW window")
    glfw.make_context_current(window)
    return window


if __name__ == "__main__":
    env = KukaTennisEnv()
    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()*0  # Random action
        obs, reward, done, _, _ = env.step(action)
        # print(i,reward)
        env.render()
    env.close()
    