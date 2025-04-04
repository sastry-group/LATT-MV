import numpy as np
import math
from scipy.spatial.transform import Rotation as R

total_dist = 0.
errors = []
q_ours = np.array([0.70682518, 0., 0.70738828, 0.])
px = 0
py = 0
pz = 1.361
for i in range(10000) :
    
    x = np.random.uniform(-0.25,0.25)
    y = np.random.uniform(-0.55,0.55)
    z = np.random.uniform(0.65,0.95)
    xr,yr,zr = np.random.uniform(-1,1,3)
    # print(xr,yr,zr)
    # yr, zr = 0, 0
    # xr = 0
    z_axis = np.array([1.,yr,zr])
    z_axis = z_axis/np.linalg.norm(z_axis)
    # print(z_axis)
    theta_z = np.arctan2(z,y)+xr*0.5
    x_axis = np.array([0.,np.cos(theta_z),np.sin(theta_z)])
    x_axis[0] = (-x_axis[1]*z_axis[1] - x_axis[2]*z_axis[2])/z_axis[0]
    x_axis = x_axis/np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis,x_axis)
    Rot = np.array([x_axis,y_axis,z_axis]).T
    # print(Rot)
    r = R.from_matrix(Rot)
    q = r.as_quat()
    
    # calculate difference between q and q_ours
    r_current = R.from_quat(q)
    r_target = R.from_quat(q_ours)
    diff_quat = r_target*r_current.inv()
    # diff_quat = diff_quat.as_quat()
    error = diff_quat.magnitude()
    dist = math.sqrt((x-px)**2+(y-py)**2+(z-pz)**2)
    total_dist += 10*error + 10*dist
    errors.append(error)
    q_ours = q.copy()
    px = x
    py = y
    pz = z

# print(np.max(errors))

print(total_dist/10000)
# print(np.pi*3/2.)
# print(total_dist/1000+(np.pi*3/2.))