import numpy as np
import matplotlib.pyplot as plt
import pickle

FILE = "pre_gt_init.pkl"
with open(FILE, "rb") as f:
    data = pickle.load(f)

target_poses, poses_achieved, where_landed = data

where_landed_filtered = []
errors = []
errors_x = []
errors_y = []
for wl in where_landed:
    if str(wl) != 'None':
        where_landed_filtered.append(wl)
        errors.append(np.linalg.norm(wl[:2] - np.array([2., 0.])))
        errors_x.append(wl[0] - 2.)
        errors_y.append(wl[1])

where_landed_filtered = np.array(where_landed_filtered)
# print(where_landed_filtered)
where_landed_filtered[:,0] -= 1.5
plt.figure()
plt.scatter(where_landed_filtered[:,0], where_landed_filtered[:,1],color='r',label='Where the ball landed',marker='x')
plt.scatter([0.75],[0.],label='target location',color='g')

# Draw rectangle with corners at (0, -0.76) and (1.37, 0.76)
rect_x = [0, 1.4, 1.4, 0, 0]
rect_y = [-0.76, -0.76, 0.76, 0.76, -0.76]
plt.plot(rect_x, rect_y, 'b-', label="Boundary Rectangle")
plt.plot([0,0],[-0.9,0.9],'--',label='Net',color='purple')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Where the ball landed")
plt.axis('equal')
plt.savefig("where_landed_"+FILE[:-4]+".png")
plt.show()

# Draw a histogram of the errors
plt.figure()
plt.hist(errors, bins=20)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Histogram of errors")
plt.savefig("histogram_"+FILE[:-4]+".png")
plt.show()


# Draw a histogram of the errors for x and y
plt.figure()
plt.hist(errors_x, bins=20)
plt.xlabel("Error in x")
plt.ylabel("Frequency")
plt.title("Histogram of errors in x")
plt.savefig("histogram_x_"+FILE[:-4]+".png")
plt.show()

plt.figure()
plt.hist(errors_y, bins=20)
plt.xlabel("Error in y")
plt.ylabel("Frequency")
plt.title("Histogram of errors in y")
plt.savefig("histogram_y_"+FILE[:-4]+".png")
plt.show()

with open(FILE[:-4]+'.txt', "a") as f:
    f.write("Mean error: "+str(np.mean(errors))+'\n')
    f.write("Mean error in x: "+str(np.mean(np.abs(errors_x)))+'\n')
    f.write("Mean error in y: "+str(np.mean(np.abs(errors_y)))+'\n')
    f.write("Median error: "+str(np.median(errors))+'\n')
