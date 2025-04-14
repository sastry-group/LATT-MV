import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.labelsize"] = 14    # x and y labels
plt.rcParams["legend.fontsize"] = 18   # legend size
plt.xticks(fontsize=13)  # X-axis tick font size
plt.yticks(fontsize=13)  # Y-axis tick font size
N = 666
losses = np.loadtxt('eval_losses.csv',delimiter=',')

alpha = 0.2
curr_loss = losses[0,2]
for i in range(1,len(losses)):
    curr_loss = alpha*losses[i,2] + (1-alpha)*curr_loss
    losses[i,2] = curr_loss
plt.plot(losses[:N,1],losses[:N,2],label='eval rewards')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.savefig('rewards_eval.png')
# plt.show()

plt.figure()
N = 699
losses = np.loadtxt('train_losses.csv',delimiter=',')

alpha = 0.2
curr_loss = losses[0,2]
for i in range(1,len(losses)):
    curr_loss = alpha*losses[i,2] + (1-alpha)*curr_loss
    losses[i,2] = curr_loss
plt.plot(losses[:N,1],losses[:N,2],label='training rewards')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.savefig('rewards_train.png')
# plt.show()

