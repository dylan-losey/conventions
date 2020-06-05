import matplotlib.pyplot as plt
import numpy as np

curriculum_fail =
cf = np.asarray(curriculum_fail)
cf_mean = np.mean(cf,axis=0)
cf_std = np.std(cf,axis=0)
cf_sem = cf_std / np.sqrt(30)

curriculum_success = [[1, 1, 2, 0, 0], [0, 0, 1, 3, 2], [1, 4, 4, 2, 3], [0, 1, 2, 4, 3], [0, 2, 4, 3, 1], [1, 2, 0, 0, 1], [1, 4, 1, 3, 3], [0, 1, 0, 1, 1], [0, 3, 4, 3, 0], [0, 1, 3, 2, 2], [0, 1, 2, 0, 4], [0, 0, 1, 4, 0], [1, 0, 0, 1, 0], [0, 1, 3, 0, 3], [0, 0, 0, 4, 4], [1, 0, 0, 3, 2], [1, 2, 2, 1, 2], [0, 0, 2, 1, 1], [0, 1, 1, 0, 3], [1, 1, 4, 2, 4], [1, 1, 2, 4, 2], [2, 0, 1, 4, 0], [2, 1, 2, 4, 2], [0, 2, 4, 1, 4], [0, 1, 1, 4, 3], [0, 1, 1, 2, 3], [1, 1, 1, 4, 0], [1, 4, 3, 1, 2], [0, 2, 0, 4, 1], [0, 1, 1, 3, 3]]
cs = np.asarray(curriculum_success)
cs_mean = np.mean(cs,axis=0)
cs_std = np.std(cs,axis=0)
cs_sem = cs_std / np.sqrt(30)

x = [0, 1, 2, 3, 4]
plt.plot(x, cf_mean, 'w-')
plt.fill_between(x, cf_mean-cf_sem, cf_mean+cf_sem)
plt.plot(x, cs_mean, 'k-')
plt.fill_between(x, cs_mean-cs_sem, cs_mean+cs_sem)
plt.xticks([0, 1, 2, 3, 4])
plt.yticks([0, 1, 2, 3, 4], ['max SA', 'mdp1', 'mdp2', 'mdp3', 'min SA'])
plt.xlabel("Iteration #")
plt.title("Greedy Curriculum")

plt.legend(('','Failure','','Success'))
plt.show()
