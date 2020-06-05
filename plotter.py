import matplotlib.pyplot as plt
import numpy as np

curriculum_fail = [[2, 1], [3, 0, 4, 3], [2, 3, 3], [4, 1, 3, 1, 0], [2, 2, 3, 3], [2, 2], [2, 3, 3, 0], [2, 2, 1, 2, 4], [2, 1, 0, 4, 3], [2, 3]]
avg = [0]*5
for idx in range(5):
    count = 0
    for item in curriculum_fail:
        if len(item) > idx:
            count += 1.0
            avg[idx] += item[idx]
    avg[idx] /= count
plt.plot(avg)

curriculum_success = [[0, 4, 0], [1, 3, 2], [0, 1, 3], [1, 1, 1], [0, 0, 2], [1, 1, 3], [0, 4, 3], [0, 1], [1, 0, 3, 3, 2, 4], [0, 1]]
avg = [0]*5
for idx in range(5):
    count = 0
    for item in curriculum_success:
        if len(item) > idx:
            count += 1.0
            avg[idx] += item[idx]
    avg[idx] /= count
plt.plot(avg)

plt.ylabel("MDP (0 = max SA, 5 is no SA)")
plt.xlabel("Iteration")
plt.legend(('Fail', 'Success'))
plt.show()
