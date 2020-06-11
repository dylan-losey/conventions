import matplotlib.pyplot as plt
import numpy as np
import pickle


def process(scores, window):
    processed_scores = []
    for item in scores:
        processed_scores.append(item[0])
    scores_np = np.asarray(processed_scores) / 500
    smoothed_scores = []
    for idx in range(window, len(scores_np) - window):
        score = np.mean(scores_np[idx-window:idx+window])
        smoothed_scores.append(score)
    return smoothed_scores


dqn_scores = pickle.load( open( "results/dqn.pkl", "rb" ) )
assist_scores = pickle.load( open( "results/assist.pkl", "rb" ) )
full_scores = pickle.load( open( "results/assist_full.pkl", "rb" ) )


window = 100
mean_dqn = process(dqn_scores, window)
mean_assist = process(assist_scores, window)
mean_full = process(full_scores, window)

x = range(len(mean_dqn))
plt.plot(x, mean_dqn, 'b')
plt.plot(x, mean_assist, 'orange')
plt.plot(x, mean_full, 'green')
plt.ylabel("Human Score")
plt.xlabel("Episode")
plt.legend(['without-sa', 'with-sa-partial', 'with-sa-full'])
plt.title('LunarLander-v2')
plt.savefig('results/rl.png')
plt.show()
