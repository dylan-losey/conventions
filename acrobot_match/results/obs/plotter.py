import matplotlib.pyplot as plt
import numpy as np
import pickle


def process(scores):
    scores = np.array(scores) / 200.0
    mean = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)
    return mean, std / np.sqrt(len(scores))


none_full = pickle.load( open( "NoneSA-Full.pkl", "rb" ) )
none_partial = pickle.load( open( "NoneSA-Partial.pkl", "rb" ) )

all_full = pickle.load( open( "AllSA-Full.pkl", "rb" ) )
all_partial = pickle.load( open( "AllSA-Partial.pkl", "rb" ) )

some_full = pickle.load( open( "SomeSA-Full.pkl", "rb" ) )
some_partial = pickle.load( open( "SomeSA-Partial.pkl", "rb" ) )

mnf, _ = process(none_full)
mnp, _ = process(none_partial)
maf, _ = process(all_full)
map, _ = process(all_partial)
saf, _ = process(some_full)
sap, _ = process(some_partial)


x = range(len(mnf))
plt.plot(x, mnf, 'blue')
plt.plot(x, maf, 'gray')
plt.plot(x, saf, 'orange')
plt.ylabel("Human Score")
plt.xlabel("Episode")
plt.legend(['without-sa','with-sa-always','with-sa-adapt'])
plt.title('Acrobot-v1: Full Obs.')
plt.savefig('il_full.png')
plt.show()

plt.plot(x, mnp, 'blue')
plt.plot(x, map, 'gray')
plt.plot(x, sap, 'orange')
plt.ylabel("Human Score")
plt.xlabel("Episode")
plt.legend(['without-sa','with-sa-always','with-sa-adapt'])
plt.title('Acrobot-v1: Partial Obs.')
plt.savefig('il_partial.png')
plt.show()
