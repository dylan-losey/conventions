import matplotlib.pyplot as plt
import numpy as np
import pickle


def process(scores):
    scores = np.array(scores) / 200.0
    mean = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)
    return mean, std / np.sqrt(len(scores))


none = pickle.load( open( "NoneSA.pkl", "rb" ) )
all = pickle.load( open( "AllSA.pkl", "rb" ) )
some = pickle.load( open( "SomeSA.pkl", "rb" ) )

nonem, _ = process(none)
allm, _ = process(all)
somem, _ = process(some)


x = range(len(nonem))
plt.plot(x, nonem, 'blue')
plt.plot(x, allm, 'gray')
plt.plot(x, somem, 'orange')
plt.ylabel("Human Score")
plt.xlabel("Episode")
plt.legend(['without-sa','with-sa-always','with-sa-adapt'])
plt.title('Acrobot-v1: Success')
plt.savefig('il.png')
plt.show()
