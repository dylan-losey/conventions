- developed environment
- trained expert on environment
- want to ensure bc is sufficient:
  - bc expert
  - added sampling noise to break tie
  - trying different bc model types
  - found that one hidden layer with tanh is enough
- get target and training set of MDPs
- init with random, try loop with

if the shared autonomy is too helpful, bad (bc human never gets experience)
if shared autonomy not helpful enough, bad (bc human ends up in bad zones)

take trajectories where human gets it right (or wrong)
then add the correct corresponding actions to the dataset

for success, only get trajectories with sa (at start)
for failure, only get trajectories without sa

with corrected trajectories, we are seeing a progression from easy to hard
clean up code
refine learning rates so that improves
there seems to be a lot of hyperparameter tuning in the bc algorithm

too glitchy... try a simpler version of the human policy that improves?
also the intuition is not clear. why should we go from lots of help to no help?
easy to see why start with lots of help --- adds more to dataset
but
even when human learns from failure, better to have representative failures
than garbage failures.


FAIL: [[2, 1], [3, 0, 4, 3], [2, 3, 3], [4, 1, 3, 1, 0], [2, 2, 3, 3], [2, 2], [2, 3, 3, 0], [2, 2, 1, 2, 4], [2, 1, 0, 4, 3], [2, 3]]
SUCCESS: [[0, 4, 0], [1, 3, 2], [0, 1, 3], [1, 1, 1], [0, 0, 2], [1, 1, 3], [0, 4, 3], [0, 1], [1, 0, 3, 3, 2, 4], [0, 1]]
