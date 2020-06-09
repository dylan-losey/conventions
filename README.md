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

Try with a RL agent for learning, and then changing the dataset avaliable
I think we can get the same two main findings --- too much assistance, sucks
but some assistance speeds up performance (other group shared this as well)
