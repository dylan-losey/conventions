# conventions

Inspired by our recent work on assistive robots, we want to better understand and formulate how conventions are formed during human-robot interaction.

## what's a convention?

Let _s_ be the robot state and let _a_ be the robot action. The human provides input _z_, and the **convention**

<img src="https://latex.codecogs.com/svg.latex?a&space;=&space;\phi(s,&space;z)" title="a = \phi(s, z)" /> 

maps the human's input to a robot action.

## environment

### linear dynamical system

Let's focus on a robot with dynamics:
$$\dot{s} = As + Ba$$
where $$A$$ and $$B$$ are constant matrices that capture the physical properties of the system (e.g., mass, damping, etc.). The robot takes action $$a$$ and observes state $$s$$.

Here we will define the *convention* as a linear funtion of the state and input:
$$a = \phi(s, a) = Fs + Gz$$
where $$F$$ and $$G$$ are constant matrices that capture the convention.

Putting together these equations, we obtain the overall robot dynamics:
$$\dot{s} = (A + BF)s + (BG)z$$
Intuitively, the convention changes the robot's dynamics to make it easier for the human to control.

### quadratic cost function

The human has in mind a cost function they want the robot to minimize. Let's assume that this cost function combines error and effort:
$$J = \int \|s^* - s(t)\|^2_Q + \|z(t) \|^2_R ~dt$$
The human knows the right task $$s^*$$ and the correct weights $$Q$$ and $$R$$.

### feedback

At the end of each task, we assume that the human reports their total cost $$J$$ to the robot.





