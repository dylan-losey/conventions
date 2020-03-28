# Conventions

Inspired by our recent work on assistive robots, we want to better understand and formulate how conventions are formed during human-robot interaction.

## What's A Convention?

Let s be the robot state and let a be the robot action. The human provides input z, and the **convention**:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?a&space;=&space;\phi(s,&space;z)" title="a = \phi(s, z)" />
</p>
maps the human's input to a robot action.

## Environment

### linear dynamical system

Let's focus on a robot with dynamics:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dot{s}&space;=&space;As&space;&plus;&space;Ba" title="\dot{s} = As + Ba" />
</p>
where A and B are constant matrices that capture the physical properties of the system (e.g., mass, damping, etc.). The robot takes action a and observes state s.


Here we will define the **convention** as a linear funtion of the state and input:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?a&space;=&space;\phi(s,&space;a)&space;=&space;Fs&space;&plus;&space;Gz" title="a = \phi(s, a) = Fs + Gz" />
</p>
where F and G are constant matrices that capture the convention.


Putting together these equations, we obtain the overall robot dynamics:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\dot{s}=(A&plus;BF)s&plus;(BG)z" title="\dot{s}=(A+BF)s+(BG)z" />
</p>
Intuitively, the convention changes the robot's dynamics to make it easier for the human to control.

### quadratic cost function

The human has in mind a cost function they want the robot to minimize. Let's assume that this cost function combines error and effort:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?J=\int{\|s^*-s(t)\|^2_Q&plus;\|z(t)\|^2_R~dt}" title="J=\int{\|s^*-s(t)\|^2_Q+\|z(t)\|^2_R~dt}" />
</p>
The human knows the right task s* and the correct weights Q and R.

### feedback

At the end of each task, we assume that the human reports their total cost J to the robot.
We also assume that the human can directly observe the convention that the robot is using.
In other words, the human knows both F and G.

## Experimental Setup

### human models

We simulate two types of humans: an **optimal** human and a **procedural** human.
Both humans provide feedback as a linear function of the current state:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?z=K(s^*-s)" title="z=K(s^*-s)" />
</p>


The optimal human solves the LQR problem to select their feedback matrix K.
Importantly, their strategy changes as a function of the current convention.


The procedural human has a fixed feedback matrix K.
This feedback is predefined, and does not change based on the convention.

### first-order robot

The robot is a 1-DoF mass-damper.
In this setting F = [f<sub>1</sub>, f<sub>2</sub>] has two parameters, and G is a scalar.
Without loss of generality, we will fix G = +1.
