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
Hence, in our experiments the convention boiled down to F = [f<sub>1</sub>, f<sub>2</sub>], where the robot chose these two parameters.

# Results

## Convexity

We want to identify the **optimal convention** F* that will minimize the human's cost.
But as the robot changes its convention to adapt to the human, the human also adapts to the robot.
In the worse case scenario, this mutual adaptation may lead to a local minima:
for instance, the human may never be able to convey their desired actions because they are mapped to expensive inputs.


Here we ask whether the process of finding the optimal convention is **convex**.
The corresponding code is **convex_test.py**.

<p align="center">
<img src=results/convex_test_opt.png>
</p>

Above we plot **the cost J** as a function of **the convention F**.
This plot was generated with a **optimal human** that has in mind a fixed task, where they want to reach s* = [1, 0].
The black dot corresponds to the global minima over the tested set of F.
This plot suggests that J is indeed convex in F: the robot can adapt its convention without worrying about local solutions.

## Task Dependency

The robot uses conventions to change its dynamics so that tasks are easier for the human.
It is not yet clear whether there is a single convention that the robot should always use,
or whether the **optimal convention depends on the current task(s)**.

Here we **vary the task**, and then learn the best possible convention for each task.
The corresponding code is **task_test.py**.

<img src=results/conventions_vs_task_opt.png width="49%"/> <img src=results/conventions_vs_task_pro.png width="49%"/>

Above we plot **the optimal convention F** as a function of **the task**.
The left plot corresponds to an **optimal human**, and the right plot corresponds to a **procedural human**.
Across all of the tasks, the human had the same goal position --- we varied the goal velocity that the human wanted.
From these plots, it is clear that the optimal convention is task dependent.


A natural follow-up question is whether these optimal conventions are actually useful.
We want to confirm that the optimal convention **makes the task easier for the human**.
Using the conventions learned above, we compared the **cost with an initial convention** to the **cost with our optimal convention**.

<img src=results/cost_improvement_opt.png width="49%"/> <img src=results/cost_improvement_pro.png width="49%"/>

The left plot corresponds to an **optimal human**, and the right plot corresponds to a **procedural human**.
From these plots, we confirm that optimal, task-dependent conventions make the human's task easier.

## Task Distributions

We've seen that the robot can adapt its convention based on the human and their intended task.
But what if the human wants to perform more than one task?
Can we identify a convention that makes a **distribution of tasks** easier for the human to perform?


Here we sample from a task distribution.
The goal position is sampled from U[-1, 1], and the goal velocity is sampled U[1, 2].
We then identify the **optimal convention** across ~10 sampled tasks.
The corresponding code is **distribution_test.py**.

<img src=results/cost_distribution_opt.png width="49%"/> <img src=results/cost_distribution_pro.png width="49%"/>

Above we plot the **average task cost** using the initial convention and the optimal convention.
The left plot corresponds to an **optimal human**, and the right plot corresponds to a **procedural human**.
From these plots, we see that we can identify conventions that not only make a single task easier, but also make distributions of tasks easier to complete.

## Consistent Humans

The above tests were performed with both an **optimal human** and a **procedural human**.
We've found that robots can learn effective conventions for each type of human.
Put another way, the human does not need to try to optimize their response to the robot's convention:
so long as the human's response is **consistent**, the robot can adapt and improve.


Let's define consistency. <i>Given a fixed convention F and task s*, the human reports cost J ~ N(J*, sigma).
The smaller sigma is, the more consistent the human is.</i>
Both our optimal and procedural humans were completely consistent, since they had a one-to-one mapping from F and s* to J.

# Next Steps

 - Insight: conventions define the relationship between human inputs and robot outputs
 - Look at a very different environment, such as correctly identifying a photo from its description
 - Conventions that evolve in scale, such as pressing up indicating open the door

## Human with Latent State

We started looking at a human whose behavior is directly influenced by the robot.
Previously, we had the optimal human.
We found that this new human can cause the robot's learning to become unstable. 
