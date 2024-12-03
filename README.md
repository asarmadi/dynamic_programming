# Dynamic Programming
In this repository, dynamic programming is implemented to solve various problems.

## Install
A python virtual environment could be generated

`
python -m venv env
`

and activates in windows

`
.\env\Scripts\activate
`

in Linux

`
source ./env/bin/activate
`

## Pit of Death Example
There is a grid of squares and the agent should avoid the obstacle to reach to the goal. In the following figure, the outcome of the algorithm could be seen:
![plot](./Figs/29.png)

## Double Integrator
There is a mass of 1 Kg and the goal is to move the mass towards the origin at minimum time. The dynamics of the system is 

$$
\ddot{x}=u
$$

![plot](./Figs/di_100.png)