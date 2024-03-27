Note that the description below is not complete

# Stochastic Calculus
In this project, a set of classes is made that is able to solve stochastic differential equations numerically. 

1) Generate stochastic processes
2) Implement stopping times
3) Use the processes from (1) as integrators for stochastic integration, for example
$$\int_0^tf(W_t,t)dW_t$$
4) Generalise (3) to solve stochastic differential equations for example
$$dX_t=f(W_t,t)dt + g(W_t,t)dW_t$$
5) Generalise (4) to solve systems of stochastic differential equations

Most numerical solutions of the equations are checked and coincide with the solutions from Itô calculus.
Multiple different processes are implemented, from Poisson processes to semi-martingales like Brownian Motion jump processes.
However, the systems (Lotka-Volterra and the Heston model) of SDE's have not been verified analytically.

In sp_notebook the project is seen in action. First, a number of stochastic processes are simulated and displayed.

In the notebook the integral $\int_0^t\sin(t)dW_t$ is solved. The solution looks like
![stochint sin dwt](https://github.com/Sgouwens/stochasticcalc/assets/150426079/4fb3e46f-fcae-4b0d-880d-2685c0b6cb39)

And a simplified version of the black scholes is solved:

![black scholes simplified](https://github.com/Sgouwens/stochasticcalc/assets/150426079/a421aa5a-acef-40ca-9193-564a368f40eb)

