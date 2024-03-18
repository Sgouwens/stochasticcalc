import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import poisson
from dataclasses import dataclass

class Plotprocess():
    """A class containing all the plotting actions."""

    @staticmethod
    def plot_mean(t, Xt):
        """Plots the mean of sample paths as a function of time"""
        Xt_mean = np.mean(Xt, axis=0)
        plt.plot(t, Xt_mean, color="black")

    @staticmethod
    def plot_var(t, Xt):
        
        """Plots the variance of sample paths as a function of time"""
        Xt_var = np.var(Xt, axis=0)
        plt.plot(t, Xt_var, color="black")

    @staticmethod
    def compute_quantiles(Mt, c=0.05):
        """Computes [a, 1-a] of sample paths as a function of time. By default, a is 0.025"""
        quantiles = np.quantile(Mt, q=[c/2, (1-c/2)], axis=0)
        return quantiles
    
    def plot_quantiles(self, t, Mt, c=0.05):
        """Plots [a, 1-a] of sample paths at each time. By default, a is 0.025"""
        quantiles = self.compute_quantiles(Mt, c=c)
        plt.fill_between(t, y1=quantiles[0,:], y2=quantiles[1,:], alpha=0.1)

    def plot_solution(self, t, Xt, num_max=20, alpha=0.4, with_mean=False, with_var=False, with_quantiles=False, color=None, show=True):
        """Function that plots (a subset of) the sample paths of a stochastic. 
        The with_* arguments allow to choose whether means, variances and quantiles are plotted."""
        for i in range(min(Xt.shape[0], num_max)):
            plt.plot(t, Xt[i], alpha=alpha, color=color)
    
        if with_quantiles:
            self.plot_quantiles(t, Xt)

        if with_mean:
            self.plot_mean(t, Xt)

        if with_var:
            self.plot_var(t, Xt)

        if show:
            plt.show()

@dataclass
class Functionmaker():
    """This dataclass contains a set of standard functions that are commonly used
    within stochastic integration and stochastic differential equaltions."""

    def generate_linears(self, 
                         constant=[0, 0], 
                         part_t=[0, 0], 
                         part_x=[0, 0], 
                         part_y=[0, 0], 
                         part_x_y=[0, 0]):
        """This function is able to generate any (list of) linear functions of the form
        f(t, x) = c0 + c1*t + c2*x + c3*t*x"""
        function_list = []
        
        for p_c, p_t, p_x, p_y, p_xy in zip(constant, part_t, part_x, part_y, part_x_y):

            def linear_function(t, x, y, pc=p_c, pt=p_t, px=p_x, py=p_y, pxy=p_xy):
                return pc + pt*t + px*x + py*y + pxy*x*y
            
            function_list.append(linear_function)

        return function_list
    
    @staticmethod
    def f_func(t, x):
        return np.sin(t) + x
    
    @staticmethod
    def f_ornstein(t,x):
        """Mean function of Ornstein-Uhlenbeck process for testing purposes"""
        return 0.7*(1.5-x)

    @staticmethod
    def g_ornstein(t,x):
        """Variance function of Ornstein-Uhlenbeck process"""
        return 0.06
    
    #dSt = St(μdt + σdWt)
    @staticmethod
    def f_blackscholes(t, x):
        return 0.1*x
    
    @staticmethod
    def g_blackscholes(t, x):
        return 0.05*x
    
class StochasticProcess(Plotprocess):
    """A framework that is able to generate a range of stochastic processes."""

    def __init__(self, time, dt, number, poisson_rate=None, scale=None, shape=None):
        self.time = time
        self.dt = dt
        self.number = number
        self.poisson_rate = poisson_rate
        self.scale = scale 
        self.shape = shape 
        self.steps = int(time/dt)

    def poissonprocess(self):
        """Generate sample paths of a poisson process. This is a step function that
        jumps by a value of 1 at exponential times."""
        num_samples = int(2*poisson.ppf(0.997, self.time / self.poisson_rate)) 

        event_times = np.cumsum(np.random.exponential(scale=self.poisson_rate, size=[self.number, num_samples]), axis=1)
        event_times_disc = np.round(event_times / self.dt) * self.dt

        t = np.arange(0, self.time, self.dt)
        Nt = np.zeros((self.number, t.shape[0]))

        for idx, time in enumerate(t):
            Nt[:, idx] = np.sum(event_times_disc < time, axis=1)

        return t, Nt.astype(int)
    
    def laplaceprocess(self):
        """Generate a laplace process with class-bound scale parameter. 
        This is a Poisson process with laplace-distributed jumps"""
        t, Nt = self.poissonprocess()
        Lt = np.zeros_like(Nt, dtype=float)

        num_jump_sizes = int(2*poisson.ppf(0.997, self.time / self.poisson_rate))
        jump_sizes = np.cumsum(np.random.laplace(loc=0, scale=self.shape, size=[self.number, num_jump_sizes]), axis=0)
        jump_sizes = np.hstack((np.zeros((self.number, 1)), jump_sizes))

        for i in range(Nt.shape[0]):
            Lt[i,] = jump_sizes[i, Nt[i,]]

        return t, Lt
    
    def compoundgamma(self):
        """Generate a compound gamma process with class-bound scale parameter. 
        This is a Poisson process with laplace-distributed jumps"""
        t, Nt = self.poissonprocess()
        Lt = np.zeros_like(Nt, dtype=float)

        num_jump_sizes = int(2*poisson.ppf(0.997, self.time / self.poisson_rate))
        jump_sizes = np.cumsum(np.random.gamma(shape=self.shape, scale=self.scale, size=[self.number, num_jump_sizes]), axis=0)
        jump_sign = np.random.binomial(1, 0.5, size=[self.number, num_jump_sizes])
    
        jump_sizes *= 2*(jump_sign-0.5)

        jump_sizes = np.hstack((np.zeros((self.number, 1)), jump_sizes))

        for i in range(Nt.shape[0]):
            Lt[i,] = jump_sizes[i, Nt[i,]]

        return t, Lt
    
    def compoundgammaprocess(self, scale=1):
        """Generate a Gamma jump process. This is a Brownian motion plus a Gamma process."""
        t, Bt = self.brownianmotion()
        t, Lt = self.compoundgamma()
        Xt = Bt + Lt
        return t, Xt
    
    def levyprocess(self):
        """Generate a Laplace jump process. This is a Brownian motion plus a Laplace process."""
        t, Bt = self.brownianmotion()
        t, Lt = self.laplaceprocess()
        Xt = Bt + Lt
        return t, Xt

    def symmetricpoissonprocess(self):
        """Not a conventional process, a Brownian motion plus a symmetric Poisson process."""
        t, Nt1 = self.poissonprocess()
        _, Nt2 = self.poissonprocess()
        Mt = Nt1 - Nt2
        return t, Mt
    
    def brownianmotion(self):
        """Generate sample paths of a Brownian motion."""
        t = np.arange(0, self.time, self.dt)
        Bt = np.random.normal(loc=0, scale=np.sqrt(self.dt), 
                              size=(self.number, self.steps-1))
        Bt = np.hstack((np.zeros((Bt.shape[0], 1)), Bt))
        Bt = np.cumsum(Bt, axis=1)
        return t, Bt
        
    @staticmethod
    def findstoppingindices(Mt, tau):
        """Finds value of stopping time. If process is not yet stopped, the last time-index is returned.
        For accurate computations, small timesteps (<0.001) are important."""

        tau_index_upper = np.argmax(Mt >= tau, axis=1)
        tau_index_lower = np.argmax(-Mt >= tau, axis=1)

        tau_index_upper[tau_index_upper==0] = Mt.shape[1]-1
        tau_index_lower[tau_index_lower==0] = Mt.shape[1]-1

        tau_index = np.minimum(tau_index_lower, tau_index_upper)

        return tau_index

    def stoppingstatistics(self, Xt, t, tau):
         # for the first part, define a function that does this fo you
        tau_index = self.findstoppingindices(Xt, tau)

        stopping_times = t[tau_index]

        mean_stopping = np.mean(stopping_times)
        var_stopping = np.var(stopping_times)

        return mean_stopping, var_stopping
    
    def stoppedprocess(self, Xt, t, tau):
         # for the first part, define a function that does this fo you
        tau_index = self.findstoppingindices(Xt, tau)

        Xt_stopped = Xt.copy()
        for i, idx in enumerate(tau_index):
            Xt_stopped[i, idx:] = np.nan

        return Xt_stopped


class StochasticIntegration(StochasticProcess, Functionmaker):
    """Contains the methods that perform stochastic integration. This class is actually unneccessary as it is a special case of solving an SDE of the form:
    dXs = f(s,Ms)dMs"""

    def __init__(self, time, dt, number, poisson_rate=None, scale=None, shape=None, integrator="brownianmotion"):

        super().__init__(time, dt, number, poisson_rate, scale, shape)

        if integrator=="brownianmotion":
            self.integrator = self.brownianmotion
        elif integrator=="levyprocess":
            self.integrator = self.levyprocess
        elif integrator=="brownianskellamprocess":
            self.integrator = self.brownianskellamprocess
        elif integrator=="compoundgammaprocess":
            self.integrator = self.compoundgammaprocess
        else:
            self.integrator = self.brownianmotion
            print("Invalid input. Brownian motion selected as integrator")


    def stochastic_integral(self, fun):
        """Computes samples of the stochastic integral w.r.t. a martingale. Requires a function f(t,x) as input (x=Xt)"""
                
        t, Mt = self.integrator()
        dMt = np.hstack((np.diff(Mt), np.zeros((Mt.shape[0], 1)))) 
        Xt = np.cumsum(fun(t, Mt) * dMt, axis=1)
        return t, Xt
    
class SdeSolver(StochasticIntegration, Functionmaker):

    def __init__(self, time, dt, number, poisson_rate=None, scale=None, shape=None, integrator="brownianmotion"):
        
        super().__init__(time, dt, number, poisson_rate, scale, shape, integrator)

    @staticmethod
    def get_derivative(func, x, y, h=10e-8):
        """Obtain the numerical derivative of a function. Function is used to compute the derivative of the function g in SDE's."""
        y_prime = (func(x, y+h/2) - func(x, y-h/2)) / h # <- Is this the correct derivative?
        return y_prime

    def timestep_euler_maruyama(self, old_value, f_func, g_func, time, dt, dMt):
        """Performs a single step of the Euler-Maruyama method"""
        new_value = old_value + f_func(time, old_value)*dt + g_func(time, old_value)*dMt
        return new_value

    def timestep_milstein(self, old_value, f_func, g_func, time, dt, dMt):
        """Performs a single step of the Milstein method"""

        g_func_part = g_func(time, old_value)
        g_func_deriv = self.get_derivative(g_func, time, old_value)

        step_euler_maruyama = self.timestep_euler_maruyama(old_value, f_func, g_func, time, dt, dMt)
        step_milstein_correction = .5 * g_func_part * g_func_deriv * (dMt**2-dt)

        new_value_milstein = step_euler_maruyama + step_milstein_correction
        return new_value_milstein
           
        
    def solve_sde(self, f_func, g_func, num=50, value_init=0, method="Euler Maruyama"):
        """Enter a SDE in the form dX(t)=f(t,Xt)X(t)dt + g(t,Xt)dB(t)"""

        if method=="Euler Maruyama":
            timestep_function = self.timestep_euler_maruyama
        elif method=="Milstein":
            timestep_function = self.timestep_milstein
        
        t, Mt = self.integrator()
        dMt = np.hstack((np.diff(Mt), np.zeros((Mt.shape[0], 1)))) 
        Xt = np.full_like(Mt, fill_value=value_init) 

        for i in range(1, Mt.shape[1]):
            Xt[:,i] = timestep_function(Xt[:,i-1], f_func, g_func, t[i-1], self.dt, dMt[:,i])

        return t, Xt
    
    def solve_sde_system(self, f_funcs, g_funcs, num=50, value_init=0):
        pass
    
    @staticmethod
    def european_option(t, x, call_time, strike_price):
        """Compute the value of a set of sample paths at a specific time. 
        Returns the value max(S-K, 0) where S is the strike price and K the option price"""
        strike_time_idx = np.where(t==call_time) 
        strike_time_values = x[:, strike_time_idx].flatten()
        
        return np.maximum(strike_time_values-strike_price, 0)
    
    def plot_returns(self, t, x, call_time, strike_price):
        """DO NOT USE! could be removed or moved to plotting class"""
        returns = self.european_option(t, x, call_time, strike_price)
        sns.kdeplot(returns, fill=True)
        plt.show()
    

if __name__=="__main__":    
    sdesolve = SdeSolver(time=4, dt=0.001, number=100, poisson_rate=1, scale=0.05, shape=0.1, integrator="levyprocess")

    f_func = sdesolve.f_blackscholes
    g_func = sdesolve.g_blackscholes
    
    t, Xt = sdesolve.solve_sde(f_func=f_func, g_func=g_func, value_init=1, method="Milstein")
    sdesolve.plot_solution(t, Xt, with_mean=True, with_var=False, with_quantiles=True)