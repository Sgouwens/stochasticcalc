import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import poisson
from dataclasses import dataclass

# source c:/users/gouwenss/miniconda3/etc/profile.d/conda.sh
# TODO

# 4) Split the classes into multiple:
# 4.1) One class for generating the functions as input for LV-sde
# 4.2) Separate class for computing different options (just do european/american/asian)
# 4.3) Think of where plot statements are best places. And does it make sense to put the plotting of paths/denstiies in the same class?
# 4.4) Separate subclass that can deal with stopping times?

# 5) Put the parameters for poisson rate etc in the ppoisson things. do not keep them so general in the class call

class Options():
    pass

# This could be a dataclass, maybe. Finish and make it work with Lotka Volterra
@dataclass
class Functionmaker():
    """This dataclass contains a set of standard functions that are commonly used
    within stochastic integration and stochastic differential equaltions."""

    def generate_linears(self, constant, part_x, part_y, part_x_y):
        """This function is able to generate any (list of) linear functions of the form
        f(t, x) = c0 + c1*t + c2*x + c3*t*x"""
        function_list = []
        
        for p_c, p_x, p_y, p_xy in zip(constant, part_x, part_y, part_x_y):

            def linear_function(x, y, pc=p_c, px=p_x, py=p_y, pxy=p_xy):
                return pc + px*x + py*y + pxy*x*y
            
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
    
class StochasticProcess():
    """A framework that is able to generate a range of stochastic processes."""

    def __init__(self, time, timestep, number, poisson_rate=None, scale=None, shape=None):
        self.time = time
        self.timestep = timestep
        self.number = number
        self.poisson_rate = poisson_rate
        self.scale = scale 
        self.shape = shape 
        self.steps = int(time/timestep)

    def poissonprocess(self):
        # Hij stijgt maar de helft van de snelheid die die moet stijgen
        # Voeg ook nog een scaler toe.
        num_samples = int(2*poisson.ppf(0.997, self.time / self.poisson_rate)) # This can be turned into a separate function

        event_times = np.cumsum(np.random.exponential(scale=self.poisson_rate, size=[self.number, num_samples]), axis=1)
        event_times_disc = np.round(event_times / self.timestep) * self.timestep

        t = np.arange(0, self.time, self.timestep)
        Nt = np.zeros((self.number, t.shape[0]))

        for idx, time in enumerate(t):
            Nt[:, idx] = np.sum(event_times_disc < time, axis=1)

        return t, Nt.astype(int)
    
    def laplaceprocess(self):
        """Generates a laplace process with class-bound scale parameter. This is simply a poisson process with laplace-distributed jumps"""
        t, Nt = self.poissonprocess()
        Lt = np.zeros_like(Nt, dtype=float)

        num_jump_sizes = int(2*poisson.ppf(0.997, self.time / self.poisson_rate))
        jump_sizes = np.cumsum(np.random.laplace(loc=0, scale=self.shape, size=[self.number, num_jump_sizes]), axis=0)
        jump_sizes = np.hstack((np.zeros((self.number, 1)), jump_sizes))

        for i in range(Nt.shape[0]):
            Lt[i,] = jump_sizes[i, Nt[i,]]

        return t, Lt
    
    def compoundgamma(self):
        """Generates a compound gamma process with class-bound scale parameter. This is simply a poisson process with laplace-distributed jumps"""
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
        """Generates a Levy process. This is a brownian motion plus a laplace process."""
        t, Bt = self.brownianmotion()
        t, Lt = self.compoundgamma()
        Xt = Bt + Lt
        return t, Xt
    
    def levyprocess(self):
        """Generates a Levy process. This is a brownian motion plus a laplace process."""
        t, Bt = self.brownianmotion()
        t, Lt = self.laplaceprocess()
        Xt = Bt + Lt
        return t, Xt

    def symmetricpoissonprocess(self):
        """Not a conventional process, a brownian motion plus a stable mixture of Poisson processes"""
        t, Nt1 = self.poissonprocess()
        _, Nt2 = self.poissonprocess()
        Mt = Nt1 - Nt2
        return t, Mt
    
    def brownianmotion(self):
        t = np.arange(0, self.time, self.timestep)
        Bt = np.random.normal(loc=0, scale=np.sqrt(self.timestep), 
                              size=(self.number, self.steps-1))
        Bt = np.hstack((np.zeros((Bt.shape[0], 1)), Bt))
        Bt = np.cumsum(Bt, axis=1)
        return t, Bt
    
    def brownianpoissonprocess(self):
        t, Bt = self.brownianmotion()
        t, Mt = self.symmetricpoissonprocess()
        return t, Bt + Mt
    
    def geometricbrownianmotion(self, mu, sigma, s0=1):
        """Could be removed. this should be the solution of black-scholes"""
        t, Bt = self.brownianmotion()
        Xt = s0 * np.exp((mu-sigma**2/2)*t + sigma*Bt)
        return t, Xt
    
    @staticmethod
    def findstoppingindices(Mt, tau):
        """Finds value of stopping time. If process is not yet stopped, the last time-index is returned"""

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


    @staticmethod
    def plot_mean(t, Xt):
        Xt_mean = np.mean(Xt, axis=0)
        plt.plot(t, Xt_mean, color="black")

    @staticmethod
    def plot_var(t, Xt):
        Xt_var = np.var(Xt, axis=0)
        plt.plot(t, Xt_var, color="black")

    def plot_solution(self, t, Xt, num_max=50, alpha=0.4, with_mean=False, with_var=False, color=None):
        for i in range(min(Xt.shape[0], num_max)):
            plt.plot(t, Xt[i], alpha=alpha, color=color)
    
        if with_mean:
            self.plot_mean(t, Xt)

        if with_var:
            self.plot_var(t, Xt)

        plt.show()

class StochasticIntegration(StochasticProcess, Functionmaker):
    """Contains the methods that perform stochastic integration. This class is actually unneccessary as it is a special case of solving an SDE of the form:
    dXs = f(s,Ms)dMs"""

    def __init__(self, time, timestep, number, poisson_rate=None, scale=None, shape=None, integrator="brownianmotion"):

        super().__init__(time, timestep, number, poisson_rate, scale, shape)

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


    def stochastic_integral(self, fun=f_func):
        """Computes samples of the stochastic integral w.r.t. a martingale. Requires a function f(t,x) as input (x=Xt)"""
        
        # t, Mt = self.integrator(*arg)
        
        t, Mt = self.integrator()
        dMt = np.hstack((np.diff(Mt), np.zeros((Mt.shape[0], 1)))) 
        Xt = np.cumsum(fun(t, Mt) * dMt, axis=1)
        return t, Xt
    
class SdeSolver(StochasticIntegration):

    def __init__(self, time, timestep, number, poisson_rate=None, scale=None, shape=None, integrator="brownianmotion"):
        
        super().__init__(time, timestep, number, poisson_rate, scale, shape, integrator)

         
    def solve_sde(self, value_init=0, f_func=f_blackscholes, g_func=g_blackscholes, num=50):
        """Enter a SDE in the form dX(t)=f(t,Xt)X(t)dt + g(t,Xt)dB(t)"""
        t, Mt = self.integrator()
        dMt = np.hstack((np.diff(Mt), np.zeros((Mt.shape[0], 1)))) 

        Xt = np.full_like(Mt, fill_value=value_init) 

        for i in range(1, Mt.shape[1]):
            Xt[:,i] = Xt[:,i-1] + f_func(t[i-1],Xt[:,i-1])*self.timestep + g_func(t[i-1],Xt[:,i-1])*dMt[:,i]

        return t, Xt
    
    @staticmethod
    def european_option(t, x, call_time, strike_price):

        strike_time_idx = np.where(t==call_time) # warning if not found. due to fragmentation. could interpolate but ,.... geen zin in
        strike_time_values = x[:, strike_time_idx].flatten() # flattening to meet dimensional requirements
        
        return np.maximum(strike_time_values-strike_price, 0)
    

    # American options can be exercised at any moment before the expiration date. In order to make a function, we also need to define an exercise strategy
    
    def plot_returns(self, t, x, call_time, strike_price):

        returns = self.european_option(t, x, call_time, strike_price)
        sns.kdeplot(returns, fill=True)
        plt.show()
    

if __name__=="__main__":
    
    # sdesolve = SdeSolver(time=10, timestep=0.02, number=2500, poisson_rate=4, shape=1, scale=1, integrator="brownianmotion")
    # t, Xt = sdesolve.solve_sde(value_init=1)
    # sdesolve.plot_solution(t, Xt, with_mean=True, with_var=False)

    # sdesolve.plot_returns(t, Xt, call_time=9, strike_price=1.2)

    process = StochasticProcess(time=10, timestep=0.01, number=10, poisson_rate=5000, shape=.1, scale=.1)
    t, Xt = process.levyprocess()
    process.plot_solution(t, Xt, with_mean=True, with_var=False)

