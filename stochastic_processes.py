import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

class StochasticProcess():
    """"""

    def __init__(self, time, timestep, number, scale=None, shape=None):
        self.time = time
        self.timestep = timestep
        self.number = number
        self.scale = scale # this will be rate for poisson
        self.shape = shape # this will be sd for laplace
        # self.dict_par = {scale=scale, shape=shape}
        self.steps = int(time/timestep)

    def poissonprocess(self, rate):
        # Hij stijgt maar de helft van de snelheid die die moet stijgen
        # Voeg ook nog een scaler toe.
        num_samples = int(2*poisson.ppf(0.997, self.time / rate))

        event_times = np.cumsum(np.random.exponential(scale=rate, size=[self.number, num_samples]), axis=1)
        event_times_disc = np.round(event_times / self.timestep) * self.timestep

        t = np.arange(0, self.time, self.timestep)
        Nt = np.zeros((self.number, t.shape[0]))

        for idx, time in enumerate(t):
            Nt[:, idx] = np.sum(event_times_disc < time, axis=1)

        return t, Nt
    
    def skellamprocess(self, rate):
        """Not a conventional process, a brownian motion plus a stable mixture of Poisson processes"""
        t, Nt1 = self.poissonprocess(rate=rate)
        _, Nt2 = self.poissonprocess(rate=rate)
        Mt = Nt1 - Nt2
        return t, Mt
    
    def vgprocess(self):
        pass

    def symalphastableprocess(self):
        pass

    def brownianmotion(self):
        t = np.arange(0, self.time, self.timestep)
        Bt = np.random.normal(loc=0, scale=np.sqrt(self.timestep), 
                              size=(self.number, self.steps-1))
        Bt = np.hstack((np.zeros((Bt.shape[0], 1)), Bt))
        Bt = np.cumsum(Bt, axis=1)
        return t, Bt
    
    def brownianskellamprocess(self, rate):
        t, Bt = self.brownianmotion()
        t, Mt = self.skellamprocess(rate=rate)
        return t, Bt + Mt
    
    def levyprocess(self, scale=1):
        t, Bt = self.brownianmotion()
        t, Lt = self.levyprocess()

        # modify poisson process to multiply the jumps (now they are 1, always)
        # np.random.laplace(loc=0, scale=1, size=10)

    
    def geometricbrownianmotion(self, mu, sigma, s0=1):
        t, Bt = self.brownianmotion()
        Xt = s0 * np.exp((mu-sigma**2/2)*t + sigma*Bt)
        return t, Xt
    
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


class StochasticIntegration():

    def __init__(self, integrator="brownianmotion"):
        self.integrator = self.brownianmotion
        # Add other options when Levy process or other martingales are used.

    def stochastic_integral(self, fun):
        """Computes samples of the stochastic integral w.r.t. a martingale. Requires a function f(t,x) as input (x=Xt)"""
        
        # t, Mt = self.integrator(*arg)
        
        t, Mt = self.brownianmotion(self.time, self.dt, self.number)
        dMt = np.hstack((np.diff(Bt), np.zeros((Bt.shape[0], 1)))) 
        Xt = np.cumsum(fun(t, Mt) * dMt, axis=1)
        return t, Xt

class SdeSolver(StochasticIntegral):
    pass

    @staticmethod
    def f_ornstein(t,x):
        """Mean function of Ornstein-Uhlenbeck process for testing purposes"""
        return 0.7*(1.5-x)

    @staticmethod
    def g_ornstein(t,x):
        """Variance function of Ornstein-Uhlenbeck process"""
        return 0.06
    
    def solve_sde(time, dt, f_func=f_ornstein, g_func=g_ornstein, num=50):
        """Enter a SDE in the form dX(t)=f(t,Xt)X(t)dt + g(t,Xt)dB(t)"""

        t, Bt = brownianmotion(time, dt, num)
        # t, Xt = self.integrator() # Add *arg for inputting the required variables
        dBt = np.hstack((np.diff(Bt), np.zeros((Bt.shape[0], 1)))) 

        Xt = np.full_like(Bt, fill_value=0) 

        for i in range(1, Bt.shape[1]):
            Xt[:,i] = Xt[:,i-1] + f_func(t[i-1],Xt[:,i-1])*dt + g_func(t[i-1],Xt[:,i-1])*dBt[:,i]

        return t, Xt



    

if __name__=="__main__":
    process = StochasticProcess(time=5, timestep=0.01, number=200)
    
    # t, Xt = process.brownianmotion()
    # t, Xt = process.geometricbrownianmotion(mu=1, sigma=0.5)
    # t, Xt = process.poissonprocess(rate=1/2)
    t, Xt = process.skellamprocess(rate=1/2)
    t, Xt = process.brownianskellamprocess(rate=5)
    
    process.plot_solution(t, Xt, with_mean=True, with_var=False)





