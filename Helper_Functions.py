import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator, CubicSpline
from math import *
from scipy.stats import norm
import matplotlib.pyplot as plt

# Constants
beta = 0.75

# SABR Model Functions
class SABR():
    def __init__(self, alpha, beta, rho, nu, bs_delta, bs_vega, f,K,T):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.bs_delta = bs_delta
        self.bs_vega = bs_vega
        self.f = f
        self.K = K
        self.T = T

    def vol(self, f,alpha):
        eps = 1e-07
        logfk = np.log(f / self.K)
        fkbeta = (f*self.K)**(1 - self.beta)
        a = (1 - self.beta)**2 * alpha**2 / (24 * fkbeta)
        b = 0.25 * self.rho * self.beta * self.nu * alpha / fkbeta**0.5
        c = (2 - 3*self.rho**2) * self.nu**2 / 24
        d = fkbeta**0.5
        v = (1 - self.beta)**2 * logfk**2 / 24
        w = (1 - self.beta)**4 * logfk**4 / 1920
        z = self.nu * fkbeta**0.5 * logfk / alpha
        # if |z| > eps
        if abs(z) > eps:
            def _x(rho,z):
                return np.log((z-rho + sqrt(z*z - 2*rho*z + 1))/(1-rho))
            vz = alpha * z * (1 + (a + b + c) * self.T) / (d * (1 + v + w) * _x(self.rho, z))
            return vz
        # if |z| <= eps
        else:
            v0 = alpha * (1 + (a + b + c) * self.T) / (d * (1 + v + w))
            return v0
        
    def delta(self):
        h = 10**(-6)
        f_h = self.vol(f=self.f + h, alpha=self.alpha)
        f = self.vol(f=self.f - h, alpha=self.alpha)
        dsigma_df = (f_h - f)/(2*h)
        #return sabr delta
        return self.bs_delta + self.bs_vega*(dsigma_df)
    
    def vega(self):
        h = 10**(-6)
        f_h = self.vol(f=self.f,alpha=self.alpha + h)
        f = self.vol(f=self.f,alpha=self.alpha - h)
        dsigma_dalpha = (f_h - f)/(2*h)
        return self.bs_vega * dsigma_dalpha
    
def lognormal_vol(k, f, t, alpha, beta, rho, volvol):
    if k <= 0 or f <= 0:
        return 0.
    eps = 1e-07
    logfk = np.log(f / k)
    fkbeta = (f*k)**(1 - beta)
    a = (1 - beta)**2 * alpha**2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / fkbeta**0.5
    c = (2 - 3*rho**2) * volvol**2 / 24
    d = fkbeta**0.5
    v = (1 - beta)**2 * logfk**2 / 24
    w = (1 - beta)**4 * logfk**4 / 1920
    z = volvol * fkbeta**0.5 * logfk / alpha
    x = (z-rho + sqrt(z*z - 2*rho*z + 1))/(1-rho)
    if (abs(z) > eps) or (x <= 0):
        return alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * log(x))
    return alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))

def fit(strikes, forward, tenor, vols, initial_guess=[0.05, 0.01, 0.5]):
    def vol_square_error(x):
        imp_vols = [lognormal_vol(k, forward, tenor, x[0], beta, x[1], x[2]) for k in strikes]
        weight = [log(forward/abs(k-forward)) for k in strikes] 
        return sum(weight*(np.array(imp_vols) - vols)**2)
    
    bounds = [(0.0001, None), (-0.9999, 0.9999), (0.0001, None)]
    res = minimize(vol_square_error, initial_guess, method='L-BFGS-B', bounds=bounds)
    return res.x
   
def mserror(alpha, rho, nu, strikes, fwd, tenor, vols):
    errors = [((lognormal_vol(strikes[i], fwd, tenor, alpha, beta, rho, nu) - vols[i])**2)/vols[i] 
            for i in range(len(strikes))]
    return sum(errors)

# Black-Scholes Functions
def get_strike(row):
    delta, f, tenor, vol = row.delta, row.fwd, row.tenor, row.vol
    return f*exp(-vol*sqrt(tenor)*norm.ppf(delta) + (vol*vol/2)*tenor)

class BlackScholes():
    def __init__(self, S, f, K, T, sigma):
        self.S = S
        self.f = f
        self.K = K
        self.T = T
        self.sigma = sigma
    
    def d1(self):
        return (np.log(self.f / self.K) + (0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def price(self):
        return (self.S * norm.cdf(self.d1()) - self.K * (self.f/self.S) * norm.cdf(self.d2()))
    
    def vega(self):
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T)