import numpy as np
import random
from math import sqrt, log
import pprint
import itertools
import ast
import glob  # Optional, it improves the aesthetics of plots
import json
import os
import sys

def parameters1():
    ns = [500*i for i in range(1, 16)]
    rate0 = [r"$\log(n)$" for n in ns]
    mu0s = [log(n) for n in ns]
    rate1 = [r"$n^{1/2}\log(n)^{-1/2}$" for n in ns]
    mu1s = [n**(1/2) * log(n) ** (-1/2) for n in ns]
    rate2 = [r"$n^{1/2}\log(n)$" for n in ns]
    mu2s = [n**(1/2) * log(n) for n in ns]
    rate3 = [r"$n^{3/4}$" for n in ns]
    mu3s = [n**(3/4) for n in ns]
    mus = mu0s + mu1s + mu2s + mu3s
    rates = rate0 + rate1 + rate2 + rate3
    ns = ns * 4
    
    params = list(zip(ns, mus, rates))
    
    # Write to file
    with open('params1.txt', 'w') as f:
        for param in params:
            formatted_param = ','.join(map(str, param))
            f.write(f'{formatted_param}\n')


def parameters2():
    params = [200+500*i for i in range(15)]
    
    # Write to file
    with open('params2.txt', 'w') as f:
        for param in params:
            f.write(f'{str(param)}\n')



def parameters3():
    params = [200+500*i for i in range(15)]
    # Write to file
    with open('params3.txt', 'w') as f:
        for param in params:
            f.write(f'{str(param)}\n')

def parameters4():
    params = [200+500*i for i in range(15)]
    # Write to file
    with open('params4.txt', 'w') as f:
        for param in params:
            f.write(f'{str(param)}\n')

def parameters5():
    ns = [700]
    mu_rates = [r"$n^{4/5}$", r"$n^{5/6}$"]
    lambdamin_rates = [r"$2.05\sqrt{n}$", r"$3\sqrt{n}$"]
    
    params = itertools.product(ns, mu_rates, lambdamin_rates)
    
    # Write to file
    with open('params5.txt', 'w') as f:
    
        for param in params:
            n, mu_rate, lambdamin_rate = param
            if mu_rate == r"$n^{4/5}$":
                mu = n**(4/5)
            else:
                mu = n**(5/6)
            
            if lambdamin_rate == r"$2.05\sqrt{n}$":
                lambdamin = 2.05*sqrt(n)
            elif lambdamin_rate == r"$3\sqrt{n}$":
                lambdamin = 3*sqrt(n)

            param = [n, mu, mu_rate, lambdamin, lambdamin_rate]
            formatted_param = ','.join(map(str, param))
            f.write(f'{formatted_param}\n')
            

    
def parameters6():
    ns = [250, 500, 1000, 2000, 4000, 8000, 16000]
    delta_ratio0s = [log(n) for n in ns]
    delta_ratio1s = [n**(1/3) for n in ns]
    delta_ratio2s = [n**(1/2) for n in ns]
    delta_ratios = delta_ratio0s + delta_ratio1s + delta_ratio2s
    rates = [r"$\log(n)$" for n in ns] + [r"$n^{1/3}$" for n in ns] + [r"$n^{1/2}$" for n in ns]
    
    ns = ns * 3
    params = list(zip(ns, delta_ratios, rates))
    # Write to file
    with open('params6.txt', 'w') as f:
        for param in params:
            formatted_param = ','.join(map(str, param))
            f.write(f'{formatted_param}\n')


def parameters7():
    ns = [3500, 4500, 5500]
    const0s = [1.3 for n in ns]
    const1s = [1.4 for n in ns]
    const2s = [1.5 for n in ns]
    const3s = [1.7 for n in ns]
    const4s = [1.8 for n in ns]
    consts = const0s + const1s + const2s + const3s + const4s
    ns = ns * 5
    
    params = list(zip(ns, consts))
    
    # Write to file
    with open('params7.txt', 'w') as f:
        for param in params:
            formatted_param = ','.join(map(str, param))
            f.write(f'{formatted_param}\n')

def parameters8():
    params = [5000+100*i for i in range(15)]
    # Write to file
    with open('params8.txt', 'w') as f:
        for param in params:
            f.write(f'{str(param)}\n')

if __name__ == "__main__":
    n = int(sys.argv[1])
    if n == 1:
        parameters1()
    elif n == 2:
        parameters2()
    elif n == 3:
        parameters3()
    elif n == 4:
        parameters4()
    elif n == 5:
        parameters5()
    elif n == 6:
        parameters6()
    elif n == 7:
        parameters7()
    elif n == 8:
        parameters8()