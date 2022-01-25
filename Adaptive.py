import numpy as np


class FIRNLMS:
    

    def __init__(self,memorysize=100,stepsize=0.1,regularization=1e-6):
        """
            Parameters:
                memorysize, stepsize and regularization
        """
        self.N = memorysize
        self.mu = stepsize
        self.psi = regularization
        self.finished = False
        self.ww = None

    def run(self,insignal,outsignal,maxiter=None):
        if not maxiter:
            rangesim = insignal.shape[0]
        else: 
            rangesim = min(insignal.shape[0],maxiter)
        if insignal.shape[0] != outsignal.shape[0]:
            raise Exception("Input and output must be vectors with same length.")
        self.xx = np.zeros(self.N)
        self.ww = np.zeros(self.N)
        self.sqerror = np.zeros(rangesim)
        for n in range(rangesim):
            self.xx[1:] = self.xx[0:-1]
            self.xx[0] = insignal[n]
            y = self.xx @ self.ww
            e = outsignal[n] - y
            self.ww = self.ww + self.mu * e * self.xx / (self.xx @ self.xx - self.psi)
            self.sqerror[n] = e**2            
        self.finished = True

