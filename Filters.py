import numpy as np

class FIR:
    """
    FIR filter class.
    """
    def __init__(self, coeffs):
        self.w = coeffs
        self.x = np.zeros(coeffs.shape[0])
        self.N = coeffs.shape[0]
    
    def reset(self):
        self.x = np.zeros(self.N)
    
    def filterstep(self, xsample):
        self.x[1:] = self.x[:-1]
        self.x[0] = xsample
        ysample = self.w @ self.x
        return ysample

    def filter(self, x):
        y = np.zeros(x.shape)
        for k in range(x.shape[0]):
            y[k] = self.filterstep(x[k])
        return y