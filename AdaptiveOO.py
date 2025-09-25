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


class FIRFxNLMS:

	def __init__(sf,mem,memsec=0):
		sf.mem = mem # Memory size
		sf.ww = np.zeros(mem) # Coefficient vector        
		sf.mu = 0.1 # Step-size parameter
		sf.fi = 1e-6 # Regularization parameter        
		if (memsec > 0):
			sf.memsec = memsec
			sf.wwsec = np.zeros(memsec) # Sec. path coefficient vector      
			sf.xxf = np.zeros(mem)
		else:
			sf.memsec = 0
		sf.secondaryfilter = None
		sf.vecsize = (mem if (mem > memsec) else memsec)
		sf.xx = np.zeros(sf.vecsize) # Input vector
		sf.y = 0 # Filter output
		sf.e = 0 # Error
		sf.norm = 0
		sf.setAlgorithm('NLMS')

	def reset(sf):
		sf.ww = np.zeros(sf.mem)
		sf.xxf = np.zeros(sf.mem)
		sf.xx = np.zeros(sf.vecsize)
		sf.y = 0
		sf.e = 0
		sf.norm = 0
		if sf.secondaryfilter is not None:
			sf.secondaryfilter.reset()

	def setSecondary(sf,secfilter: FIR):
		sf.secondaryfilter = secfilter

	def setParams(sf,mu,fi):
		sf.mu = mu
		sf.fi = fi

	def evalout(sf,x):
		sf.xx[1:sf.vecsize] = sf.xx[0:sf.vecsize-1]
		sf.xx[0] = x
		sf.y = sf.xx[0:sf.mem] @ sf.ww
		sf.xxf[1:sf.mem] = sf.xxf[0:sf.mem-1]
		sf.xxf[0] = sf.secondaryfilter.filterstep(sf.xx[0])

	def LMSupdate(sf,e):
		sf.norm = sf.ww @ sf.ww
		sf.ww = sf.ww + 2 * sf.mu * e * sf.xxf

	def NLMSupdate(sf,e):
		sf.norm = sf.ww @ sf.ww
		sf.ww = sf.ww + sf.mu * e * sf.xxf / ((sf.xxf@sf.xxf) + sf.fi)

	def setAlgorithm(sf,alg='NLMS'):
		if alg == 'LMS':
			sf.update = sf.LMSupdate
		else:
			sf.update = sf.NLMSupdate



