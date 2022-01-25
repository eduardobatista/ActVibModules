import numpy as np

class DCRemover():
  
  # http://sam-koblenski.blogspot.com/2015/11/everyday-dsp-for-programmers-dc-and.html
  def __init__(sf,alpha=0.99):
    sf.alpha = alpha
    sf.wn = 0
    sf.wn1 = 0
    
  def filter(sf,x):
    sf.y = np.zeros(x.shape[0])
    sf.wn = 0
    sf.wn1 = 0
    for k in range(x.shape[0]):
      sf.wn1 = sf.wn
      sf.wn = x[k] + sf.alpha * sf.wn1
      sf.y[k] = sf.wn - sf.wn1
    return sf.y