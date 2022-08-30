import numpy as np

def easyFourier(x: np.ndarray, fs: float = 1.0, N: int = None, downsamplemode: str = "decimate"):
  """
    Evaluate the magnitude Fourier spectrum in dB using the FFT.
    Parameters:
      x: vector containing the signal in time.
      fs: the sampling frequency. 
      N: Either the number of samples for the transformed signal (when downsamplemode = "truncate") 
         or the approximate number os samples of the transformed signal (when downsamplemode = "decimate")
         (must be smaller than or equal to the length of x)
         (WARNING: obtained via decimation, data may be lost).
      downsamplemode: "decimate" (where the whole signal is considered when carrying out the FFT) or
                      "truncate" (considers only the first N samples of the signal - standard for numpy.fft)
    Returns: (magdb,freqvec)
      magdb: vector containing the magnitude samples of the fft (dB).
      freqs: vector containing the frequency values (Hz).
  """
  nsamples = x.shape[0]
  if downsamplemode == "decimate":
    magdb = 20*np.log10( 2*np.abs(np.fft.fft(x)/nsamples)[0:int(np.floor(nsamples/2))] )
    freqs = (np.fft.fftfreq(nsamples) * fs)[0:magdb.shape[0]]
    if N:
      factor = int(np.ceil(float(nsamples)/float(N)))
      return magdb[::factor],freqs[::factor]
    else:
      return magdb,freqs
  elif downsamplemode == "truncate":
    if not N:
      N = nsamples
    magdb = 20*np.log10( 2*np.abs(np.fft.fft(x,N)/N)[0:int(np.floor(N/2))] )
    freqs = (np.fft.fftfreq(N) * fs)[0:magdb.shape[0]]
    return magdb,freqs
  else:
    raise BaseException("Invalid value for downsamplemode.")


def ARSmooth(sig,coef=0.95):
  """
    AR signal smoother.
    - sig is the signal;
    - coef is the AR coef.
  """
  last = 0
  out = np.zeros(sig.shape)
  for n in range(0,sig.shape[0]):
    out[n] = coef*last + sig[n]*(1-coef)
    last = out[n]
  return out


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


