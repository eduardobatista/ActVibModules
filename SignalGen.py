import numpy as np
from scipy.signal import butter, lfilter
from scipy.fftpack import fft

#class SignalGen:

def sine(Npoints,freqHz,samplingT,Amplitude=1):
    x = np.zeros(Npoints)
    for k in range(0,Npoints):
        x[k] = np.sin(2*np.pi*freqHz*k*samplingT)   
    return x

def filteredNoise(Npoints,var,lowcut,highcut,samplingFreq,order=10):
    xa = np.sqrt(var) * np.random.randn(Npoints)
    nyq = 0.5 * samplingFreq
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    x = lfilter(b, a, xa)
    return x

def chirp(seconds,samplingfreq,freq,ampl=1,deltai=10,deltaf=10,tinicio=10,tfim=None,alpha=0.9):
    dt = 1/samplingfreq
    tt = np.arange(0,seconds,dt) # Vetor de tempo de 0 a Nseconds com passos de dt
    out = np.zeros(tt.shape[0]) # Retorno
    f = 0 # Freq. inicial = 0
    integ = 0 # Integral inicialmente = 0
    cA = ampl/(freq**2) # Constante para amplitude ficar igual a "ampl" em regime
    A = 0 # Amplitude inicial = 9
    if tfim is None:
        tfim = seconds
    for n,t in enumerate(tt): # Varia n de 0 ao número de elementos no vetor tt, e t = tt[n]
        if (t < tinicio) or (t > (tfim+deltaf)):
            f = 0
            #A = 0                
        elif t <= (tinicio+deltai):
            f = freq * (t-tinicio) / deltai
            #A = ampl * (t-tinicio) / deltat                
        elif (t >= tfim) and (t <= (tfim+deltaf)):
            f = freq - freq * (t-tfim) / deltaf
            #A = ampl - ampl * (t-tfim) / deltat                
        A = (f**2) * cA
        integ = integ + 2*np.pi*f*dt
        out[n] = A * np.sin(integ)
    return out