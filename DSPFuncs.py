import numpy as np
from scipy import signal

def easyFourier(x: np.ndarray, fs: float = 1.0, N: int = None, phasealso: bool = False, downsamplemode: str = "decimate"):
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
        # Calcula a FFT para todo o sinal
        fftaux = np.fft.fft(x)
        magdb = 20 * np.log10(2 * np.abs(fftaux / nsamples)[:int(np.floor(nsamples / 2))])
        freqs = (np.fft.fftfreq(nsamples) * fs)[:magdb.shape[0]]

        if N:
            # Calcula o fator de decimação
            factor = int(np.ceil(float(nsamples) / float(N)))
            if phasealso:
                return magdb[::factor], freqs[::factor], np.angle(fftaux[:int(np.floor(nsamples / 2))][::factor])
            else:
                return magdb[::factor], freqs[::factor]
        else:
            if phasealso:
                return magdb, freqs, np.angle(fftaux[:int(np.floor(nsamples / 2))])
            else:
                return magdb, freqs

    elif downsamplemode == "truncate":
        if not N:
            N = nsamples
        fftaux = np.fft.fft(x, N)
        magdb = 20 * np.log10(2 * np.abs(fftaux / N)[:int(np.floor(N / 2))])
        freqs = (np.fft.fftfreq(N) * fs)[:magdb.shape[0]]

        if phasealso:
            return magdb,freqs,np.angle(fftaux[0:int(np.floor(nsamples/2))])
        else:
            return magdb,freqs
    else:
        raise BaseException("Invalid value for downsamplemode.")

def ARSmooth(sig, coef=0.95):
    """
    Suavizador de sinal baseado em modelo AR (Autoregressivo).

    Parâmetros:
    -----------
    sig : numpy.ndarray
        Sinal a ser suavizado.
    coef : float, opcional
        Coeficiente do modelo AR. Controla o grau de suavização. Padrão é 0.95.

    Retorno:
    --------
    out : numpy.ndarray
        Sinal suavizado.
    """
    last = 0
    out = np.zeros(sig.shape)
    for n in range(sig.shape[0]):
        out[n] = coef * last + sig[n] * (1 - coef)
        last = out[n]
    return out

class DCRemover:
    """
    Classe para remoção de componente DC de um sinal usando um filtro IIR.

    Parâmetros:
    -----------
    alpha : float, opcional
        Coeficiente do filtro, deve estar próximo de 1 para permitir uma remoção eficaz da componente DC. Padrão é 0.99.
    
    Métodos:
    --------
    filter(x: numpy.ndarray) -> numpy.ndarray
        Remove a componente DC do sinal `x`.
    """
    
    # http://sam-koblenski.blogspot.com/2015/11/everyday-dsp-for-programmers-dc-and.html

    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.wn = 0
        self.wn1 = 0

    def filter(self, x):
        self.y = np.zeros(x.shape[0])
        self.wn = 0
        self.wn1 = 0
        for k in range(x.shape[0]):
            self.wn1 = self.wn
            self.wn = x[k] + self.alpha * self.wn1
            self.y[k] = self.wn - self.wn1
        return self.y

def NLargestPeaks(n, freq, mag, distance=20):
    """
    Encontra os `n` maiores picos de magnitude no espectro de frequência.

    Parâmetros:
    -----------
    n : int
        Número de maiores picos a serem encontrados.
    freq : numpy.ndarray
        Vetor contendo as frequências.
    mag : numpy.ndarray
        Vetor contendo as magnitudes.
    distance : int, opcional
        Distância mínima entre picos, em amostras. Padrão é 20.

    Retorno:
    --------
    dict
        Dicionário contendo as frequências e magnitudes dos maiores picos.
    """
    pks = signal.find_peaks(mag, distance=distance)
    freqpks = freq[pks[0]]
    magpks = mag[pks[0]]
    idxnlargest = np.argsort(magpks)[-n:]
    freqpks = freqpks[idxnlargest]
    magpks = magpks[idxnlargest]
    idxsortbyfreq = np.argsort(freqpks)
    freqpks = freqpks[idxsortbyfreq]
    magpks = magpks[idxsortbyfreq]
    return {'freqs': freqpks, 'mags': magpks}

def freqAnalysis(signal, axis, fs=250, removeDC=True, labelsize=8, npeaks=3, peakdistance=50):
    """
    Análise de frequência de um sinal e plot do espectro de magnitude.

    Parâmetros:
    -----------
    signal : numpy.ndarray
        Sinal a ser analisado.
    axis : matplotlib.axes._subplots.AxesSubplot
        Eixo onde o espectro será plotado.
    fs : float, opcional
        Frequência de amostragem (Hz). Padrão é 250 Hz.
    removeDC : bool, opcional
        Se True, remove a componente DC do sinal antes da análise. Padrão é True.
    labelsize : int, opcional
        Tamanho da fonte dos rótulos dos eixos. Padrão é 8.
    npeaks : int, opcional
        Número de maiores picos a serem destacados no espectro. Padrão é 3.
    peakdistance : int, opcional
        Distância mínima entre picos, em amostras. Padrão é 50.

    Retorno:
    --------
    None
    """
    if removeDC:
        mag, freq = easyFourier(signal - np.mean(signal), fs=fs)
    else:
        mag, freq = easyFourier(signal, fs=fs)

    axis.plot(freq, mag)
    # axis.set_ylim(-100,20)
    axis.tick_params(labelsize=labelsize)
    
    # Encontrando e plotando os picos mais significativos
    pks = NLargestPeaks(npeaks, freq, mag, distance=peakdistance)
    axis.plot(pks['freqs'], pks['mags'], "xr")
    for f, m in zip(pks['freqs'], pks['mags']):
        axis.text(f, m, f" {f:.2f} Hz", fontsize=7)
