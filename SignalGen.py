import numpy as np
from scipy.signal import butter, lfilter
from scipy.fftpack import fft

def sine(Npoints, freqHz, samplingT, Amplitude=1):
    """
    Gera um sinal senoidal com número específico de pontos, frequência e período de amostragem.

    Parâmetros:
    -----------
    Npoints : int
        Número de pontos do sinal senoidal.
    freqHz : float
        Frequência do sinal senoidal em Hertz (Hz).
    samplingT : float
        Tempo de amostragem (intervalo entre amostras) em segundos.
    Amplitude : float, opcional
        Amplitude da senoide (valor padrão é 1).

    Retorno:
    --------
    x : numpy.ndarray
        Sinal senoidal gerado.
    """
    x = np.zeros(Npoints)
    for k in range(0, Npoints):
        x[k] = Amplitude * np.sin(2 * np.pi * freqHz * k * samplingT)
    return x

def filteredNoise(Npoints, var, lowcut, highcut, samplingFreq, order=10):
    """
    Gera um sinal de ruído filtrado em uma faixa de frequência específica.

    Parâmetros:
    -----------
    Npoints : int
        Número de pontos do sinal de ruído.
    var : float
        Variância do ruído a ser gerado.
    lowcut : float
        Frequência de corte inferior do filtro (Hz).
    highcut : float
        Frequência de corte superior do filtro (Hz).
    samplingFreq : float
        Frequência de amostragem (Hz).
    order : int, opcional
        Ordem do filtro Butterworth (valor padrão é 10).

    Retorno:
    --------
    x : numpy.ndarray
        Sinal de ruído filtrado gerado.
    """
    # Gerando o ruído gaussiano com variância especificada.
    xa = np.sqrt(var) * np.random.randn(Npoints)

    # Calculando as frequências normalizadas para o filtro.
    nyq = 0.5 * samplingFreq
    low = lowcut / nyq
    high = highcut / nyq

    # Criando o filtro Butterworth do tipo passa-faixa.
    b, a = butter(order, [low, high], btype='band')

    # Aplicando o filtro ao sinal de ruído.
    x = lfilter(b, a, xa)
    return x

def chirp(seconds, samplingfreq, freq, ampl=1, deltai=10, deltaf=10, tinicio=10, tfim=None):
    """
    Generates a signal that resembles the start, steady state and end of the functioning of a rotating machine.
    Such a signal is useful for testing active vibration control systems.
    The generated signal is based on the chirp signal.

    Args:
        seconds (int): Total time of the signal
        samplingfreq (int): Sampling frequency (Hz)
        freq (int): Signal frequency (Hz)
        ampl (int, optional): Signal amplitude at steady state. Defaults to 1.
        deltai (int, optional): Rising time of the signal. Defaults to 10.
        deltaf (int, optional): Fall time of the signal. Defaults to 10.
        tinicio (int, optional): Time of the beginning of signal rise. Defaults to 10.
        tfim (int, optional): Time of the beginning of signal fall. Defaults to None.

    Returns:
        numpy.ndarray: Generated signal
    """    
    # Definindo o passo de amostragem.
    dt = 1 / samplingfreq

    # Criando o vetor de tempo de 0 a "seconds" com passos de "dt".
    tt = np.arange(0, seconds, dt)

    # Inicializando o sinal de saída.
    out = np.zeros(tt.shape[0])

    # Inicializando variáveis de frequência e amplitude.
    f = 0  # Frequência inicial = 0.
    integ = 0  # Valor inicial da integral (fase).
    cA = ampl / (freq ** 2)  # Constante para garantir que a amplitude seja igual a "ampl" em regime.
    A = 0  # Amplitude inicial.

    # Caso tfim não seja especificado, assume-se o valor de "seconds".
    if tfim is None:
        tfim = seconds

    # Iterando sobre cada ponto de tempo para gerar o sinal.
    for n, t in enumerate(tt): # Varia n de 0 ao número de elementos no vetor tt, e t = tt[n]
        # Definindo a frequência ao longo do tempo.
        if (t < tinicio) or (t > (tfim + deltaf)):
            f = 0
        elif t <= (tinicio + deltai):
            f = freq * (t - tinicio) / deltai
        elif (t >= tfim) and (t <= (tfim + deltaf)):
            f = freq - freq * (t - tfim) / deltaf

        # Calculando a amplitude baseada na frequência ao quadrado.
        A = (f ** 2) * cA

        # Calculando a integral para definir a fase da senoide.
        integ += 2 * np.pi * f * dt

        # Calculando o valor do sinal no ponto atual.
        out[n] = A * np.sin(integ)

    return out
