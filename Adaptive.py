import numpy as np

class FIRNLMS:    
    def __init__(self, memorysize=100, stepsize=0.1, regularization=1e-6, wwavgwindow=None):
        """
        Implementa um filtro adaptativo FIR usando o algoritmo NLMS (Normalized Least Mean Squares).

        Parâmetros:
        - memorysize (int): Tamanho da memória do filtro FIR, ou seja, o número de coeficientes.
        - stepsize (float): Tamanho do passo de atualização do NLMS, controlando a taxa de adaptação.
        - regularization (float): Parâmetro de regularização para evitar divisão por zero.
        - wwavgwindow (int, opcional): Tamanho da janela para cálculo da média dos coeficientes, se fornecido.
        """
        self.N = memorysize
        self.mu = stepsize
        self.psi = regularization
        self.finished = False
        self.ww = None
        self.wwavgwindow = wwavgwindow
        self.wwavg = None

    def run(self, insignal, outsignal, maxiter=None):
        """
        Método run: Executa o algoritmo NLMS com base no sinal de entrada e saída.

        Parâmetros:
        - insignal (np.array): Sinal de entrada.
        - outsignal (np.array): Sinal de saída desejado.
        - maxiter (int, opcional): Número máximo de iterações a serem realizadas.

        Lança uma exceção se o comprimento do sinal de entrada e saída não for o mesmo.
        Atualiza os coeficientes do filtro com base no erro entre o sinal de saída calculado e o sinal de saída esperado.
        """
        if not maxiter:
            rangesim = insignal.shape[0]
        else: 
            rangesim = min(insignal.shape[0], maxiter)
        if insignal.shape[0] != outsignal.shape[0]:
            raise Exception("Input and output must be vectors with same length.")
        self.xx = np.zeros(self.N)
        self.ww = np.zeros(self.N)
        self.sqerror = np.zeros(rangesim)
        if self.wwavgwindow:
            self.wwavg = np.zeros(self.N)
            for n in range(rangesim):
                self.xx[1:] = self.xx[0:-1]
                self.xx[0] = insignal[n]
                y = self.xx @ self.ww
                e = outsignal[n] - y
                self.ww = self.ww + self.mu * e * self.xx / (self.xx @ self.xx + self.psi)
                self.sqerror[n] = e**2
                if n >= (rangesim - self.wwavgwindow):
                    self.wwavg += self.ww
            self.wwavg = self.wwavg / self.wwavgwindow
        else:
            for n in range(rangesim):
                self.xx[1:] = self.xx[0:-1]
                self.xx[0] = insignal[n]
                y = self.xx @ self.ww
                e = outsignal[n] - y
                self.ww = self.ww + self.mu * e * self.xx / (self.xx @ self.xx + self.psi)
                self.sqerror[n] = e**2
        self.finished = True


class FIRFxNLMS:
    def __init__(sf, mem, memsec):
        """
        Extensão do FIRNLMS com um caminho secundário para modelagem de características adicionais.

        Parâmetros:
        - mem (int): Tamanho da memória do filtro principal.
        - memsec (int): Tamanho da memória do caminho secundário (opcional).
        """
        sf.mem = mem  # Tamanho da memória do filtro principal (número de coeficientes).
        sf.ww = np.zeros(mem)  # Vetor de coeficientes do filtro principal.
        sf.mu = 0.1  # Parâmetro de passo 
        sf.fi = 1e-6  # Parâmetro de regularização 
        if memsec > 0:
            sf.memsec = memsec 
            sf.wwsec = np.zeros(memsec)  
            sf.xxf = np.zeros(mem) # Vetor de entrada do caminho secundário  
        else:
            sf.memsec = 0 
        sf.vecsize = max(mem, memsec) # Tamanho do vetor de entrada.
        sf.xx = np.zeros(sf.vecsize)  # Vetor de entrada.
        sf.y = 0  # Saída do filtro.
        sf.e = 0  # Erro.
        sf.norm = 0 # Norma dos coeficientes.
        sf.setAlgorithm('NLMS')

    def reset(sf):
        """
        Reinicializa os coeficientes do filtro e as variáveis internas.
        """
        sf.ww = np.zeros(sf.mem) 
        sf.xxf = np.zeros(sf.mem)  
        sf.xx = np.zeros(sf.vecsize) 
        sf.y = 0
        sf.e = 0
        sf.norm = 0

    def setSecondary(sf, wwsec):
        """
        Define os coeficientes do caminho secundário.

        Parâmetros:
        - wwsec (np.array): Vetor de coeficientes para o caminho secundário.
        """
        sf.wwsec = wwsec

    def setParams(sf, mu, fi):
        """
        Define os parâmetros do filtro.

        Parâmetros:
        - mu (float): Parâmetro de passo.
        - fi (float): Parâmetro de regularização.
        """
        sf.mu = mu
        sf.fi = fi

    def evalout(sf, x):
        """
        Avalia a saída do filtro dado um novo valor de entrada.

        Parâmetros:
        - x (float): Sinal de entrada.
        """
        sf.xx[1:sf.vecsize] = sf.xx[0:sf.vecsize - 1]
        sf.xx[0] = x
        sf.y = sf.xx[0:sf.mem] @ sf.ww
        sf.xxf[1:sf.mem] = sf.xxf[0:sf.mem - 1]
        sf.xxf[0] = sf.xx[0:sf.memsec] @ sf.wwsec

    def LMSupdate(sf, e):
        """
        Atualiza os coeficientes usando o algoritmo LMS.

        Parâmetros:
        - e (float): Erro entre a saída calculada e a saída esperada.
        """
        sf.norm = sf.ww @ sf.ww
        sf.ww = sf.ww + 2 * sf.mu * e * sf.xxf

    def NLMSupdate(sf, e):
        """
        Atualiza os coeficientes usando o algoritmo NLMS.

        Parâmetros:
        - e (float): Erro entre a saída calculada e a saída esperada.
        """
        sf.norm = sf.ww @ sf.ww
        sf.ww = sf.ww + sf.mu * e * sf.xxf / ((sf.xxf @ sf.xxf) + sf.fi)

    def setAlgorithm(sf, alg='NLMS'):
        """
        Define qual algoritmo de atualização será usado (LMS ou NLMS).

        Parâmetros:
        - alg (str): Nome do algoritmo ('LMS' ou 'NLMS').
        """
        if alg == 'LMS':
            sf.update = sf.LMSupdate
        else:
            sf.update = sf.NLMSupdate


class LeakyFxNLMS(FIRFxNLMS):
    def __init__(sf, mem, memsec, leakfactor):
        """
        Implementa o algoritmo FIRFxNLMS com vazamento, que atenua os coeficientes a cada iteração.

        Parâmetros:
        - mem (int): Tamanho da memória do filtro principal.
        - memsec (int): Tamanho da memória do caminho secundário.
        - leakfactor (float): Fator de vazamento.
        """
        super().__init__(mem, memsec)
        sf.leakfactor = leakfactor

    def LMSupdate(sf, e):
        """
        Atualiza os coeficientes usando o algoritmo LMS com vazamento.

        Parâmetros:
        - e (float): Erro entre a saída calculada e a saída esperada.
        """
        sf.norm = sf.ww @ sf.ww
        sf.ww = sf.leakfactor * sf.ww + 2 * sf.mu * e * sf.xxf

    def NLMSupdate(sf, e):
        """
        Atualiza os coeficientes usando o algoritmo NLMS com vazamento.

        Parâmetros:
        - e (float): Erro entre a saída calculada e a saída esperada.
        """
        sf.norm = sf.ww @ sf.ww
        sf.ww = sf.leakfactor * sf.ww + sf.mu * e * sf.xxf / ((sf.xxf @ sf.xxf) + sf.fi)


class CVAFxNLMS:
    def __init__(sf, mem, memsec=0, mem2=0, memsec2=0):
        """
        Implementa um filtro adaptativo com dois filtros paralelos, cada um com seu próprio caminho secundário.

        Parâmetros:
        - mem (int): Tamanho da memória do primeiro filtro.
        - memsec (int): Tamanho da memória do caminho secundário do primeiro filtro.
        - mem2 (int): Tamanho da memória do segundo filtro.
        - memsec2 (int): Tamanho da memória do caminho secundário do segundo filtro.
        """
        sf.mem = mem
        sf.ww = np.zeros(mem)
        sf.mu = 0.1
        sf.fi = 1e-6
        sf.mem2 = mem2
        sf.ww2 = np.zeros(mem2)
        sf.mu2 = 0.1
        sf.fi2 = 1e-6
        sf.memsec = memsec
        sf.memsec2 = memsec2
        sf.xx = np.zeros(max(mem, mem2, memsec, memsec2))
        sf.xxf = np.zeros(mem)
        sf.xxf2 = np.zeros(mem2)
        sf.setAlgorithm('NLMS')

    def reset(sf):
        """
        Reinicializa os coeficientes e as variáveis internas de ambos os filtros.
        """
        sf.ww = np.zeros(sf.mem)
        sf.xxf = np.zeros(sf.mem)
        sf.xx = np.zeros(sf.xx.shape[0])
        sf.ww2 = np.zeros(sf.mem2)
        sf.xxf2 = np.zeros(sf.mem2)

    def setSecondary(sf, wwsec):
        """
        Define os coeficientes dos caminhos secundários.

        Parâmetros:
        - wwsec (np.array): Vetores de coeficientes dos caminhos secundários.
        """
        sf.wwsec = wwsec

    def setParams(sf, mu, fi, mu2=0):
        """
        Define os parâmetros de adaptação dos filtros.

        Parâmetros:
        - mu (float): Parâmetro de passo do primeiro filtro.
        - fi (float): Parâmetro de regularização do primeiro filtro.
        - mu2 (float, opcional): Parâmetro de passo do segundo filtro.
        """
        sf.mu = mu
        sf.fi = fi
        sf.mu2 = mu2

    def evalout(sf, x, x2):
        """
        Avalia a saída dos dois filtros dado os sinais de entrada.

        Parâmetros:
        - x (float): Sinal de entrada do primeiro filtro.
        - x2 (float): Sinal de entrada do segundo filtro.
        """
        sf.xx[1:] = sf.xx[0:-1]
        sf.xx[0] = x
        sf.xx2[0] = x2
        sf.y = sf.xx[:sf.mem] @ sf.ww
        sf.y2 = sf.xx[:sf.mem2] @ sf.ww2
        sf.xxf[1:sf.mem] = sf.xxf[:sf.mem - 1]
        sf.xxf[0] = sf.xx[:sf.memsec] @ sf.wwsec

    def LMSupdate(sf, e):
        """
        Atualiza os coeficientes de ambos os filtros usando o algoritmo LMS.

        Parâmetros:
        - e (float): Erro entre a saída calculada e a saída esperada.
        """
        sf.norm = sf.ww @ sf.ww
        sf.ww = sf.ww + 2 * sf.mu * e * sf.xxf
        sf.ww2 = sf.ww2 + 2 * sf.mu2 * e * sf.xxf2

    def NLMSupdate(sf, e):
        """
        Atualiza os coeficientes de ambos os filtros usando o algoritmo NLMS.

        Parâmetros:
        - e (float): Erro entre a saída calculada e a saída esperada.
        """
        sf.norm = sf.ww @ sf.ww
        sf.ww = sf.ww + sf.mu * e * sf.xxf / ((sf.xxf @ sf.xxf) + sf.fi)
        sf.ww2 = sf.ww2 + sf.mu2 * e * sf.xxf2 / ((sf.xxf2 @ sf.xxf2) + sf.fi2)

    def setAlgorithm(sf, alg='NLMS'):
        """
        Define qual algoritmo será usado para atualizar os coeficientes dos dois filtros.

        Parâmetros:
        - alg (str): Nome do algoritmo ('LMS' ou 'NLMS').
        """
        if alg == 'LMS':
            sf.update = sf.LMSupdate
        else:
            sf.update = sf.NLMSupdate
