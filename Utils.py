import numpy as np
from .CantileverBeam import CantileverBeam

class PathModeling:
    """
    Modelagem de trajetórias em uma viga específica.
    
    Atributos:
    ----------
    cbeam : CantileverBeam
        A viga a ser simulada (do tipo CantileverBeam).
    N : int, opcional
        Tamanho da memória dos modelos obtidos (padrão é 1000).
    type : int, opcional
        0 para modelagem ideal usando a resposta ao impulso.
        1 para modelagem adaptativa usando o algoritmo NLMS com fator de passo 0.25 e fator de normalização 1e-3 (padrão é 1).
    simtime : int, opcional
        Tempo de simulação para type = 1 (modelagem adaptativa) (padrão é 120).
    mode : int, opcional
        0 para aceleração (acelerômetro).
        1 para velocidade de rotação (giroscópio) (padrão é 0).
    """

    def __init__(self, cbeam: CantileverBeam, N=1000, type=1, simtime=120, mode=0):
        self.beam = cbeam
        self.N = N
        self.type = type
        self.simtime = simtime

        # Definindo a função de movimento de acordo com o modo selecionado.
        if mode == 0:
            self.movefunc = self.beam.getaccelms2  # Aceleração em m/s^2
        elif mode == 1:
            self.movefunc = self.beam.getrotationvel  # Velocidade angular

    def runModelling(self, pinput, poutput):
        """
        Executa a modelagem da trajetória aplicando uma força em um ponto específico da viga e 
        monitorando a resposta em outro ponto.

        Parâmetros:
        -----------
        pinput : int
            Posição para aplicação da força na viga.
        poutput : int
            Posição para leitura da aceleração (m/s²) ou velocidade angular.

        Retorno:
        --------
        ww : ndarray
            Vetor de pesos representando o modelo obtido para a viga.
        """
        # Resetando o estado da viga antes da simulação.
        self.beam.reset()
        N = self.N
        ww = np.zeros(N)  # Inicializando os pesos como zero.

        # Modelagem ideal usando a resposta ao impulso.
        if self.type == 0:
            x = np.zeros(N)
            x[0] = 10  # Definindo um impulso inicial.

            # Executa a simulação aplicando o impulso e atualizando a viga.
            for n in range(0, N):
                self.beam.setforce(pinput, x[n])  # Aplicando a força no ponto de entrada.
                self.beam.update()  # Atualizando o estado da viga.
                #wfcb[n] = cbeam.a[pref]
                # wfcb[n] = cbeam.getaccel(pref)/10
                ww[n] = self.movefunc(poutput)  # Obtendo a resposta no ponto de saída.

        # Modelagem adaptativa usando o algoritmo NLMS.
        else:
            NN = self.simtime * int(np.round(1 / self.beam.Ts))
            x = np.random.rand(NN) * 4 - 2  # Gerando entradas aleatórias no intervalo [-2, 2].
            xx = np.zeros(N)

            # Executa a simulação com entrada aleatória e ajusta os pesos via NLMS.
            for n in range(0, NN):
                xx[1:] = xx[:-1]  # Deslocando as amostras anteriores.
                xx[0] = x[n]  # Inserindo a nova amostra.

                self.beam.setforce(pinput, x[n])  # Aplicando a força no ponto de entrada.
                self.beam.update()  # Atualizando o estado da viga.

                # Calculando o erro entre a saída esperada e a saída atual.
                e = self.movefunc(poutput) - ww @ xx

                # Atualizando os pesos com o algoritmo NLMS.
                ww = ww + 0.25 * e * xx / (xx.T @ xx + 1e-3)

        return ww
