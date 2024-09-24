import pandas as pd
from numpy import linspace

class ActVibData(pd.DataFrame):
    """
    Classe para manipulação de dados de vibração e sensores.
    
    Leitura dos dados de arquivos CSV ou Feather e fornecimento de métodos para acessar
    e processar dados de séries temporais provenientes de sensores como IMUs e ADCs. 
    Assim como gerenciamento de logs e notas, se disponíveis.
    """

    def __init__(self, filename):    
        """
        Inicializa a classe ActVibData carregando um arquivo no formato CSV ou Feather.
        
        Parâmetros:
        - filename: str
            Caminho para o arquivo a ser carregado. O arquivo pode estar em formato CSV 
            (separado por tabulação) ou Feather.
        """
        if str(filename).endswith(".csv"):
            super().__init__(pd.read_csv(filename, index_col=0, sep="\t"))
        else:    
            super().__init__(pd.read_feather(filename))
        
        self.filename = filename
        
        # Verifica se há uma coluna de tempo e padroniza os nomes das colunas
        if "Tempo (s)" in self.columns:
            self.oldcnamestonew()
        
        # Verifica a presença de logs nos dados
        if "log" in self.columns:
            self.hasLog = True
        else:
            self.hasLog = False
        
    def oldcnamestonew(self):
        """
        Padroniza os nomes das colunas antigas para novos, mais usáveis.
        
        Exemplos de transformações:
        - "Tempo (s)" -> "time"
        - Colunas de "IMU" -> letras minúsculas ("imu1accx", etc.)
        - "Perturbacao" -> "perturb"
        - "Controle" -> "ctrl"
        """
        cnames = self.columns
        newcnames = []
        for k in range(len(cnames)):
            if cnames[k].startswith("Tempo"):
                newcnames.append("time")
            elif cnames[k].startswith("IMU") or (cnames[k] == "Log"):
                newcnames.append(cnames[k].lower())
            elif cnames[k].startswith("DAC") or cnames[k].startswith("ADC"):
                newcnames.append(cnames[k].replace(" ","").lower())
            elif cnames[k] == "Perturbacao":
                newcnames.append("perturb")
            elif cnames[k] == "Controle":
                newcnames.append("ctrl")
            elif cnames[k] == "Referencia":
                newcnames.append("ref")
            elif cnames[k] == "Erro": 
                newcnames.append("err")
        self.columns = newcnames 

    def getTime(self):
        """
        Retorno:
        - Valores de tempo como um array numpy.
        """
        return self.time.values

    def getSignalNames(self):
        """
        Retorno:
        - Lista de nomes das colunas.
        """
        return list(self.columns)

    def getSignal(self, signalname):
        """ 
        Parâmetros:
        - signalname: str
            Nome do sinal (coluna) a ser retornado.

        Retorno:
        - Array numpy com os valores do sinal.
        """
        return self[signalname].values
    
    def getAccX(self, imuidx=1):
        """
        Parâmetros:
        - imuidx: int, opcional (padrão=1)
            Índice da IMU (por exemplo, 1 para "imu1accx").
        
        Retorno:
        - Array numpy com os dados de aceleração no eixo X.
        """
        return self[f"imu{imuidx}accx"].values
    
    def getAccY(self, imuidx=1):
        """ 
        Parâmetros:
        - imuidx: int, opcional (padrão=1)
            Índice da IMU (por exemplo, 1 para "imu1accy").
        
        Retorno:
        - Array numpy com os dados de aceleração no eixo Y.
        """
        return self[f"imu{imuidx}accy"].values

    def getAccZ(self, imuidx=1):
        """
        Parâmetros:
        - imuidx: int, opcional (padrão=1)
            Índice da IMU (por exemplo, 1 para "imu1accz").
        
        Retorno:
        - Array numpy com os dados de aceleração no eixo Z.
        """
        return self[f"imu{imuidx}accz"].values

    def getGyroX(self, imuidx=1):
        """
        Parâmetros:
        - imuidx: int, opcional (padrão=1)
            Índice da IMU (por exemplo, 1 para "imu1gyrox").
        
        Retorno:
        - Array numpy com os dados do giroscópio no eixo X.
        """
        return self[f"imu{imuidx}gyrox"].values
    
    def getGyroY(self, imuidx=1):
        """
        Parâmetros:
        - imuidx: int, opcional (padrão=1)
            Índice da IMU (por exemplo, 1 para "imu1gyroy").
        
        Retorno:
        - Array numpy com os dados do giroscópio no eixo Y.
        """
        return self[f"imu{imuidx}gyroy"].values

    def getGyroZ(self, imuidx=1):
        """
        Parâmetros:
        - imuidx: int, opcional (padrão=1)
            Índice da IMU (por exemplo, 1 para "imu1gyroz").
        
        Retorno:
        - Array numpy com os dados do giroscópio no eixo Z.
        """
        return self[f"imu{imuidx}gyroz"].values

    def getADCData(self, adcid=1):
        """
        Parâmetros:
        - adcid: int, opcional (padrão=1)
            Índice do canal ADC (1-4).
        
        Retorno:
        - Array numpy com os dados do ADC.

        Erros:
        - BaseException se o canal ADC não estiver entre 1 e 4 ou se os dados do ADC não forem encontrados.
        """
        if (adcid < 1) or (adcid > 4):
            raise BaseException("ADCid deve estar entre 1 e 4.")
        adccols = list(filter(lambda x: x.startswith("adc"), self.getSignalNames()))
        if len(adccols) == 0:
            raise BaseException("Dados do ADC não encontrados.")
        return self[adccols[adcid-1]].values
    
    def getadc1k(self):
        """
        Retornos:
        - timevec: array numpy
            Vetor de tempo com amostragem de 1 kHz.
        - dt: array numpy
            Valores reorganizados dos dados do ADC.
        
        Erros:
        - BaseException se não forem encontrados dados do ADC.
        """
        adccols = list(filter(lambda x: x.startswith("adc"), self.getSignalNames()))
        if len(adccols) == 0:
            raise BaseException("Dados do ADC não encontrados.")
        dt = self[adccols].values.reshape((self.shape[0]*4))
        timevec = linspace(0, self["time"].values[-1]+3e-3, num=dt.shape[0])
        return timevec, dt

    def getADC1kHzData(self):
        """
        Retorno:
        - Array numpy com os dados do ADC reorganizados.
        
        Erros:
        - BaseException se não forem encontrados dados do ADC.
        """
        adccols = list(filter(lambda x: x.startswith("adc"), self.getSignalNames()))
        if len(adccols) == 0:
            raise BaseException("Dados do ADC não encontrados.")
        return self[adccols].values.reshape((self.shape[0]*4))

    def getNotes(self):
        """
        Retorno:
        - str: Notas da coluna 'log'.
        
        Erros:
        - BaseException se logs não forem encontrados ou as notas estiverem ausentes.
       """
