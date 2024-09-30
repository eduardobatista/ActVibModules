# CantileverBeam Module

import numpy as np
from scipy import linalg

class CantileverBeam:
    '''
        Classe similar à classe Viga, mas com saídas em aceleração ao invés de deslocamento.
    '''
    # Parâmetros gerais:
    m = 1  # massa - sempre 1 para todas as vigas

    def __init__(self,npoints=60,width=0.05,thickness=0.00575,length=0.58,density=7900,
                    elasticmod=2e11,Tsampling=0.004,nmodes=5,
                    damp=[0.002, 0.002, 0.001, 0.001, 0.001],
                    forcescaler=1,noisestd=0):
        # Inicialização dos parâmetros da viga
        self.Ts = Tsampling
        self.npoints = npoints
        self.nmodes = nmodes
        self.width = width
        self.thickness = thickness
        self.length = length        
        self.density = density
        self.elasticmod = elasticmod
        self.forcescaler = forcescaler  # Permite definir um "scaler" para força, simulando por exemplo uma conversão de Volts para Newton caso a força seja gerada a partir de uma tensão elétrica. 
        self.forcescaler1 = forcescaler
        self.deltax = self.length / (self.npoints - 1)  # Discretização espacial ao longo da viga (Checar isso)
        self.rotvelmultiplier = (1 / self.deltax) * 180 / np.pi  # Conversão para velocidade rotacional
        self.magnetdist = 1e-3  # Distância inicial do ímã
        self.evaluateModesAndFreqs()  # Avalia modos e frequências naturais
        self.memiir = 3  # Ordem do filtro IIR
        self.Fs = 1 / self.Ts  # Frequência de amostragem
        self.wn = 2 * np.pi * self.freqsHz  # Conversão das frequências para radianos/segundo
        self.zeta = np.array(damp)  # Fatores de amortecimento
        self.wd = self.wn * np.sqrt(1-self.zeta[:self.nmodes]**2)  # Frequências amortecidas
        self.Aiir = np.zeros((self.nmodes,self.memiir-1))   # Coefs dos denominadores dos IIRs de cada modo
        self.Biir = np.zeros((self.nmodes,self.memiir))  #  Coefs dos numeradores dos IIRs de cada modo      
        # Cálculo dos coeficientes dos filtros IIR
        for k in range(0,self.nmodes):            
            self.Biir[k,0] = 0
            self.Biir[k,1] = np.exp(-self.zeta[k]*self.wn[k]*self.Ts) * np.sin(self.wd[k]*self.Ts)
            self.Biir[k,2] = 0
            self.Aiir[k,0] = -2 * np.exp(-self.zeta[k]*self.wn[k]*self.Ts) * np.cos(self.wd[k]*self.Ts)
            self.Aiir[k,1] = np.exp(-2*self.zeta[k]*self.wn[k]*self.Ts)        
        self.reset()  # Reseta as condições da viga
        self.noisestd = noisestd  # Desvio padrão do ruído
        self.setaccelg(False)  # Define o método de cálculo de aceleração como m/s²

    
    def getModeShapes(self):
        """
            Retorna os dados para plotar as formas modais.

            Returns:
                x: Vetor contendo as posições ao longo da viga.
                m: Formas modais (m[:,0] para o primeiro modo, m[:,1] para o segundo, etc)
        """
        x = np.linspace(0,self.length,num=self.npoints)  # Posições ao longo da viga
        return x,self.vmod  # Retorna posições e formas modais


    def evaluateModesAndFreqs(self):
        # Calcula os modos de vibração e as frequências naturais da viga
        I = (self.width * self.thickness**3) / 12  # Momento de inércia
        beam_mass = self.density * self.width * self.thickness * self.length  # Massa da viga
        # Matriz de massa:
        M = np.matrix( np.eye(self.npoints) * (beam_mass/self.npoints) )
        M[0,0] = M[0,0]/2  # Ajusta a primeira massa (extremidade fixa)
        # Matriz de rigidez:
        A = np.zeros((self.npoints,self.npoints))
        for r in range(self.npoints):
            for c in range(r+1):
                k = self.npoints-r
                l = self.npoints-c
                A[r,c] = ((self.deltax**3)/(6*self.elasticmod*I)) * (3* k**2 * l - k**3)
                A[c,r] = A[r,c]  # Simetria da matriz de rigidez
        K = linalg.inv(A)  # Matriz de rigidez invertida
        L = linalg.cholesky(M)  # Decomposição de Cholesky da matriz de massa
        Kt = linalg.inv(L.T) @ K @ linalg.inv(L)  # Matriz transformada para o sistema de modos
        ww,P = linalg.eig(Kt)  # Calcula autovalores e autovetores
        ww = np.real(ww)  # Frequências naturais (sem partes imaginárias)
        sidxs = np.argsort(ww)  # Ordena as frequências
        U = linalg.inv(L) @ P  # Modos normalizados
        self.vmod = np.zeros((self.npoints,self.nmodes))  # Inicializa as formas modais
        self.freqsHz = np.zeros(self.nmodes)  # Inicializa as frequências
        for k in range(self.nmodes):
            self.vmod[:,k] = U[::-1,sidxs[k]]  # Formas modais
            self.freqsHz[k] = np.sqrt(ww[sidxs[k]])/2/np.pi  # Conversão para Hz

    def configforcescaler(self,forcescl,magnetdist=1e-3):
        # Configura o escalador de força baseado na distância do ímã
        self.forcescaler1 = forcescl
        self.forcescaler = forcescl / ( (magnetdist * 1000) ** 2 )
        self.magnetdist = magnetdist

    def setforce(self,pos,val):
        # Define a força aplicada em uma posição específica da viga
        self.f[pos] = self.forcescaler * val

    def setforcenl(self,pos,val):
        # Define uma força não linear aplicada em uma posição específica da viga
        self.f[pos] = self.forcescaler1 * val / ( (( self.magnetdist + self.x[pos] ) * 1000) ** 2 )

    def setaccelg(self,val):
        # Alterna entre o cálculo da aceleração em m/s² e g
        if val: 
            self.getaccel = self.getaccelg
        else:
            self.getaccel = self.getaccelms2

    def getaccelms2(self,pos):
        # Retorna a aceleração em m/s² com ruído
        return self.a[pos] + np.random.randn()*self.noisestd

    def getaccelg(self,pos):
        # Retorna a aceleração em g com ruído
        return (self.a[pos] + np.random.randn()*self.noisestd)/9.80665
    
    def getrotationvel(self,pos):
        # Retorna a velocidade rotacional com ruído
        return self.rotvel[pos] + np.random.randn()*self.noisestd

    def reset(self):
        # Reseta as forças, deslocamentos, acelerações e buffers da viga
        self.f = np.zeros(self.npoints)
        self.x = np.zeros(self.npoints)
        self.a = np.zeros(self.npoints)
        self.xiir = np.zeros((self.npoints,self.memiir))
        self.yiir = np.zeros((self.npoints,self.memiir))
        self.bufdesloc = np.zeros(self.npoints)
        self.bufvel = np.zeros((self.npoints,2))
        self.rotvel = np.zeros(self.npoints)  # Trying to implement rotation velocity, in degrees per second.

    def update(self):
        # Atualiza o estado da viga, incluindo deslocamento, velocidade e aceleração
        self.bufvel[:,1] = self.bufvel[:,0]  # Armazena a velocidade anterior
        self.bufdesloc[:] = self.x  # Armazena o deslocamento anterior
        self.x = np.zeros(self.npoints)  # Reseta os deslocamentos
        for k in range(0,self.nmodes):
            self.xiir[k,1:self.memiir] = self.xiir[k,0:self.memiir-1]  # Desloca os valores do filtro IIR
            self.xiir[k,0] = self.vmod