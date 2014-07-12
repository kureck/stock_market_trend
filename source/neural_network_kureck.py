#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import data_prepare as dp

def sigmoid(x):
    ''' Função sigmoid representada pela função tangente hiperbólica'''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivada da tangente hiperbólica '''
    return 1.0-x**2

class MLP:
    """ 
        Classe do perceptron multicamada.

        A ativação é dada por: sigmoid(dot(layers,weights))

        :type args: int
        :param args: quantidade de camadas que a rede terá. 

        """

    def __init__(self, *args):
        ''' Inicialização do perceptron com o tamanho das camadas.  '''

        self.shape = args
        # args é a quantidade de camadas que a rede terá. Por exemplo: se
        # a rede tiver dois perceptrons de entrada, dois na camada escondida e uma na saída
        # a MLP deverá ser iniciada como MLP(2,2,1)

        n = len(args)

        # Cria as camadas
        self.layers = []
        
        # Construção da camada de entrada (+1 para o bias)
        self.layers.append(np.ones(self.shape[0]+1))
        
        # Camadas de entrada e camada de saída
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Matriz de pesos aleatória entre -0.25 e +0.25
        # TODO: dá pra encontrar um método melhor para iniciar os pesos
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # armazena os últimos pesos alterados
        self.dw = [0,]*len(self.weights)

        # Reinicia os pesos
        self.reset()

    def reset(self):
        ''' Reinicia os pesos '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propaga os dados da camada de entrada para a camada de saída. '''

        # Entra com os dados na camada de entrada
        self.layers[0][0:-1] = data
        # print "data", data
        # print "self.layers\n", self.layers
        # print "self.layers[0][0:-1]\n", self.layers[0][0:-1]
        # print "self.weights\n", self.weights
        # print "\n"

        # Propaga as mudanças dos pesos da camada 0 ate n-1 usando a sigmoid como 
        # função de ativação
        for i in range(1,len(self.shape)):
            # print "self.layers[i][...]"
            # print self.layers[i][...]
            # print "\n"
            # print "sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))"
            # print sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))
            # print "\n"
            # print "---"
            # print self.layers[i-1]
            # print self.weights[i-1]
            ## Propagação
            # A camada i recebe os pesos atualizados da camada i-1 através 
            # da ativação sigmoid(dot(layers,weights))
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # print self.layers
        # Retorna a saída
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Retropropaga o erro. '''

        deltas = []

        # Calcula o erro na camada de saída
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Calcula o erro nas camadas escondidas
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)

        # Atualiza os pesos
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Retorna erro
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    def learn(network,samples, epochs=2500, lrate=.1, momentum=0.1):
        # Treino 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            # print "n ", n
            # print "samples['input'][n] ", samples['input'][n]
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )
            # raw_input()
        # Teste
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print i, samples['input'][i], ' ', o,
            print '(esperado)', samples['output'][i]
        print

    # # Teste 0 : Bolsas selecionadas
    # -------------------------------------------------------------------------
    file = '../data/all_closes_percentage.csv'
    st1 = 'americas_bvsp'
    st2 = 'americas_gsptse'
    st3 = ['americas_ipsa',"americas_merv","americas_mxx","asia_000001ss","asia_aord","asia_axjo","asia_bsesn","asia_hsi","asia_jkse"]
    stock = [st1,st2]
    data_prepare = dp.DataPrepare(file)
    samples = data_prepare.prepare_data()
    network = MLP(len(stock),3,3,3) 
    learn(network, samples, 100, 0.1)

    # # Teste 1 : Apenas IBOVESPA
    # -------------------------------------------------------------------------
    # print "Aprendizado da bolsa: apenas IBOVESPA"
    # samples = np.zeros(4, dtype=[('input',  float, 2), ('output', list, 1)])
    # network = MLP(2,3,3,3)
    # samples[0] = (53635., 15207.1), [0,0,1]
    # samples[1] = (53802., 15215.0), [0,0,1]
    # samples[2] = (54055., 15172.9), [1,0,0]
    # samples[3] = (53875., 15137.2), [1,0,0]
    # learn(network, samples, 100, 0.1)

    # Teste 2 : Função lógica OR
    # -------------------------------------------------------------------------
    # print "Aprendizado da função lógica OR"
    # network.reset()
    # samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])
    # network = MLP(2,3,3,1)
    # samples[0] = (0,0), 0
    # samples[1] = (1,0), 0
    # samples[2] = (0,1), 0
    # samples[3] = (1,1), 1
    # learn(network, samples)


