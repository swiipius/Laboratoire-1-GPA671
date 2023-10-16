import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class Execrcice:
    tp, tn = 0, 0

    def execute(self):
        data = self.load()

        self.question_4(data)

    def question_1(self, data):
        self.graph(data)

    def question_2(self, data):
        layer = Layer(2, 1, 0.01)
        print(self.exactitude(layer, data))

    def question_4(self, data):
        layer = Layer(2, 1, 0.01)
        w0_init, w1_init, w2_init = layer.W[0][0], layer.W[0][1], layer.W[0][2]

        vec_forward = np.vectorize(layer.forward, signature='(n)->()')

        data_w_bia = np.insert(data['X'], 0, 1., axis=1)
        forward_out = vec_forward(data_w_bia)

        grad_input_temp = np.dot(forward_out - data['D'], np.dot(forward_out, (1 - forward_out)))
        print(grad_input_temp)
        grad_input = []
        # for index, elem in enumerate(grad_input_temp):
        #     print(elem)
            # grad_input.append([ data_w_bia[index][0] * elem, data_w_bia[index][1] * data_w_bia[index][2] * elem ])

        # print(np.array(grad_input))

        # for _ in range(200):
        #     grad_input = layer.update(data['X'], grad_input, data['D'])

        # print(w0_init, w1_init, w2_init)
        # print(layer.W[0][0], layer.W[0][1], layer.W[0][2])





    def load(self, path="lab1_2.npz"):
        return np.load(path)
    
    def graph(self, data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        class1 = False
        class2 = False

        for i in range(len(data['X'])):
            x = data['X'][i][0]
            y = data['X'][i][1]
            if data['D'][i] == 1:
                marker = 'o'
                color = 'b'

                if not class1:
                    Axes3D.scatter(ax, x, y, marker=marker, color=color, label='Class 1')
                else:
                    Axes3D.scatter(ax, x, y, marker=marker, color=color)

                class1 = True
            else:
                marker = '^'
                color = 'r'

                if not class2:
                    Axes3D.scatter(ax, x, y, marker=marker, color=color, label='Class 2')
                else:
                    Axes3D.scatter(ax, x, y, marker=marker, color=color)

                class2 = True

        plt.legend()
        plt.show()

    def exactitude(self, layer, data):
        tp, fp, tn, fn = 0, 0, 0, 0

        data_w_bia = np.insert(data['X'], 0, 1., axis=1)

        vec_forward = np.vectorize(layer.forward, signature='(n)->()')

        forward_out = vec_forward(data_w_bia)

        vec_check = np.vectorize(self.check_tp_tn, excluded=['data_d'], otypes=[None])

        vec_check(forward_result=forward_out, data_d=data['D'])      
        
        return (self.tp + self.tn) / len(data['X']) * 100

    index = 0

    def check_tp_tn(self, forward_result, data_d):
        if forward_result >= 0.5:
            if data_d[self.index][0] == 1:
                self.tp += 1
        else:
            if data_d[self.index][0] == 0:
                self.tn += 1

        self.index += 1

    def graph_front_and_update_array(self, data_x, data_d, w0_up, w1_up, w2_up, w0, w1, w2) -> None:
        x = data_x[:, 0]
        y = data_x[:, 1]

        class1 = False
        class2 = False

        for i in range(len(data_x)):
            if data_d[i] == 1.:
                if not class1:
                    plt.plot(x[i], y[i], 'o', color='b', label='Class 1')
                    class1 = True
                else:
                    plt.plot(x[i], y[i], 'o', color='b')
            else:
                if not class2:
                    plt.plot(x[i], y[i], 'o', color='r', label='Class 0')
                    class2 = True
                else:
                    plt.plot(x[i], y[i], 'o', color='r')
        plt.plot(np.linspace(-2, 2), self.frontiere(np.linspace(-2, 2), w0, w1, w2), label='Frontiere initiale')
        plt.plot(np.linspace(-2, 2), self.frontiere(np.linspace(-2, 2), w0_up, w1_up, w2_up), label='Frontiere mise a jour')
        plt.legend()
        plt.show() 

class Layer:
    
    def __init__(self, n_in: int, n_out: int, lr: float):
        """
        Constructeur de la classe Layer.
        
        Parameters
        ----------
        n_int : int
            Nombre d'entrées.
        n_out : int
            Nombre de sorties.
        lr : float
            Taux d'apprentissage pour la mise à jour des poids.
        
        """
        bound = (6 / (n_in + n_out)) ** 0.5  # initialisation selon Glorot & Bengio (2010)
        # matrice de poids (+1 pour le biais)
        self.W = np.random.uniform(low=-bound, high=bound, size=(n_out, n_in + 1)) 
        self.W[:, 0] = 0  # biais initialisé à 0
        self.lr = lr
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagation directe de la couche avec sa fonction d'activation $Y = F(W^T X)$.
        
        Parameters
        ----------
        X : np.ndarray
            Données d'entrées. Taille : (n_in, # exemples).
        
        Returns
        -------
        Y : np.ndarray
            Sorties de la couche après la fonction d'activation. Taille : (n_out, # exemples)

        """
        return self.sigmoide(np.dot(self.W, X))

    def sigmoide(self, x):
        return 1 / (1 + math.exp(-x))
        
    def update(self, X: np.ndarray, grad_output: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Mise à jour des poids w et retourne le gradient de l'erreur par rapport à l'entrée de la couche $\nabla_X E$.
        
        Parameters
        ----------
        X : np.ndarray
            Données d'entrées. Taille : (n_in, # exemples).
        grad_output : np.ndarray.
            Gradient de l'erreur par rapport à la sortie Y de la couche $\nabla_Y E$. Taille : (n_out, # exemples)
            
        Returns
        -------
        grad_input : np.ndarray
            Gradient de l'erreur par rapport à l'entrée X de la couche $\nabla_X E$. Taille : (n_in, # exemples)

        """
        vec_forward = np.vectorize(self.forward, signature='(n)->()')

        data_w_bia = np.insert(X, 0, 1., axis=1)
        forward_out = vec_forward(data_w_bia)

        self.W = self.W - np.dot(self.lr, grad_output)

        return np.dot(forward_out - d[:][0], np.dot(forward_out, (1 - forward_out)))

if __name__ == '__main__':
    ex = Execrcice()
    ex.execute()