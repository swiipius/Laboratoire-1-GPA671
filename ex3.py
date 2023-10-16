import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time

class Part1:
    def execute(self):
        # Question 1
        data = self.load()

        # # Question 2
        # self.graph(data)

        # Question 3
        # self.graph_frontiere(data, 0, -1, -1)

        # # Question 4
        # per_0 = Perceptron(0.01)
        # print(self.exactitude(per_0, data))

        # # Question 6
        # grad = per_0.grad_output(data['X'][j], data['D'][j])
        # per_0.update(grad_output=grad)
        # self.graph_front_and_update(data, w0_up=per_0.w[0], w1_up=per_0.w[1], w2_up=per_0.w[2])

        # Question 7
        # per_1 = Perceptron(0.01)
        # exactitude = []
        # for i in range(1000):
        #     exactitude.append(self.exactitude(per_1, data))
        #     for j in range(len(data['X'])):
        #         grad = per_1.grad_output(data['X'][j], data['D'][j])
        #         up = per_1.update(grad_output=grad)
        # Le taux d'exactitude atteint 100% à la 139ème époque
        # plt.plot(exactitude)
        # plt.show()

        # Question 8
        # per_2 = Perceptron(0.01)
        # for _ in range (200): # on a choisit 200 pour etre sur d'atteindre le taux d'exactitude de 100%
        #     for j in range(len(data['X'])):
        #         grad = per_2.grad_output(data['X'][j], data['D'][j])
        #         per_2.update(grad_output=grad)

        # self.graph_front_and_update(data, w0_up=per_2.w[0], w1_up=per_2.w[1], w2_up=per_2.w[2])
        # print(self.exactitude(per_2, data))

        # Question 10
        permut  = np.random.permutation(len(data['X']))
        print("permutation : ", permut)
        data_permut_x = np.array([data['X'][i] for i in permut])
        data_permut_d = np.array([data['D'][i] for i in permut])
        
        per_3 = Perceptron(0.01)
        for i in range (200): # on a choisit 200 pour etre sur d'atteindre le taux d'exactitude de 100% (avec la permutation le breakpoint est a 138)
            for j in range(len(data_permut_x)):
                grad = per_3.grad_output(data_permut_x[j], data_permut_d[j])
                per_3.update(grad_output=grad)
            if self.exactitude_array(per_3, data_permut_x, data_permut_d) == 100:
                print("Taux de 100% (premier index) : ", i)
                break

        self.graph_front_and_update_array(data_permut_x, data_permut_d, w0_up=per_3.w[0], w1_up=per_3.w[1], w2_up=per_3.w[2])
        print("Exactitude : ", self.exactitude_array(per_3, data_permut_x, data_permut_d))
        

    def load(self, path="lab1_1.npz"):
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
    
    def graph_frontiere(self, data: np.ndarray, w0=0, w1=-1, w2=-1) -> None:
        x = data['X'][:, 0]
        y = data['X'][:, 1]
        plt.plot(x, y, 'o', label='Donnees')
        plt.plot(np.linspace(-2, 2), self.frontiere(np.linspace(-2, 2), w0, w1, w2), label='Frontiere')
        plt.legend()
        plt.show()

    def frontiere(self, x, w0, w1, w2):
        return (-w1/w2)*x - (w0/w2)
    
    def graph_front_and_update(self, data: np.ndarray, w0_up, w1_up, w2_up, w0=0, w1=-1, w2=-1) -> None:
        x = data['X'][:, 0]
        y = data['X'][:, 1]

        class1 = False
        class2 = False

        for i in range(len(data['X'])):
            if data['D'][i] == 1.:
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

    def graph_front_and_update_array(self, data_x, data_d, w0_up, w1_up, w2_up, w0=0, w1=-1, w2=-1) -> None:
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

    def exactitude(self, perceptron, data):
        tp, fp, tn, fn = 0, 0, 0, 0

        data_w_bia = np.insert(data['X'], 0, 1, axis=1)

        for index, elem in enumerate(data_w_bia):
            # On a décidé d'inclure le 0.5
            if perceptron.forward(elem) >= 0.5:
                if data['D'][index] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if data['D'][index] == 0:
                    tn += 1
                else:
                    fn += 1
            
        try:
            return (tp + tn) / len(data['X']) * 100
        except ZeroDivisionError:
            return 0
        
    def exactitude_array(self, perceptron, data_x, data_d):
        tp, fp, tn, fn = 0, 0, 0, 0

        data_w_bia = np.insert(data_x, 0, 1, axis=1)

        for index, elem in enumerate(data_w_bia):
            # On a décidé d'inclure le 0.5
            if perceptron.forward(elem) >= 0.5:
                if data_d[index] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if data_d[index] == 0:
                    tn += 1
                else:
                    fn += 1
            
        try:
            return (tp + tn) / len(data_x) * 100
        except ZeroDivisionError:
            return 0

    
class Perceptron:
    
    def __init__(self, lr: float):
        """
        Constructeur de la classe Perceptron.
        
        Parameters
        ----------
        lr : float
            Taux d'apprentissage pour la mise à jour des poids.
        
        """
        self.w = np.array([0, -1, -1], dtype=float) # vecteur de poids   
        self.lr = lr
    
    def forward(self, x: np.ndarray) -> float:
        """
        Propagation directe du perceptron avec sa fonction d'activation $y = F(w^T x)$.
        
        Parameters
        ----------
        x : np.ndarray
            Vecteur d'entrée de longueur 2.
        
        Returns
        -------
        y : np.float
            Sortie du perceptron après la fonction d'activation.
        
        """
        return self.sigmoide(np.dot(self.w, x))
    
    def update(self, grad_output: np.ndarray, x: np.ndarray = None) -> None:
        """
        Mise à jour des poids w.
        
        Parameters
        ----------
        x : np.ndarray
            Vecteur d'entrée de longueur 2.
        grad_output : np.ndarray
            Gradient de l'erreur par rapport à la sortie y perceptron $\nabla_y E$.

        """
        self.w = self.w - self.lr * grad_output

    def sigmoide(self, x):
        return 1 / (1 + math.exp(-x))
    
    def grad_output(self, data, d) -> float:
        # Add bias to data
        data = np.insert(data, 0, 1)

        y_i = self.forward(data)

        return (y_i - d)* y_i * (1 - y_i) * data
    
if __name__ == "__main__":
    Part1().execute()