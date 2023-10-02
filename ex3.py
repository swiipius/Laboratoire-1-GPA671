import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class Part1:
    def execute(self):
        data = self.load()
        # self.graph(data)
        # self.graph_frontiere(data)

        per = Perceptron(0.01)
        out = per.grad_output(data['X'][0], data['D'][0])
        print(out)

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
    
    def graph_frontiere(self, data: np.ndarray) -> None:
        x = data['X'][:, 0]
        y = data['X'][:, 1]
        plt.plot(x, y, 'o')
        plt.plot(self.frontiere(np.linspace(-2, 2)), np.linspace(-2, 2))
        plt.show()

    def frontiere(self, x, w0=0, w1=-1, w2=-1):
        return (-w1/w2)*x - (w0/w2)
    
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
    
    def update(self, x: np.ndarray, grad_output: float) -> None:
        """
        Mise à jour des poids w.
        
        Parameters
        ----------
        x : np.ndarray
            Vecteur d'entrée de longueur 2.
        grad_output : np.float
            Gradient de l'erreur par rapport à la sortie y perceptron $\nabla_y E$.

        """
        self.w = self.w - self.lr * grad_output

    def sigmoide(self, x):
        return 1 / (1 + math.exp(-x))
    
    def grad_output(self, data, d) -> float:
        grad = []

        temp = 0
        for index, w_i in enumerate(self.w):
            try:
                y_j = self.sigmoide(np.dot(data[index], w_i))
            except:
                y_j = self.sigmoide(np.dot(1, w_i))
            temp += y_j
        grad.append((temp - d)* temp * (1 - temp) * data)       

        return np.array(grad)[0]
    
if __name__ == "__main__":
    Part1().execute()