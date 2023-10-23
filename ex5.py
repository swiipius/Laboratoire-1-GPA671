import numpy as np
import math

def question_1(data):
    MLP([Layer(2, 2, 0.01), Layer(2, 1, 0.01)])

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

        for i in range(len(grad_output)):
                self.W = self.W - np.dot(self.lr, grad_output[i])

        return np.dot(forward_out - d[:][0], np.dot(forward_out, (1 - forward_out)))

class MLP:
    def __init__(self, list_of_layers: list):
        """
        Classe pour implémenter un réseau de neurones multicouches.

        Parameters
        ----------
        list_of_layers : list
            Liste ordonnée des couches successives du réseau de neurones.
        """
        self.list_of_layers = list_of_layers

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagation directe à travers les couches du MLP.
        
        Parameters
        ----------
        X : np.ndarray
            Données d'entrées.
        
        Returns
        -------
        Y : np.ndarray
            Sorties après propagation directe.

        """
        self.list_of_layers[0].forward(X)
        
    def update(self, X: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Mise à jour des poids des couches successives du MLP. Renvoie le gradient de l'erreur par 
        rapport à l'entrée du MLP.
        
        Parameters
        ----------
        X : np.ndarray
            Données d'entrées.
        grad_output : np.ndarray
            Gradient de l'erreur par rapport à la sortie Y du MLP $\nabla_Y E$.
            
        Returns
        -------
        grad_input : np.ndarray
            Gradient de l'erreur par rapport à l'entrée X de la couche $\nabla_X E$.

        """
        pass

if __name__ == "__main__":
    data = np.load("lab1_2.npz")

    question_1(data)
