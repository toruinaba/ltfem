import numpy as np

class Dmatrix:
    # constructor
    # young : young modulus
    # poisson: poisson ratio
    def __init__(self, young, poisson):
        self.young = young
        self.poisson = poisson
    
    def makeDematrix(self):
        tmp = self.young / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        matD = np.array([
                [1.0 - self.poisson, self.poisson, self.poisson, 0.0, 0.0, 0.0],
                [self.poisson, 1.0 - self.poisson, self.poisson, 0.0, 0.0, 0.0],
                [self.poisson, self.poisson, 1.0 - self.poisson, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson)]
            ]
        )
        matD = tmp * matD
        return matD
