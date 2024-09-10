class MaterialSet:
    def __init__(self, name, young, poisson, material=None):
        self.name = name
        self.young = young
        self.poisson = poisson
        self.material = material
        self.density = 0.0
