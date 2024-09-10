from enum import Enum
import numpy as np


class Tensor_type(Enum):
    STRESS = 0
    STRAIN = 1

class Tensor_base:
    DIM = 0

    @property
    def vector(self):
        raise NotImplementedError

    @property
    def tensor(self):
        raise NotImplementedError

    @property
    def hydropressure(self):
        raise NotImplementedError

    @property
    def deviatric(self):
        raise NotImplementedError

    @classmethod
    def from_vector(cls):
        raise NotImplementedError

    @classmethod
    def from_matrix(cls):
        raise NotImplementedError

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__.from_vector(-self.vector, self.type)
    
    def __abs__(self):
        return self.__class__.from_vector(abs(self.vector), self.type)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            vector = self.vector + other.vector
            is_deviatric = self.is_deviatric or other.is_deviatric
            return self.__class__.from_vector(vector, self.type, is_deviatric)
        
        if isinstance(other, Tensor_base):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            if self.DIM == other.DIM:
                vector = self.vector + other.vector
                is_deviatric = self.is_deviatric or other.is_deviatric
                return self.__class__.from_vector(vector, self.type, is_deviatric)
            raise ValueError("Not match dimension.")

        if not isinstance(other, np.ndarray):
            arr = np.array(other)
        else:
            arr = other
        if arr.ndim == 1:
            vector = self.vector + arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)
        elif arr.ndim >= 2:
            tensor = self.tensor + arr
            if self.DIM == arr.ndim:
                return self.__class__.from_matrix(tensor, self.type, self.is_deviatric)
            return tensor
        else:
            vector = self.vector + arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            vector = self.vector - other.vector
            is_deviatric = self.is_deviatric or other.is_deviatric
            return self.__class__.from_vector(vector, self.type, is_deviatric)
        
        if isinstance(other, Tensor_base):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            if self.DIM == other.DIM:
                vector = self.vector - other.vector
                is_deviatric = self.is_deviatric or other.is_deviatric
                return self.__class__.from_vector(vector, self.type, is_deviatric)
            raise ValueError("Not match dimension.")

        if not isinstance(other, np.ndarray):
            arr = np.array(other)
        else:
            arr = other
        if arr.ndim == 1:
            vector = self.vector - arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)
        elif arr.ndim >= 2:
            tensor = self.tensor - arr
            if self.DIM == arr.ndim:
                return self.__class__.from_matrix(tensor, self.type, self.is_deviatric)
            return tensor
        else:
            vector = self.vector - arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            vector = self.vector * other.vector
            is_deviatric = self.is_deviatric or other.is_deviatric
            return self.__class__.from_vector(vector, self.type, is_deviatric)
        
        if isinstance(other, Tensor_base):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            if self.DIM == other.DIM:
                vector = self.vector * other.vector
                is_deviatric = self.is_deviatric or other.is_deviatric
                return self.__class__.from_vector(vector, self.type, is_deviatric)
            raise ValueError("Not match dimension.")

        if not isinstance(other, np.ndarray):
            arr = np.array(other)
        else:
            arr = other
        if arr.ndim == 1:
            vector = self.vector * arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)
        elif arr.ndim >= 2:
            tensor = self.tensor * arr
            if self.DIM == arr.ndim:
                return self.__class__.from_matrix(tensor, self.type, self.is_deviatric)
            return tensor
        else:
            vector = self.vector * arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            vector = self.vector / other.vector
            is_deviatric = self.is_deviatric or other.is_deviatric
            return self.__class__.from_vector(vector, self.type, is_deviatric)
        
        if isinstance(other, Tensor_base):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            if self.DIM == other.DIM:
                vector = self.vector / other.vector
                is_deviatric = self.is_deviatric or other.is_deviatric
                return self.__class__.from_vector(vector, self.type, is_deviatric)
            raise ValueError("Not match dimension.")

        if not isinstance(other, np.ndarray):
            arr = np.array(other)
        else:
            arr = other
        if arr.ndim == 1:
            vector = self.vector / arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)
        elif arr.ndim >= 2:
            tensor = self.tensor / arr
            if self.DIM == arr.ndim:
                return self.__class__.from_matrix(tensor, self.type, self.is_deviatric)
            return tensor
        else:
            vector = self.vector / arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            vector = other.vector / self.vector
            is_deviatric = self.is_deviatric or other.is_deviatric
            return self.__class__.from_vector(vector, self.type, is_deviatric)
        
        if isinstance(other, Tensor_base):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            if self.DIM == other.DIM:
                vector = other.vector / self.vector
                is_deviatric = self.is_deviatric or other.is_deviatric
                return self.__class__.from_vector(vector, self.type, is_deviatric)
            raise ValueError("Not match dimension.")

        if not isinstance(other, np.ndarray):
            arr = np.array(other)
        else:
            arr = other
        if arr.ndim == 1:
            vector = arr / self.vector
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)
        elif arr.ndim >= 2:
            tensor = arr / self.tensor
            if self.DIM == arr.ndim:
                return self.__class__.from_matrix(tensor, self.type, self.is_deviatric)
            return tensor
        else:
            vector = arr / self.vector
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            vector = self.vector @ other.vector
            is_deviatric = self.is_deviatric or other.is_deviatric
            return self.__class__.from_vector(vector, self.type, is_deviatric)
        
        if isinstance(other, Tensor_base):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            if self.DIM == other.DIM:
                vector = self.vector @ other.vector
                is_deviatric = self.is_deviatric or other.is_deviatric
                return self.__class__.from_vector(vector, self.type, is_deviatric)
            raise ValueError("Not match dimension.")

        if not isinstance(other, np.ndarray):
            arr = np.array(other)
        else:
            arr = other
        if arr.ndim == 1:
            vector = self.vector @ arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)
        elif arr.ndim >= 2:
            tensor = self.tensor @ arr
            if self.DIM == arr.ndim:
                return self.__class__.from_matrix(tensor, self.type, self.is_deviatric)
            return tensor
        else:
            vector = self.vector @ arr
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)

    def __rmatmul__(self, other):
        if isinstance(other, self.__class__):
            if self.type != other.type:
                raise ValueError(f"Not matched tensor type. self:{self.type}, other: {other.type}")
            vector = other.vector @ self.vector
            is_deviatric = self.is_deviatric or other.is_deviatric
            return self.__class__.from_vector(vector, self.type, is_deviatric)
        
        if isinstance(other, Tensor_base):
            if self.DIM == other.DIM:
                vector = other.vector @ self.vector
                is_deviatric = self.is_deviatric or other.is_deviatric
                return self.__class__.from_vector(vector, self.type, is_deviatric)
            raise ValueError("Not match dimension.")

        if not isinstance(other, np.ndarray):
            arr = np.array(other)
        else:
            arr = other
        if arr.ndim == 1:
            vector = arr @ self.vector
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)
        elif arr.ndim >= 2:
            tensor = arr @ self.tensor
            if self.DIM == arr.ndim:
                return self.__class__.from_matrix(tensor, self.type, self.is_deviatric)
            return tensor
        else:
            vector = arr @ self.vector
            return self.__class__.from_vector(vector, self.type, self.is_deviatric)


class Tensor3d(Tensor_base):
    DIM = 3

    def __init__(
        self,
        xx: float,
        yy: float,
        zz: float,
        yz: float,
        xz: float,
        xy: float,
        type: Tensor_type,
        is_deviatric: bool=False
    ):
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.xy = xy
        self.yz = yz
        self.xz = xz
        self.type = type
        self.is_deviatric = is_deviatric


    @property
    def vector(self) -> np.ndarray:
        return np.array(
            [self.xx, self.yy, self.zz, self.yz, self.xz, self.xy]
        )

    @property
    def tensor(self) -> np.ndarray:
        if self.type == Tensor_type.STRESS:
            return np.array(
                [
                    [self.xx, self.xy, self.xz],
                    [self.xy, self.yy, self.yz],
                    [self.xz, self.yz, self.zz]
                ]
            )
        elif self.type == Tensor_type.STRAIN:
            return np.array(
                [
                    [self.xx, 0.5 * self.xy, 0.5 * self.xz],
                    [0.5 * self.xy, self.yy, 0.5 * self.yz],
                    [0.5 * self.xz, 0.5 * self.yz, self.zz]
                ]
            )
        raise NotImplementedError

    @property
    def J1(self) -> float:
        return self.xx + self.yy + self.zz

    @property
    def hydropressure(self) -> float:
        if self.is_deviatric:
            print("Warning: This tensor is already deviatric stress or strain.")
            return 0.0
        return 1 / 3 * self.J1

    @property
    def J2(self) -> float:
        return (
            (self.xx - self.yy)**2 +
            (self.yy - self.zz)**2 +
            (self.zz - self.xx)**2 +
            6 * (self.yz**2 + self.xz**2 + self.xy**2)
        )

    @property
    def norm(self) -> float:
        if self.type == Tensor_type.STRESS:
            return np.sqrt(1 / 3 * self.J2)
        raise NotImplementedError 

    @property
    def mises(self) -> float:
        if self.type == Tensor_type.STRESS:
            return np.sqrt(0.5 * self.J2)
        elif self.type == Tensor_type.STRAIN:
            return np.sqrt(4 / 9 * self.J2)
        raise NotImplementedError

    @property
    def deviatric(self):
        if self.is_deviatric:
            print("Warning: this tensor is already deviatric stress or strain. return self")
            return self
        return self.__class__.from_matrix(self.tensor - self.hydropressure * np.eye(self.DIM), self.type, True)

    @classmethod
    def from_vector(cls, vector: np.ndarray,  tensor_type: Tensor_type, is_deviatric: bool=False):
        return cls(
            vector[0],
            vector[1],
            vector[2],
            vector[3],
            vector[4],
            vector[5],
            tensor_type,
            is_deviatric
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, tensor_type: Tensor_type, is_deviatric: bool=False):
        if tensor_type == Tensor_type.STRESS:
            return cls(
                matrix[0, 0],
                matrix[1, 1],
                matrix[2, 2],
                matrix[1, 2],
                matrix[0, 2],
                matrix[0, 1],
                Tensor_type.STRESS,
                is_deviatric
            )
        elif tensor_type == Tensor_type.STRAIN:
            return cls(
                matrix[0, 0],
                matrix[1, 1],
                matrix[2, 2],
                2.0 * matrix[1, 2],
                2.0 * matrix[0, 2],
                2.0 * matrix[0, 1],
                Tensor_type.STRAIN,
                is_deviatric
            )
        raise NotImplementedError

    def to_stress(self, factor=1.0):
        if self.type == Tensor_type.STRESS:
            return self * factor
        elif self.type == Tensor_type.STRAIN:
            return self.__class__(
                self.xx * factor,
                self.yy * factor,
                self.zz * factor,
                0.5 * self.yz * factor,
                0.5 * self.xz * factor,
                0.5 * self.xy * factor,
                Tensor_type.STRESS,
                self.is_deviatric
            )

    def to_strain(self, factor=1.0):
        if self.type == Tensor_type.STRESS:
            return self.__class__(
                self.xx * factor,
                self.yy * factor,
                self.zz * factor,
                2 * self.yz * factor,
                2 * self.xz * factor,
                2 * self.xy * factor,
                Tensor_type.STRAIN,
                self.is_deviatric
            )