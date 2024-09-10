import numpy as np
import numpy.linalg as LA
from tensor import Tensor3d, Tensor_type


class Multiline_isotropic_hardening:
    ITERNUM: int = 100
    TOL: float = 1.0e-6

    def __init__(self, young: float, poisson: float):
        self.young: float = young
        self.poisson: float = poisson
        self.stresses: list = []
        self.p_strains: list = []

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def K(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def De(self) -> np.ndarray:
        factor: float = self.young / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        De: np.ndarray = np.array([
                [1.0 - self.poisson, self.poisson, self.poisson, 0.0, 0.0, 0.0],
                [self.poisson, 1.0 - self.poisson, self.poisson, 0.0, 0.0, 0.0],
                [self.poisson, self.poisson, 1.0 - self.poisson, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson)]
            ]
        ) * factor
        return De

    @property
    def P(self) -> np.ndarray:
        return (1.0 / 3.0) * np.array(
            [
                [2.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, 2.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
            ]
        )

    @property
    def sigma_y(self):
        if len(self.stresses) == 0:
            raise ValueError("Defined no hardening points")
        return self.stresses[0]

    def add_hardening_point(self, stress: float, p_strain: float):
        if len(self.p_strains) == 0:
            if not p_strain == 0.0:
                raise ValueError("Stress-Strain data")
        elif self.p_strains[-1] > p_strain:
            raise ValueError("Wrong data")
        self.stresses.append(stress)
        self.p_strains.append(p_strain)

    def yield_function(self, mises_stress: float, ep_strain: float) -> float:
        return mises_stress - self.calculate_yield_stress(ep_strain)

    def calculate_yield_stress(self, ep_strain: float) -> float:
        yield_stress: float = self.sigma_y
        no = None
        for i in range(len(self.p_strains) - 1):
            if self.p_strains[i] <= ep_strain and ep_strain <= self.p_strains[i+1]:
                no = i
                break
            if i == len(self.p_strains) - 2:
                no = len(self.p_strains) - 2
        if no is None:
            raise ValueError("Range Error")
        h_dash = self.calculate_plastic_modulus(ep_strain)
        yield_stress = h_dash * (ep_strain - self.p_strains[no]) + self.stresses[no]
        return yield_stress

    def calculate_plastic_modulus(self, ep_strain: float) -> float:
        no = None
        for i in range(len(self.p_strains) - 1):
            if self.p_strains[i] <= ep_strain and ep_strain <= self.p_strains[i+1]:
                no = i
                break
            if i == len(self.p_strains) - 2:
                no = len(self.p_strains) - 2
        if no is None:
            raise ValueError("Range Error")
    
        h = (self.stresses[no+1] - self.stresses[no]) / (self.p_strains[no+1] - self.p_strains[no])
        return h

    def returnMapping3D(
        self,
        strain: Tensor3d,
        prev_p_strain: Tensor3d,
        prev_ep_strain: float
    ):
        # 偏差ひずみに変換
        deviatric_strain: Tensor3d = strain.deviatric
        # 試行応力を算出
        tri_stress: Tensor3d = 2.0 * self.G * (strain.deviatric - prev_p_strain).to_stress()
        # 試行降伏関数を算出
        triF: float = self.yield_function(tri_stress.mises, prev_ep_strain)
        # 降伏判定
        if triF > 0.0:
            y_flg: bool = True
        else:
            y_flg: bool = False
        # ニュートン法
        # 変数の初期化
        delta_gamma: float = 0.0
        if triF > 0.0:
            for i in range(self.ITERNUM):
                # 降伏応力度の算出
                yield_stress: float = self.calculate_yield_stress(prev_ep_strain + np.sqrt(2.0 / 3.0) * delta_gamma)
                # 関数の現在の値を算出
                y: float = tri_stress.norm - 2.0 * self.G * delta_gamma - np.sqrt(2.0 / 3.0) * yield_stress
                # 次ステップの勾配を算出
                h_dash: float = self.calculate_plastic_modulus(prev_ep_strain + np.sqrt(2.0 / 3.0) * delta_gamma)
                # 1次導関数の算出
                y_dash: float = - 2.0 * self.G - (2.0 / 3.0) * h_dash
                # 関数値の収束判定
                if np.abs(y) < self.TOL:
                    break
                elif (i + 1) == self.ITERNUM:
                    raise ValueError("Not Converged")
                # 解の更新
                delta_gamma -= y / y_dash
        
        if delta_gamma < 0:
            raise ValueError("Negative value Gamma")

        # ひずみ進展方向nの算出（流れ則）
        n: Tensor3d = tri_stress.to_strain() / tri_stress.norm # ひずみにつかうのでひずみ変換が必要
        # 塑性ひずみテンソル
        p_strain: Tensor3d = prev_p_strain + n * delta_gamma
        # 弾性ひずみテンソル
        e_strain: Tensor3d = strain - p_strain
    
        # 静水圧応力
        stress_m: float = self.K * strain.J1
        # 偏差応力テンソル
        deviatric_stress: Tensor3d = 2.0 * self.G * (deviatric_strain - p_strain).to_stress() # 弾性ひずみから算出
        # 応力テンソル
        stress: Tensor3d  = deviatric_stress + Tensor3d.from_matrix(stress_m * np.eye(3), Tensor_type.STRESS, False) #塑性変形中静水圧応力は変化しない
        stress.is_deviatric = False # 静水圧応力を足したので偏差フラグを外す

        # 相当塑性ひずみ
        ep_strain: float = prev_ep_strain + np.sqrt(2.0 / 3.0) * delta_gamma

        # Deqマトリクスの計算
        if y_flg == True:
            Dep: np.ndarray = self.calculate_Dep(stress, ep_strain, prev_ep_strain)
        else:
            Dep: np.ndarray = self.De
        return stress, e_strain, p_strain, ep_strain, y_flg, Dep

    def calculate_Dep(self, stress: Tensor3d, ep_strain: float, prev_ep_strain: float) -> np.ndarray:
        delta_ep_strain: float = ep_strain - prev_ep_strain # Δepを計算
        gamma_dash: float = 3.0 * delta_ep_strain / (2.0 * stress.mises) # ひずみ増分の方向ベクトルの微分?
        A: np.ndarray = LA.inv(LA.inv(self.De) + gamma_dash * self.P)
        h_dash: float = self.calculate_plastic_modulus(ep_strain)
        a: float = np.power(1.0 - (2.0 / 3.0) * gamma_dash * h_dash, -1)
        d_stress: np.ndarray = self.P @ stress.vector
        factor1: np.ndarray = np.array(A @ (np.matrix(d_stress).T * np.matrix(d_stress)) @ A)
        factor2: np.ndarray = (4.0 / 9.0) * a * h_dash * (stress.mises)**2 + (np.matrix(d_stress) @ A @ np.matrix(d_stress).T)[0,0]
        Dep: np.ndarray = np.array(A - factor1 / factor2)
        return Dep



class Linear_isotropic_hardening:
    ITERNUM: int = 100
    TOL: float = 1.0e-6

    def __init__(self, young: float, poisson: float, sigma_y: float, plastic_modulus: float):
        self.young: float = young
        self.poisson: float = poisson
        self.sigma_y: float = sigma_y
        self.plastic_modulus: float = plastic_modulus

    @property
    def G(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def K(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def De(self) -> np.ndarray:
        factor: float = self.young / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        De: np.ndarray = np.array([
                [1.0 - self.poisson, self.poisson, self.poisson, 0.0, 0.0, 0.0],
                [self.poisson, 1.0 - self.poisson, self.poisson, 0.0, 0.0, 0.0],
                [self.poisson, self.poisson, 1.0 - self.poisson, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson)]
            ]
        ) * factor
        return De

    @property
    def P(self) -> np.ndarray:
        return (1.0 / 3.0) * np.array(
            [
                [2.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, 2.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
            ]
        )

    def yield_function(self, mises_stress: float, ep_strain: float) -> float:
        return mises_stress - self.calculate_yield_stress(ep_strain)

    def calculate_yield_stress(self, ep_strain: float) -> float:
        return self.sigma_y + self.plastic_modulus * ep_strain

    def calculate_plastic_modulus(self, ep_strain: float) -> float:
        return self.plastic_modulus

    def returnMapping3D(
        self,
        strain: Tensor3d,
        prev_p_strain: Tensor3d,
        prev_ep_strain: float
    ):
        # 偏差ひずみに変換
        deviatric_strain: Tensor3d = strain.deviatric
        # 試行応力を算出
        tri_stress: Tensor3d = 2.0 * self.G * (strain.deviatric - prev_p_strain).to_stress()
        # 試行降伏関数を算出
        triF: float = self.yield_function(tri_stress.mises, prev_ep_strain)
        # 降伏判定
        if triF > 0.0:
            y_flg: bool = True
        else:
            y_flg: bool = False
        # ニュートン法
        # 変数の初期化
        delta_gamma: float = 0.0
        if triF > 0.0:
            delta_gamma = triF / (self.young + self.plastic_modulus)
        # ひずみ進展方向nの算出（流れ則）
        n: Tensor3d = tri_stress.to_strain() / tri_stress.norm # ひずみにつかうのでひずみ変換が必要
        # 塑性ひずみテンソル
        p_strain: Tensor3d = prev_p_strain + n * delta_gamma
        # 弾性ひずみテンソル
        e_strain: Tensor3d = strain - p_strain
        # 静水圧応力
        stress_m: float = self.K * strain.J1
        # 偏差応力テンソル
        deviatric_stress: Tensor3d = 2.0 * self.G * (deviatric_strain - p_strain).to_stress() # 弾性ひずみから算出
        # 応力テンソル
        stress: Tensor3d  = deviatric_stress + Tensor3d.from_matrix(stress_m * np.eye(3), Tensor_type.STRESS, False) #塑性変形中静水圧応力は変化しない
        stress.is_deviatric = False # 静水圧応力を足したので偏差フラグを外す

        # 相当塑性ひずみ
        ep_strain: float = prev_ep_strain + np.sqrt(2.0 / 3.0) * delta_gamma

        # Deqマトリクスの計算
        if y_flg == True:
            Dep: np.ndarray = self.calculate_Dep(stress, ep_strain, prev_ep_strain)
        else:
            Dep: np.ndarray = self.De
        return stress, e_strain, p_strain, ep_strain, y_flg, Dep

    def calculate_Dep(self, stress: Tensor3d, ep_strain: float, prev_ep_strain: float) -> np.ndarray:
        delta_ep_strain: float = ep_strain - prev_ep_strain # Δepを計算
        gamma_dash: float = 3.0 * delta_ep_strain / (2.0 * stress.mises) # ひずみ増分の方向ベクトルの微分?
        A: np.ndarray = LA.inv(LA.inv(self.De) + gamma_dash * self.P)
        h_dash: float = self.plastic_modulus
        a: float = np.power(1.0 - (2.0 / 3.0) * gamma_dash * h_dash, -1)
        d_stress: np.ndarray = self.P @ stress.vector
        factor1: np.ndarray = np.array(A @ (np.matrix(d_stress).T * np.matrix(d_stress)) @ A)
        factor2: np.ndarray = (4.0 / 9.0) * a * h_dash * (stress.mises)**2 + (np.matrix(d_stress) @ A @ np.matrix(d_stress).T)[0,0]
        Dep: np.ndarray = np.array(A - factor1 / factor2)
        return Dep
