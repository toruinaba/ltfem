import numpy as np
import numpy.linalg as LA
from Dmatrix import Dmatrix
from Tensor import Tensor3d, Tensor_type

class MisesMaterial:
    ITERNUM = 100
    TOL = 1.0e-6
    # young: young modulus
    # poisson: poisson ratio
    def __init__(self, young, poisson):
        self.young = young
        self.poisson = poisson
        self.stressLine = []
        self.pStrainLine = []

    @property
    def G(self):
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def K(self):
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    def addStressPStrainLine(self, stress, pStrain):
        if len(self.pStrainLine) == 0:
            if not pStrain == 0.0:
                raise ValueError("Stress-Strain data")    
        elif self.pStrainLine[-1] > pStrain:
            raise ValueError("Wrong data")
        if len(self.stressLine) == 0:
            self.yieldStress = stress
        
        self.stressLine.append(stress)
        self.pStrainLine.append(pStrain)

    def yieldFunction(self, mStress, ePStrain):
        f = 0.0
        if hasattr(self, 'yieldStress'):
            f = mStress - self.makeYieldStress(ePStrain)
        return f

    def makeYieldStress(self, ePStrain):
        yieldStress = 0.0
        if hasattr(self, 'yieldStress'):
            no = None
            for i in range(len(self.pStrainLine) - 1):
                if self.pStrainLine[i] <= ePStrain and ePStrain <= self.pStrainLine[i+1]:
                    no = i
                    break
            if no is None:
                raise ValueError("Range Error")
        
            hDash = self.makePlasticModule(ePStrain)
            yieldStress = hDash * (ePStrain - self.pStrainLine[no]) + self.stressLine[no]
        return yieldStress

    def makePlasticModule(self, ePStrain):
        no = None
        for i in range(len(self.pStrainLine) - 1):
            if self.pStrainLine[i] <= ePStrain and ePStrain <= self.pStrainLine[i+1]:
                no = i
        if no is None:
            raise ValueError("Range Error")
    
        h = (self.stressLine[no+1] - self.stressLine[no]) / (self.pStrainLine[no+1] - self.pStrainLine[no])
        return h

    def returnMapping3D(self, vecStrain, vecPrevPStrain, prevEPStrain):
        # 偏差ひずみのテンソルを求める
        mStrain = (1.0 / 3.0) * (vecStrain[0] + vecStrain[1] + vecStrain[2])
        tenStrain = np.array([[vecStrain[0], vecStrain[5] * 0.5, vecStrain[4] * 0.5],
                              [vecStrain[5] * 0.5, vecStrain[1], vecStrain[3] * 0.5],
                              [vecStrain[4] * 0.5, vecStrain[3] * 0.5, vecStrain[2]]])
        tenDStrain = tenStrain - mStrain * np.eye(3)

        # 前回の塑性ひずみのテンソルを求める
        tenPrevPStrain = np.array([[vecPrevPStrain[0], vecPrevPStrain[5] * 0.5, vecPrevPStrain[4] * 0.5],
                                   [vecPrevPStrain[5] * 0.5, vecPrevPStrain[1], vecPrevPStrain[3] * 0.5],
                                   [vecPrevPStrain[4] * 0.5, vecPrevPStrain[3] * 0.5, vecPrevPStrain[2]]])

        # 試行弾性偏差応力のテンソルを求める
        tenTriDStress = 2.0 * self.G * (tenDStrain - tenPrevPStrain)

        # 試行弾性応力のミーゼス応力を求める
        mTriStress = np.sqrt(3.0 / 2.0) * LA.norm(tenTriDStress, "fro")

        triF = self.yieldFunction(mTriStress, prevEPStrain) #試行降伏関数

        if triF > 0.0:
            yieldFlg = True
    
        else:
            yieldFlg = False

        deltaGamma = 0.0
        if triF > 0.0:
            normTriDStress = LA.norm(tenTriDStress, "fro")
            for i in range(self.ITERNUM):
                print(f"iter{i}: gamma{deltaGamma}")
                yieldStress = self.makeYieldStress(prevEPStrain + np.sqrt(2.0 / 3.0) * deltaGamma)
                y = normTriDStress - 2.0 * self.G * deltaGamma - np.sqrt(2.0 / 3.0) * yieldStress

                hDash = self.makePlasticModule(prevEPStrain + np.sqrt(2.0 / 3.0) * deltaGamma)
                yDash = - 2.0 * self.G - (2.0 / 3.0) * hDash

                if np.abs(y) < self.TOL:
                    break
                elif (i + 1) == self.ITERNUM:
                    raise ValueError("Not Converged")
            
                deltaGamma -= y / yDash
        
        if deltaGamma < 0:
            raise ValueError("Negative value Gamma")

        tenN = tenTriDStress / LA.norm(tenTriDStress, "fro") # ひずみ進展方向の算出（流れ則）
        tenPStrain = tenPrevPStrain + tenN * deltaGamma # 塑性ひずみテンソル
        tenEStrain = tenStrain - tenPStrain # 弾性ひずみテンソル(全ひずみと塑性ひずみから)

        vStrain = vecStrain[0] + vecStrain[1] + vecStrain[2] # 体積ひずみ
        mSigma = self.K * vStrain # 静水圧応力
        tenDStress = 2.0 * self.G * (tenDStrain - tenPStrain)
        tenStress  = tenDStress + mSigma * np.eye(3) #リターンマップ前後で静水圧応力は変化しない→応力テンソルに直す

        ePStrain = prevEPStrain + np.sqrt(2.0 / 3.0) * deltaGamma # 相当塑性ひずみ
        vecStress = np.array(
            [
                tenStress[0, 0],
                tenStress[1, 1],
                tenStress[2, 2],
                tenStress[1, 2],
                tenStress[0, 2],
                tenStress[0, 1]
            ]
        ) # 応力テンソル→ベクトルに変換
        vecEStrain = np.array(
            [
                tenEStrain[0, 0],
                tenEStrain[1, 1],
                tenEStrain[2, 2],
                2.0 * tenEStrain[1, 2],
                2.0 * tenEStrain[0, 2],
                2.0 * tenEStrain[0, 1]
            ]
        ) # 弾性ひずみテンソル→ベクトルに変換
        vecPStrain = np.array(
            [
                tenPStrain[0, 0],
                tenPStrain[1, 1],
                tenPStrain[2, 2],
                2.0 * tenPStrain[1, 2],
                2.0 * tenPStrain[0, 2],
                2.0 * tenPStrain[0, 1]
            ]
        ) # 塑性ひずみテンソル→ベクトルに変換

        if yieldFlg == True:
            matDep = self.makeDepmatrix3D(vecStress, ePStrain, prevEPStrain)
        
        else:
            matDep = Dmatrix(self.young, self.poisson).makeDematrix()

        return vecStress, vecEStrain, vecPStrain, ePStrain, yieldFlg, matDep
    
    def misesStress3D(self, vecStress):
        # ベクトルからmises応力を作成
        factor1 = np.square(vecStress[0] - vecStress[1]) + np.square(vecStress[1] - vecStress[2]) + np.square(vecStress[2] - vecStress[0])
        factor2 = 6.0 * (np.square(vecStress[3]) + np.square(vecStress[4]) + np.square(vecStress[5]))
        mises = np.sqrt(0.5 * (factor1 + factor2))
        return mises

    def makeDepmatrix3D(self, vecStress, ePStrain, prevEPStrain):
        matDe = Dmatrix(self.young, self.poisson).makeDematrix() # Deマトリクス
        deltaEPStrain = ePStrain - prevEPStrain # Δepを計算
        mStress = self.misesStress3D(vecStress) # mises応力を計算
        gammaDash = 3.0 * deltaEPStrain / (2.0 * mStress) # ひずみ増分の方向ベクトルの微分?
        matP = (1.0 / 3.0) * np.array(
            [
                [2.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, 2.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
            ]
        )

        matA = LA.inv(LA.inv(matDe) + gammaDash * matP)
        hDash = self.makePlasticModule(ePStrain)
        a = np.power(1.0 - (2.0 / 3.0) * gammaDash * hDash, -1)
        vecDStress = matP @ vecStress
        factor1 = np.array(matA @ (np.matrix(vecDStress).T * np.matrix(vecDStress)) @ matA)
        factor2 = (4.0 / 9.0) * a * hDash * mStress**2 + (np.matrix(vecDStress) @ matA @ np.matrix(vecDStress).T)[0,0]
        matDep = np.array(matA - factor1 / factor2)
        return matDep