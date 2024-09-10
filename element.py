import numpy as np
import numpy.linalg as LA
from dmatrix import Dmatrix
from tensor import Tensor3d, Tensor_type
from element_output_data import ElementOutputData

class C3D8:
    NODENUM = 8
    NODEDOF = 3

    def __init__(self, no, nodes, matSet):
        self.no = no
        self.nodes = nodes
        self.young = matSet.young
        self.poisson = matSet.poisson
        self.density = matSet.density
        self.material = matSet.material #構成則
        self.ipNum = 8 #積分点の数
        self.w1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.w2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.w3 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.ai = np.array(
            [
                -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0),
                -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0),
            ]
        ) # 積分点のa座標（8点）
        self.bi = np.array(
            [
                -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0),
                -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)
            ]
        ) # 積分点のb座標（8点）
        self.ci = np.array(
            [
                -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0),
                np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)
            ]
        ) # 積分点のc座標（8点）
        self.incNo = 0 # インクリメントNo

        self.vecDisp = np.zeros(self.NODENUM * self.NODEDOF)
        self.yieldFlgList = [] #
        self.vecEStrainList = []
        self.vecPStrainList = []
        self.ePStrainList = []
        self.vecStressList = []
        self.misesList = []

        for i in range(self.ipNum):
            self.yieldFlgList.append(False)
            self.vecEStrainList.append(Tensor3d.from_vector(np.zeros(6), Tensor_type.STRAIN))
            self.vecPStrainList.append(Tensor3d.from_vector(np.zeros(6), Tensor_type.STRAIN))
            self.ePStrainList.append(0.0)
            self.vecStressList.append(Tensor3d.from_vector(np.zeros(6), Tensor_type.STRESS))
            self.misesList.append(0.0)
    
        self.vecPrevEStrainList = []
        self.vecPrevPStrainList = []
        self.prevEPStrainList = []
        for i in range(self.ipNum):
            self.vecPrevEStrainList.append(Tensor3d.from_vector(np.zeros(6), Tensor_type.STRAIN))
            self.vecPrevPStrainList.append(Tensor3d.from_vector(np.zeros(6), Tensor_type.STRAIN))
            self.prevEPStrainList.append(0.0)
        
        self.matD = []
        for i in range(self.ipNum):
            self.matD.append(self.makeDematrix())
    
    def makeKetmatrix(self):
        matJ = [] # ヤコビ行列
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))

        matBbar = [] # Bbarマトリクス
        for i in range(self.ipNum):
            matBbar.append(self.makeBbarmatrix(self.ai[i], self.bi[i], self.ci[i]))

        matKet = np.zeros([self.NODEDOF * self.NODENUM, self.NODEDOF * self.NODENUM])
        for i in range(self.ipNum):
            matKet += self.w1[i] * self.w2[i] * self.w3[i] * matBbar[i].T @ self.matD[i] @ matBbar[i] * LA.det(matJ[i])
        
        return matKet

    def makeDematrix(self):
        matD = Dmatrix(self.young, self.poisson).makeDematrix()
        return matD

    def makeJmatrix(self, a, b, c):
        matdNdabc = self.makedNdabc(a, b, c)

        matxiyizi = np.array(
            [
                [self.nodes[0].x, self.nodes[0].y, self.nodes[0].z],
                [self.nodes[1].x, self.nodes[1].y, self.nodes[1].z],
                [self.nodes[2].x, self.nodes[2].y, self.nodes[2].z],
                [self.nodes[3].x, self.nodes[3].y, self.nodes[3].z],
                [self.nodes[4].x, self.nodes[4].y, self.nodes[4].z],
                [self.nodes[5].x, self.nodes[5].y, self.nodes[5].z],
                [self.nodes[6].x, self.nodes[6].y, self.nodes[6].z],
                [self.nodes[7].x, self.nodes[7].y, self.nodes[7].z]
            ]
        )
        matJ = matdNdabc @ matxiyizi
        
        if LA.det(matJ) < 0:
            raise ValueError("fail calculate. element jacobian is zero.")
    
        return matJ

    def makeBmatrix(self, a, b, c):
        matdNdabc = self.makedNdabc(a, b, c)
        matJ = self.makeJmatrix(a, b, c)
        matdNdxyz = LA.solve(matJ, matdNdabc)

        matB = np.empty((6, 0))
        for i in range(self.NODENUM):
            matTmp = np.array(
                [
                    [matdNdxyz[0, i], 0.0, 0.0],
                    [0.0, matdNdxyz[1, i], 0.0],
                    [0.0, 0.0, matdNdxyz[2, i]],
                    [0.0, matdNdxyz[2, i], matdNdxyz[1, i]],
                    [matdNdxyz[2, i], 0.0, matdNdxyz[0, i]],
                    [matdNdxyz[1, i], matdNdxyz[0, i], 0.0]
                ]
            )
            matB = np.hstack((matB, matTmp))
        return matB

    def makeBbarmatrix(self, a, b, c):
        matB = self.makeBmatrix(a, b, c) # Bmatrix
        matBv = self.makeBvmatrix(a, b, c) # Bvmatrix
        matBvbar = self.makeBvbarmatrix()
        matBbar = matBvbar + matB - matBv
        return matBbar
    
    def makeBvmatrix(self, a, b, c):
        matdNdabc = self.makedNdabc(a, b, c)
        matJ = self.makeJmatrix(a, b, c)
        matdNdxyz = LA.solve(matJ, matdNdabc)
        matBv = np.empty((6, 0))
        for i in range(self.NODENUM):
            matTmp = np.array(
                [
                    [matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                    [matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                    [matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]
                ]
            )
            matBv= np.hstack((matBv, matTmp))
        matBv *= 1.0 / 3.0
        return matBv

    def makeBvbarmatrix(self):
        v = self.getVolume()
        matBv = []
        for i in range(self.ipNum):
            matBv.append(self.makeBvmatrix(self.ai[i], self.bi[i], self.ci[i]))
        
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))
        
        Bvbar = np.zeros([6, self.NODENUM * self.NODEDOF])
        for i in range(self.ipNum):
            Bvbar += self.w1[i] * self.w2[i] * self.w3[i] * matBv[i] * LA.det(matJ[i])
        Bvbar *= 1.0 / v
        return Bvbar

    def makedNdabc(self, a, b, c):
        dN1da = -0.125 * (1.0 - b) * (1.0 - c)
        dN2da = 0.125 * (1.0 - b) * (1.0 - c)
        dN3da = 0.125 * (1.0 + b) * (1.0 - c)
        dN4da = -0.125 * (1.0 + b) * (1.0 - c)
        dN5da = -0.125 * (1.0 - b) * (1.0 + c)
        dN6da = 0.125 * (1.0 - b) * (1.0 + c)
        dN7da = 0.125 * (1.0 + b) * (1.0 + c)
        dN8da = -0.125 * (1.0 + b) * (1.0 + c)
        dN1db = -0.125 * (1.0 - a) * (1.0 - c)
        dN2db = -0.125 * (1.0 + a) * (1.0 - c)
        dN3db = 0.125 * (1.0 + a) * (1.0 - c)
        dN4db = 0.125 * (1.0 - a) * (1.0 - c)
        dN5db = -0.125 * (1.0 - a) * (1.0 + c)
        dN6db = -0.125 * (1.0 + a) * (1.0 + c)
        dN7db = 0.125 * (1.0 + a) * (1.0 + c)
        dN8db = 0.125 * (1.0 - a) * (1.0 + c)
        dN1dc = -0.125 * (1.0 - a) * (1.0 - b)
        dN2dc = -0.125 * (1.0 + a) * (1.0 - b)
        dN3dc = -0.125 * (1.0 + a) * (1.0 + b)
        dN4dc = -0.125 * (1.0 - a) * (1.0 + b)
        dN5dc = 0.125 * (1.0 - a) * (1.0 - b)
        dN6dc = 0.125 * (1.0 + a) * (1.0 - b)
        dN7dc = 0.125 * (1.0 + a) * (1.0 + b)
        dN8dc = 0.125 * (1.0 - a) * (1.0 + b)

        dNdabc = np.array(
            [
                [dN1da, dN2da, dN3da, dN4da, dN5da, dN6da, dN7da, dN8da],
                [dN1db, dN2db, dN3db, dN4db, dN5db, dN6db, dN7db, dN8db],
                [dN1dc, dN2dc, dN3dc, dN4dc, dN5dc, dN6dc, dN7dc, dN8dc]
            ]
        )
        return dNdabc
    
    def getVolume(self):
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))
        
        volume = 0
        for i in range(self.ipNum):
            volume += self.w1[i] * self.w2[i] * self.w3[i] * LA.det(matJ[i])
        return volume

    def update(self, vecDisp, incNo):
        self.returnMapping(vecDisp, incNo)
        self.vecDisp = vecDisp
        self.incNo = incNo

    def returnMapping(self, vecDisp, incNo):
        vecStrainList = self.makeStrainVector(vecDisp)

        for i in range(self.ipNum):
            if self.incNo == incNo:
                (
                    vecStress,
                    vecEStrain,
                    vecPStrain,
                    ePStrain,
                    yieldFlg,
                    matDep
                ) = self.material.returnMapping3D(
                    vecStrainList[i],
                    self.vecPrevPStrainList[i],
                    self.prevEPStrainList[i]
                )
            else:
                self.vecPrevEStrainList = self.vecEStrainList
                self.vecPrevPStrainList = self.vecPStrainList
                self.prevEPStrainList = self.ePStrainList

                (
                    vecStress,
                    vecEStrain,
                    vecPStrain,
                    ePStrain,
                    yieldFlg,
                    matDep
                ) = self.material.returnMapping3D(
                    vecStrainList[i],
                    self.vecPStrainList[i],
                    self.ePStrainList[i]
                )

            self.vecStressList[i] = vecStress
            self.vecEStrainList[i] = vecEStrain
            self.vecPStrainList[i] = vecPStrain
            self.ePStrainList[i] = ePStrain
            self.yieldFlgList[i] = yieldFlg
            self.misesList[i] = vecStress.mises
            self.matD[i] = matDep

    def makeqVector(self):
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))
        
        matBbar = []
        for i in range(self.ipNum):
            matBbar.append(self.makeBbarmatrix(self.ai[i], self.bi[i], self.ci[i]))
        
        vecq = np.zeros(self.NODEDOF * self.NODENUM)
        for i in range(self.ipNum):
            vecq += self.w1[i] * self.w2[i] * self.w3[i] * matBbar[i].T @ self.vecStressList[i].vector * LA.det(matJ[i])
        return vecq

    def makeStrainVector(self, vecDisp):
        matBbar = []
        for i in range(self.ipNum):
            matBbar.append(self.makeBbarmatrix(self.ai[i], self. bi[i], self.ci[i]))
    
        vecIpStrains = []
        for i in range(self.ipNum):
            vecIpStrains.append(Tensor3d.from_vector(np.array(matBbar[i] @ vecDisp), Tensor_type.STRAIN))
        return vecIpStrains

    def makeOutputData(self):
        output = ElementOutputData(
            self,
            self.vecStressList,
            self.vecEStrainList,
            self.vecPStrainList,
            self.ePStrainList,
            self.misesList,
            self.yieldFlgList
        )
        return output