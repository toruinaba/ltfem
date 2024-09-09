import numpy as np

class Boundary:
    def __init__(self, nodeNum):
        self.nodeNum = nodeNum # 全節点数
        self.nodeDof = 3 # 節点自由度
        self.vecDisp = np.array(nodeNum * self.nodeDof * [None]) # 単点拘束の強制変位
        self.vecForce = np.array(nodeNum * self.nodeDof * [0.0]) # 荷重ベクトル

    def addSPC(self, nodeNo, dispX, dispY, dispZ):
        # 単点拘束を追加
        self.vecDisp[self.nodeDof * (nodeNo - 1) + 0] = dispX
        self.vecDisp[self.nodeDof * (nodeNo - 1) + 1] = dispY
        self.vecDisp[self.nodeDof * (nodeNo - 1) + 2] = dispZ

    def makeDispVector(self):
        return self.vecDisp

    def addForce(self, nodeNo, fx, fy, fz):
        self.vecForce[self.nodeDof * (nodeNo - 1) + 0] = fx
        self.vecForce[self.nodeDof * (nodeNo - 1) + 1] = fy
        self.vecForce[self.nodeDof * (nodeNo - 1) + 2] = fz
    
    def makeForceVector(self):
        return self.vecForce
