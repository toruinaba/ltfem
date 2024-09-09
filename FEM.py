import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import copy

class FEM:
    def __init__(self, nodes, elements, bound, incNum):
        self.nodeDof = 3 # 節点自由度
        self.nodes = nodes # 1から始まる順番で並んだNodeリスト
        self.elements = elements # 要素は種類ごとにソートされているものとする
        self.bound = bound # 境界条件
        self.incNum = incNum # インクリメント数
        self.itrNum = 100 # ニュートン法のイテレーションの上限数
        self.rn = 0.005 # ニュートン法の残差力の収束判定パラメータ
        self.cn = 0.01 # ニュートン法の変位の収束判定パラメータ
        self.curve_t = []
        self.curve_x = []

    def calculate_amplitude(self, value):
        no = None
        for i in range(len(self.curve_t) - 1):
            if self.curve_t[i] <= value and value <= self.curve_t[i+1]:
                no = i
                break
        if no is None:
            raise ValueError("RangeError")
        amp_dash = (self.curve_x[no+1] - self.curve_x[no]) / (self.curve_t[no+1] - self.curve_t[no])
        return self.curve_x[no] + amp_dash * (value - self.curve_t[no])

    def add_curve_point(self, t, x):
        self.curve_t.append(t)
        self.curve_x.append(x)

    def impAnalysis(self):
        # 陰解法
        self.vecDispList = [] # 各インクリメントの変位ベクトルのリスト
        self.vecRFList = [] # 各インクリメントの反力ベクトルのリスト
        self.elemOutputDataList = [] # 各インクリメントの要素出力のリスト（makeOutputData型）

        # 荷重をインクリメントに分割
        vecfList = []
        for i in range(self.incNum):
            amp = self.calculate_amplitude((i + 1) / self.incNum)
            vecfList.append(self.makeForceVector() * amp)
        
        vecdList = []
        for i in range(self.incNum):
            dispv = self.makeDispVector()
            amp = self.calculate_amplitude((i + 1) / self.incNum)
            dived = np.array([None if x is None else x * amp for x in dispv])
            vecdList.append(dived)

        # ニュートン法
        vecDisp = np.zeros(len(self.nodes) * self.nodeDof) # 全節点の変位ベクトルの初期化
        vecR = np.zeros(len(self.nodes) * self.nodeDof) # 残差力ベクトルの初期化
        for i in range(self.incNum):
            print(f"INCREMENT {i+1}")
            vecDispFirst = vecDisp.copy() # 初期全節点変位ベクトル
            vecf = vecfList[i] # i+1番インクリメントの荷重
            vecBoundDisp = vecdList[i]

            # 境界条件を考慮しないインクリメント初期の残差力ベクトルRを作成
            if i == 0:
                vecR = vecfList[0]
            else:
                vecR = vecfList[i] - vecfList[i - 1]

            for j in range(self.itrNum):
                print(f"Iteration {j}")
                # 接線剛性マトリクスKtを作成
                matKt = self.makeKtmatrix()

                # 境界条件を考慮したKtcマトリクス, Rcベクトルを作成
                matKtc, vecRc = self.setBoundCondition(matKt, vecR, vecBoundDisp, vecDisp)

                # Ktcの逆行列が計算可能か確認
                if np.isclose(LA.det(matKtc), 0.0):
                    raise ValueError("Calculation failed. Cannot calculate inversion of Ktc.")
                
                # 変位ベクトルを計算
                vecd = LA.solve(matKtc, vecRc)
                vecDisp += vecd

                # 要素内変数の更新
                self.updateElements(vecDisp, i)

                # 残差ベクトルRを求める
                vecQ = np.zeros(len(self.nodes) * self.nodeDof)
                for elem in self.elements:
                    vecq = elem.makeqVector()
                    for k in range(len(elem.nodes)):
                        for l in range(elem.NODEDOF):
                            vecQ[(elem.nodes[k].no - 1) * self.nodeDof + l] += vecq[k * elem.NODEDOF + l]
                vecR = vecf - vecQ
                
                #境界条件を考慮したRcベクトルを作成
                matKt = self.makeKtmatrix()
                matKtc, vecRc = self.setBoundCondition(matKt, vecR, vecBoundDisp, vecDisp)

                # 時間平均力を計算
                aveForce = 0.0
                cnt = len(self.nodes) * self.nodeDof
                for k in range(len(vecQ)):
                    aveForce += np.abs(vecQ[k])
                for k in range(len(vecf)):
                    if not vecf[k] == 0.0:
                        aveForce += np.abs(vecf[k])
                        cnt += 1
                aveForce = aveForce / cnt

                if np.allclose(vecRc, 0.0):
                    print("Converged with Rc")
                    break
                if np.isclose(LA.norm(vecd), 0.0):
                    print("Converged with Disp")
                    break
                dispRate = LA.norm(vecd) / LA.norm(vecDisp - vecDispFirst)
                ResiForceRate = np.abs(vecRc).max() / aveForce
                if dispRate < self.cn and ResiForceRate < self.rn:
                    print("Converged with cn and rn")
                    break
            
            # 最終的な変位ベクトルを格納
            self.vecDispList.append(vecDisp.copy())

            # 変位ベクトルから要素出力データを計算
            elemOutputDatas = []
            for elem in self.elements:
                elemOutputData = elem.makeOutputData()
                elemOutputDatas.append(copy.deepcopy(elemOutputData))
            self.elemOutputDataList.append(elemOutputDatas)

            # 節点反力を計算
            vecRF = np.array(vecQ - vecf).flatten()

            # 最終的な節点反力を格納
            self.vecRFList.append(vecRF)
    
    def makeKtmatrix(self):
        matKt = np.matrix(np.zeros((len(self.nodes) * self.nodeDof, len(self.nodes) * self.nodeDof)))
        for elem in self.elements:
            matKet = elem.makeKetmatrix()

            for c in range(len(elem.nodes) * self.nodeDof):
                ct = (elem.nodes[c // self.nodeDof].no - 1) * self.nodeDof + c % self.nodeDof
                for r in range(len(elem.nodes) * self.nodeDof):
                    rt = (elem.nodes[r //self.nodeDof].no - 1) * self.nodeDof + r % self.nodeDof
                    matKt[ct, rt] += matKet[c, r]
        return matKt

    def makeForceVector(self):
        vecCondiForce = self.bound.makeForceVector()
        vecf = vecCondiForce
        return vecf

    def makeDispVector(self):
        vecCondiDisp = self.bound.makeDispVector()
        vecd = vecCondiDisp
        return vecd
    

    def setBoundCondition(self, matKt, vecR, vecBoundDisp, vecDisp):
        matKtc = np.copy(matKt)
        vecRc = np.copy(vecR)

        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] is None:
                vecx = np.array(matKt[:, i]).flatten()

                # 変位ベクトルの影響を荷重ベクトルに適用
                vecRc = vecRc - (vecBoundDisp[i] - vecDisp[i]) * vecx

                # 境界条件が与えられている自由度を変数から除外
                matKtc[:, i] = 0.0
                matKtc[i, :] = 0.0
                matKtc[i, i] = 1.0
        
        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] is None:
                vecRc[i] = vecBoundDisp[i] - vecDisp[i]
        return matKtc, vecRc

    def updateElements(self, vecDisp, incNo):
        for elem in self.elements:
            vecElemDisp = np.zeros(len(elem.nodes) * self.nodeDof)
            for i in range(len(elem.nodes)):
                for j in range(elem.NODEDOF):
                    vecElemDisp[i * elem.NODEDOF + j] = vecDisp[(elem.nodes[i].no - 1) * self.nodeDof + j]
            elem.update(vecElemDisp, incNo)

    def outputTxt(self, filePath):
        f = open(filePath + ".txt", 'w')

        columNum = 20
        floatDigits = ".10g"

        f.write("**************************************\n")
        f.write("**            Input Data            **\n")
        f.write("**************************************\n")
        f.write("\n")

        # 節点情報
        f.write("***** Node Data *****\n")
        f.write("No".rjust(columNum) + "X".rjust(columNum) + "Y".rjust(columNum) + "Z".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        for node in self.nodes:
            strNo = str(node.no).rjust(columNum)
            strX = str(format(node.x, floatDigits).rjust(columNum))
            strY = str(format(node.y, floatDigits).rjust(columNum))
            strZ = str(format(node.z, floatDigits).rjust(columNum))
            f.write(strNo + strX + strY + strZ + "\n")
        f.write("\n")

        # 要素情報
        nodeNoColumNum = 36
        f.write("***** Element Data *****\n")
        f.write(
            "No".rjust(columNum)  + "Type".rjust(columNum) + "Node No".rjust(nodeNoColumNum) +
            "Young".rjust(columNum) + "Poisson".rjust(columNum) + "Thickness".rjust(columNum) +
            "Area".rjust(columNum) + "Density".rjust(columNum) + "YieldStress".rjust(columNum) + "\n"
        )
        f.write("-" * columNum * 8 + "-" * nodeNoColumNum + "\n")
        for elem in self.elements:
            strNo = str(elem.no).rjust(columNum)
            strType = str(elem.__class__.__name__).rjust(columNum)
            strNodeNo = ""
            for node in elem.nodes:
                strNodeNo += " " + str(node.no)
            strNodeNo = strNodeNo.rjust(nodeNoColumNum)
            strYoung = str(format(elem.young, floatDigits).rjust(columNum))
            strPoisson = "None".rjust(columNum)
            if hasattr(elem, 'poisson'):
                strPoisson = str(format(elem.poisson, floatDigits).rjust(columNum))
            strThickness = "None".rjust(columNum)
            if hasattr(elem, "thickness"):
                strThickness = str(format(elem.thickness, floatDigits).rjust(columNum))
            strArea = "None".rjust(columNum)
            if hasattr(elem, 'area'):
                strArea = str(format(elem.area, floatDigits).rjust(columNum))
            strDensity = "None".rjust(columNum)
            if not elem.density is None:
                strDensity = str(format(elem.density, floatDigits).rjust(columNum))
            strYieldStress = "None".rjust(columNum)
            if not hasattr(elem.material, 'yieldStress'):
                strYieldStress = str(format(elem.material.sigma_y, floatDigits).rjust(columNum))
            f.write(
                strNo + strType + strNodeNo + strYoung + strPoisson + strThickness + strArea + strDensity + strYieldStress + "\n"
            )
        
        f.write("\n")

        # 拘束情報
        f.write("***** SPC Constraint Data *****\n")
        f.write(
            "NodeNo".rjust(columNum) +
            "X Displacement".rjust(columNum) +
            "Y Displacement".rjust(columNum) +
            "Z Displacement".rjust(columNum)
        )
        f.write("-" * columNum * 4 + "\n")
        vecBoundDisp = self.bound.makeDispVector()
        for i in range(self.bound.nodeNum):
            strFlg = False
            for j in range(self.bound.nodeDof):
                if not vecBoundDisp[i * self.bound.nodeDof + j] is None:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof] is None:
                    strXDisp = str(format(vecBoundDisp[i * self.bound.nodeDof], floatDigits).rjust(columNum))
                strYDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof + 1] is None:
                    strYDisp = str(format(vecBoundDisp[i * self.bound.nodeDof + 1], floatDigits).rjust(columNum))
                strZDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof + 2] is None:
                    strZDisp = str(format(vecBoundDisp[i * self.bound.nodeDof + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXDisp + strYDisp + strZDisp + "\n")

        # 荷重条件出力（等価節点力を含む）
        f.write("***** Nodal Force Data ******\n")
        f.write(
            "NodeNo".rjust(columNum) +
            "X Force".rjust(columNum) +
            "Y Force".rjust(columNum) +
            "Z Force".rjust(columNum)
        )
        f.write("-" * columNum * 4 + "\n")
        vecf = self.makeForceVector()
        for i in range(len(self.nodes)):
            strFlg = False
            for j in range(self.bound.nodeDof):
                if not vecf[i * self.bound.nodeDof + j] == 0.0:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXForce = str(format(vecf[i * self.bound.nodeDof], floatDigits).rjust(columNum))
                strYForce = str(format(vecf[i * self.bound.nodeDof + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecf[i * self.bound.nodeDof + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + strYForce + strZForce + "\n")
        
        f.write("\n")

        #　結果データ
        f.write("**************************************\n")
        f.write("**           Result Data            **\n")
        f.write("**************************************\n")
        f.write("\n")

        for i in range(self.incNum):
            f.write("*Increment " + str(i + 1) + "\n")
            f.write("\n")

            # 変位
            f.write("***** Displacement Data ******\n")
            f.write(
                "NodeNo".rjust(columNum) +
                "Magnitude".rjust(columNum) +
                "X Displacement".rjust(columNum) +
                "Y Displacement".rjust(columNum) +
                "Z Displacement".rjust(columNum) + "\n"
            )
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecDisp = self.vecDispList[i]
                mag = np.linalg.norm(
                    np.array(
                        (vecDisp[self.nodeDof * j], vecDisp[self.nodeDof * j + 1], vecDisp[self.nodeDof * j + 2])
                    )
                )
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXDisp = str(format(vecDisp[self.nodeDof * j], floatDigits).rjust(columNum))
                strYDisp = str(format(vecDisp[self.nodeDof * j + 1], floatDigits).rjust(columNum))
                strZDisp = str(format(vecDisp[self.nodeDof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXDisp + strYDisp + strZDisp + "\n")
            f.write("\n")

            # 応力
            f.write("***** Stress Data ******\n")
            f.write(
                "Element No".rjust(columNum) +
                "Integral No".rjust(columNum) +
                "Stress XX".rjust(columNum) +
                "Stress YY".rjust(columNum) +
                "Stress ZZ".rjust(columNum) +
                "Stress XY".rjust(columNum) +
                "Stress XZ".rjust(columNum) +
                "Stress YZ".rjust(columNum) +
                "Mises".rjust(columNum) +
                "YieldFlg".rjust(columNum) + "\n"
            )
            for elemOutputData in self.elemOutputDataList[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strStressXX = str(format(elemOutputData.vecIpStressList[j].xx, floatDigits).rjust(columNum))
                    strStressYY = str(format(elemOutputData.vecIpStressList[j].yy, floatDigits).rjust(columNum))
                    strStressZZ = str(format(elemOutputData.vecIpStressList[j].zz, floatDigits).rjust(columNum))
                    strStressXY = str(format(elemOutputData.vecIpStressList[j].xy, floatDigits).rjust(columNum))
                    strStressXZ = str(format(elemOutputData.vecIpStressList[j].xz, floatDigits).rjust(columNum))
                    strStressYZ = str(format(elemOutputData.vecIpStressList[j].yz, floatDigits).rjust(columNum))
                    strMises = str(format(elemOutputData.ipMiseses[j], floatDigits).rjust(columNum))
                    strYieldFlg = str(elemOutputData.yieldFlgList[j]).rjust(columNum)
                    f.write(
                        strElemNo +
                        strIntNo +
                        strStressXX +
                        strStressYY +
                        strStressZZ +
                        strStressXY +
                        strStressXZ +
                        strStressYZ +
                        strMises +
                        strYieldFlg + "\n"
                    )
            
            f.write("\n")

            # 全ひずみ
            f.write("***** Strain Data ******\n")
            f.write(
                "Element No".rjust(columNum) +
                "Integral No".rjust(columNum) +
                "Strain XX".rjust(columNum) +
                "Strain YY".rjust(columNum) +
                "Strain ZZ".rjust(columNum) +
                "Strain XY".rjust(columNum) +
                "Strain XZ".rjust(columNum) +
                "Strain YZ".rjust(columNum) + "\n"
            )
            for elemOutputData in self.elemOutputDataList[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strStrainXX = str(format(elemOutputData.vecIpStrainList[j].xx, floatDigits).rjust(columNum))
                    strStrainYY = str(format(elemOutputData.vecIpStrainList[j].yy, floatDigits).rjust(columNum))
                    strStrainZZ = str(format(elemOutputData.vecIpStrainList[j].zz, floatDigits).rjust(columNum))
                    strStrainXY = str(format(elemOutputData.vecIpStrainList[j].xy, floatDigits).rjust(columNum))
                    strStrainXZ = str(format(elemOutputData.vecIpStrainList[j].xz, floatDigits).rjust(columNum))
                    strStrainYZ = str(format(elemOutputData.vecIpStrainList[j].yz, floatDigits).rjust(columNum))
                    f.write(
                        strElemNo +
                        strIntNo +
                        strStrainXX +
                        strStrainYY +
                        strStrainZZ +
                        strStrainXY +
                        strStrainXZ +
                        strStrainYZ + "\n"
                    )
            
            f.write("\n")

            # 塑性ひずみ
            f.write("***** Plastic Strain Data ******\n")
            f.write(
                "Element No".rjust(columNum) +
                "Integral No".rjust(columNum) +
                "PStrain XX".rjust(columNum) +
                "PStrain YY".rjust(columNum) +
                "PStrain ZZ".rjust(columNum) +
                "PStrain XY".rjust(columNum) +
                "PStrain XZ".rjust(columNum) +
                "PStrain YZ".rjust(columNum) + "\n"
            )
            for elemOutputData in self.elemOutputDataList[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strPStrainXX = str(format(elemOutputData.vecIpPStrainList[j].xx, floatDigits).rjust(columNum))
                    strPStrainYY = str(format(elemOutputData.vecIpPStrainList[j].yy, floatDigits).rjust(columNum))
                    strPStrainZZ = str(format(elemOutputData.vecIpPStrainList[j].zz, floatDigits).rjust(columNum))
                    strPStrainXY = str(format(elemOutputData.vecIpPStrainList[j].xy, floatDigits).rjust(columNum))
                    strPStrainXZ = str(format(elemOutputData.vecIpPStrainList[j].xz, floatDigits).rjust(columNum))
                    strPStrainYZ = str(format(elemOutputData.vecIpPStrainList[j].yz, floatDigits).rjust(columNum))
                    f.write(
                        strElemNo +
                        strIntNo +
                        strPStrainXX +
                        strPStrainYY +
                        strPStrainZZ +
                        strPStrainXY +
                        strPStrainXZ +
                        strPStrainYZ + "\n"
                    )
            
            f.write("\n")

            # 反力
            f.write("***** Reaction Force Data ******\n")
            f.write(
                "NodeNo".rjust(columNum) +
                "Magnitude".rjust(columNum) +
                "X Force".rjust(columNum) +
                "Y Force".rjust(columNum) +
                "Z Force".rjust(columNum)
            )
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecRF = self.vecRFList[i]
                mag = np.linalg.norm(
                    np.array(
                        (vecRF[self.nodeDof * j], vecRF[self.nodeDof * j + 1], vecRF[self.nodeDof * j + 2])
                    )
                )
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXForce = str(format(vecRF[self.nodeDof * j], floatDigits).rjust(columNum))
                strYForce = str(format(vecRF[self.nodeDof * j + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecRF[self.nodeDof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXForce + strYForce + strZForce + "\n")
            f.write("\n")
        f.close()
