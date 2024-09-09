class ElementOutputData:
    def __init__(
        self,
        element,
        vecStressList,
        vecEStrainList,
        vecPStrainList,
        ePStrainList,
        misesList,
        yieldFlgList
    ):
        self.element = element
        self.vecIpPStrainList = vecPStrainList
        self.vecIpStrainList = [] # 全ひずみ
        for i in range(len(vecEStrainList)):
            vecIpStrain = vecEStrainList[i] + vecPStrainList[i]
            self.vecIpStrainList.append(vecIpStrain)
        self.ipEPStrainList = ePStrainList
        self.vecIpStressList = vecStressList
        self.ipMiseses = misesList
        self.yieldFlgList = yieldFlgList