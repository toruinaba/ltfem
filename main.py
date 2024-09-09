import copy
from Node import Node
from MaterialSet import MaterialSet
from MisesMaterial import Multiline_isotropic_hardening, Linear_isotropic_hardening
from Boundary import Boundary
from FEM import FEM
from C3D8 import C3D8

def main():
    node1 = Node(1, 0.0, 0.0, 0.0)
    node2 = Node(2, 1.0, 0.0, 0.0)
    node3 = Node(3, 2.0, 0.0, 0.0)
    node4 = Node(4, 3.0, 0.0, 0.0)
    node5 = Node(5, 0.0, 0.0, 1.0)
    node6 = Node(6, 1.0, 0.0, 1.0)
    node7 = Node(7, 2.0, 0.0, 1.0)
    node8 = Node(8, 3.0, 0.0, 1.0)
    node9 = Node(9, 0.0, 1.0, 0.0)
    node10 = Node(10, 1.0, 1.0, 0.0)
    node11 = Node(11, 2.0, 1.0, 0.0)
    node12 = Node(12, 3.0, 1.0, 0.0)
    node13 = Node(13, 0.0, 1.0, 1.0)
    node14 = Node(14, 1.0, 1.0, 1.0)
    node15 = Node(15, 2.0, 1.0, 1.0)
    node16 = Node(16, 3.0, 1.0, 1.0)
    nodes = [
        node1, node2, node3, node4, node5, node6, node7, node8,
        node9, node10, node11, node12, node13, node14, node15, node16
    ]
    nodes1 = [node1, node2, node10, node9, node5, node6, node14, node13]
    nodes2 = [node2, node3, node11, node10, node6, node7, node15, node14]
    nodes3 = [node3, node4, node12, node11, node7, node8, node16, node15]

    young = 210000.0
    poisson = 0.3
    mat = Linear_isotropic_hardening(young, poisson, 400, 50000.0)
    matSet = MaterialSet("None", young, poisson, material=mat)

    elem1 = C3D8(1, nodes1, matSet)
    elem2 = C3D8(2, nodes2, matSet)
    elem3 = C3D8(3, nodes3, matSet)
    elems = [elem1, elem2, elem3]
    elems_s = copy.deepcopy(elems)
    # 荷重制御
    """
    bound1 = Boundary(len(nodes))
    bound1.addSPC(1, 0.0, 0.0, 0.0)
    bound1.addSPC(5, 0.0, 0.0, 0.0)
    bound1.addSPC(9, 0.0, 0.0, 0.0)
    bound1.addSPC(13, 0.0, 0.0, 0.0)
    bound1.addForce(4, 0.0, 10.0, 0.0)
    bound1.addForce(8, 0.0, 10.0, 0.0)
    bound1.addForce(12, 0.0, 10.0, 0.0)
    bound1.addForce(16, 0.0, 10.0, 0.0)

    incNum1 = 100

    fem1 = FEM(nodes, elems, bound1, incNum1)
    fem1.add_curve_point(0.0, 0.0)
    fem1.add_curve_point(1.0, 1.0)

    fem1.impAnalysis()

    """
    # 変位制御
    bound2 = Boundary(len(nodes))
    bound2.addSPC(1, 0.0, 0.0, 0.0)
    bound2.addSPC(5, 0.0, 0.0, 0.0)
    bound2.addSPC(9, 0.0, 0.0, 0.0)
    bound2.addSPC(13, 0.0, 0.0, 0.0)
    bound2.addSPC(4, None, 0.1, None)
    bound2.addSPC(8, None, 0.1, None)
    bound2.addSPC(12, None, 0.1, None)
    bound2.addSPC(16, None, 0.1, None)

    incNum2 = 100
    fem2 = FEM(nodes, elems_s, bound2, incNum2)
    fem2.add_curve_point(0.0, 0.0)
    fem2.add_curve_point(0.1, 0.5)
    fem2.add_curve_point(0.3, -0.5)
    fem2.add_curve_point(0.6, 1.0)
    fem2.add_curve_point(1.0, -1.0)
    

    fem2.impAnalysis()


    #rfs1 = [0]
    #disps1 = [0]

    rfs2 = [0]
    disps2 = [0]
    """
    for i in range(incNum1):
        rf1 = (
            fem1.vecRFList[i][fem1.nodeDof * (1 - 1) + 1] +
            fem1.vecRFList[i][fem1.nodeDof * (5 - 1) + 1] +
            fem1.vecRFList[i][fem1.nodeDof * (9 - 1) + 1] +
            fem1.vecRFList[i][fem1.nodeDof * (13 - 1) + 1]
        )
        disp1 =  (
            fem1.vecDispList[i][fem1.nodeDof * (4 - 1) + 1] +
            fem1.vecDispList[i][fem1.nodeDof * (8 - 1) + 1] +
            fem1.vecDispList[i][fem1.nodeDof * (12 - 1) + 1] +
            fem1.vecDispList[i][fem1.nodeDof * (16 - 1) + 1]
        ) / 4
        rfs1.append(rf1)
        disps1.append(disp1)

    """
    for i in range(incNum2):
        rf2 = -(
            fem2.vecRFList[i][fem2.nodeDof * (1 - 1) + 1] +
            fem2.vecRFList[i][fem2.nodeDof * (5 - 1) + 1] +
            fem2.vecRFList[i][fem2.nodeDof * (9 - 1) + 1] +
            fem2.vecRFList[i][fem2.nodeDof * (13 - 1) + 1]
        )
        disp2 = (
            fem2.vecDispList[i][fem2.nodeDof * (4 - 1) + 1] +
            fem2.vecDispList[i][fem2.nodeDof * (8 - 1) + 1] +
            fem2.vecDispList[i][fem2.nodeDof * (12 - 1) + 1] +
            fem2.vecDispList[i][fem2.nodeDof * (16 - 1) + 1]
        ) / 4
        rfs2.append(rf2)
        disps2.append(disp2)


    from matplotlib import pyplot as plt

    fig = plt.figure()
    #plt.plot(disps1, rfs1, marker="o")
    plt.plot(disps2, rfs2, marker="v")
    #plt.plot(p_strains, stresses)
    #plt.plot(mat.p_strains, mat.stresses, ":")
    plt.show()

    fem1.outputTxt("test")

if __name__ == "__main__":
    main()