class Node:
    def __init__(self, no, x, y, z):
        self.no = no
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"<Node{self.no}:x{self.x}-y{self.y}-z{self.z}>"