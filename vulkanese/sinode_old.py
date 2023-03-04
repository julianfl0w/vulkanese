import pickle


class Node(object):
    def __init__(self):
        self.children = []

    def __str__(self):
        retstr = ""
        retstr += str(type(self)) + "\n"
        if hasattr(self, "name"):
            retstr += "  " + self.name + "\n"
        for child in self.children:
            retstr += "  " + str(child).replace("\n", "  \n") + "\n"

        return retstr

    def asDict(self):
        retList = []
        if hasattr(self, "name"):
            retList += [self.name]

        for child in self.children:
            try:
                retList += [child.asDict()]
            except:
                retList += [str(child)]  # + " " + hex(id(child))]

        retDict = {}
        retDict[str(type(self))] = retList
        return retDict

    def getSiblings(self):
        return self.parent.children

    def descendants(self):
        d = self.children
        for c in self.children:
            d += c.descendants()
        return d

    def getAncestor(self, ancestorType):
        if not hasattr(self, "parent"):
            return false
        if ancestorType in str(type(self.parent)).lower():
            return self
        return self.parent.getAncestor(ancestorType)

    def __unicode__(self):
        return self.__str__()

    def release(self):
        for child in self.children:
            # print("PK releasing " + str(child))
            child.release()

    def toFile(self, filename):
        # open a file, where you ant to store the data
        with open(filename, "wb+") as f:
            # dump information to that file
            pickle.dump(self, f)


def NodeFromFile(filename):
    # open a file, where you ant to store the data
    with open(filename, "rb") as f:
        # dump information to that file
        a = pickle.load(f)
    return a


class Sinode(Node):
    def __init__(self, parent=None):
        Node.__init__(self)
        self.parent = parent  # enforce single inheritance
        if not hasattr(self, "index"):
            self.index = 0
        # accumulate path
        if parent is not None:
            self.ancestors = parent.ancestors + [parent]
            self.path = parent.path + [self.index]
            self.parent.children += [self]
        else:
            self.ancestors = []
            self.path = [self.index]

        self.apex = self.getApex()

    def getDecendentGenerationCount(self):
        if self.children == []:
            return 0
        else:
            return max([c.getHeight() for c in children])

    def getHeight(self):
        return getDecendentGenerationCount(self)

    def getAncestors(self):
        if self.parent is None:
            return [self]
        else:
            return self.parent.getAncestors() + [self]

    def getAncestor(self, ancestorType):
        print("checking " + str(self))
        print("type " + str(type(self)))
        if not hasattr(self, "parent"):
            return false
        if ancestorType in str(type(self)).lower():
            return self
        return self.parent.getAncestor(ancestorType)

    def isApex(self):
        return self.parent is None

    def getApex(self):
        if self.parent is None:
            return self
        return self.parent.getApex()


class Minode(Node):
    def __init__(self, parents):
        self.parents = parents  # allow multi inheritance through a set
        self.children = []

    def getAncestor(self, ancestorType):
        if not hasattr(self, "parent"):
            return false
        if str(type(self.parent)) == ancestorType:
            return self
        return self.parent.getAncestor(ancestorType)
