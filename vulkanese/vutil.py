class Node(object):
	def __init__(self):
		self.children = []
		
	def __str__(self):
		retstr = str(type(self)) + "\n"
		for child in self.children:
			retstr += "  " + str(child).replace("\n", "  \n") + "\n"
		
		return retstr
	
	def asDict(self):
		retList = []
		for child in self.children:
			try:
				retList += [child.asDict()]
			except:
				retList += [str(child)] # + " " + hex(id(child))]
				
		retDict = {}
		retDict[str(type(self))] = retList
		return retDict
		
	def getSiblings(self):
		return self.parent.children
	
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
			#print("PK releasing " + str(child))
			child.release()


class Sinode(Node):
	def __init__(self, parent):
		Node.__init__(self)
		self.parent   = parent # enforce single inheritance
		
	def getAncestor(self, ancestorType):
		print("checking " + str(self))
		print("type " + str(type(self)))
		if not hasattr(self, "parent"):
			return false
		if ancestorType in str(type(self)).lower():
			return self
		return self.parent.getAncestor(ancestorType)
		

class Minode(Node):
	def __init__(self, parents):
		self.parents  = parents # allow multi inheritance through a set
		self.children = []
		
	def getAncestor(self, ancestorType):
		if not hasattr(self, "parent"):
			return false
		if str(type(self.parent)) == ancestorType:
			return self
		return self.parent.getAncestor(ancestorType)
		

import inspect
import os
def jlog(instr):
	previous_frame = inspect.currentframe().f_back
	(filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)
	print(os.path.basename(filename) + ":" + str(line_number) + ": " + str(instr))
	