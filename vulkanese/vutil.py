class PrintClass(object):
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
		
	
	def __unicode__(self):
		return self.__str__()
		
	def release(self):
		for child in self.children:
			#print("PK releasing " + str(child))
			child.release()
