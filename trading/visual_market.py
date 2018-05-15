from dataproc import datadb

class OpVMarket:
	def __init__(self, interval, curtime=None):
		self.curtime=curtime
		self.interval=interval
		self.notcurOptions=[]
		self.curOptions=[]
		self.etf=None
		self.endFlag=False
		self.initData()
		pass

	def initData(self):
		self.notcurOptions=datadb.getOptions(self.interval, self.curtime)
		self.etf=datadb.get50ETF(self.interval, self.curtime)
		self.curtime=self.etf.info[0].datetime
		self.initMarket(self.curtime)
		pass

	def getOptions(self):
		copys=[]
		for op in self.curOptions:
			copy=op.copy()
			copy.info.append(op.info[op.curi])
			copys.append(copy)
		return copys

	def getETF(self):
		e=self.etf.copy()
		e.info.append(self.etf.info[self.etf.curi])
		return e

	def getData(self, curtime):
		if self.endFlag: return None
		o, e=self.getOptions(), self.getETF()
		self.nextState()
		if not self.endFlag:
			self.curtime=self.etf.info[self.etf.curi].datetime
			self.joinOptions()
		return o, e

	def nextState(self):
		i=0
		while i<len(self.curOptions):
			self.curOptions[i].curi+=1
			if 	self.curOptions[i].curi==len(self.curOptions[i].info):
				self.curOptions.remove(self.curOptions[i])
			else: i+=1
		self.etf.curi+=1
		if(len(self.curOptions)==0): self.endFlag=True
		if self.etf.curi==len(self.etf.info): self.endFlag=True

	def judgeTradeDay(self, curtime):
		pass

	def initMarket(self, curtime):
		#join new options
		rops=[]
		for op in self.notcurOptions:
			if op.info[0].datetime<=curtime:
				self.curOptions.append(op)
				rops.append(op)
				op.seek(curtime)
		for rop in rops:
			self.notcurOptions.remove(rop)
		self.etf.seek(curtime)

	def joinOptions(self):
		#join new options
		jops=[]
		for op in self.notcurOptions:
			if op.info[0].datetime==self.curtime:
				self.curOptions.append(op)
				jops.append(op)
		for jop in jops:
			self.notcurOptions.remove(jop)

	def removeExpiredOptions(self):
		for op in self.curOptions:
			if(op.curi==len(op.info)-1): self.curOptions.remove(op)

class AsVMarket:
	def __init__(self, interval, curtime=None):
		self.curtime=curtime
		self.interval=interval
		self.notcurAssets=[]
		self.curAssets=[]
		self.endFlag=False
		self.initData()
		pass

	def initData(self):
		self.notcurOptions=datadb.getOptions(self.interval, self.curtime)
		self.etf=datadb.get50ETF(self.interval, self.curtime)
		self.curtime=self.etf.info[0].datetime
		self.initMarket(self.curtime)
		pass

	def getOptions(self):
		copys=[]
		for op in self.curOptions:
			copy=op.copy()
			copy.info.append(op.info[op.curi])
			copys.append(copy)
		return copys

	def getETF(self):
		e=self.etf.copy()
		e.info.append(self.etf.info[self.etf.curi])
		return e

	def getData(self, curtime):
		if self.endFlag: return None
		o, e=self.getOptions(), self.getETF()
		self.nextState()
		if not self.endFlag:
			self.curtime=self.etf.info[self.etf.curi].datetime
			self.joinOptions()
		return o, e

	def nextState(self):
		i=0
		while i<len(self.curOptions):
			self.curOptions[i].curi+=1
			if 	self.curOptions[i].curi==len(self.curOptions[i].info):
				self.curOptions.remove(self.curOptions[i])
			else: i+=1
		self.etf.curi+=1
		if(len(self.curOptions)==0): self.endFlag=True
		if self.etf.curi==len(self.etf.info): self.endFlag=True

	def judgeTradeDay(self, curtime):
		pass

	def initMarket(self, curtime):
		#join new options
		rops=[]
		for op in self.notcurOptions:
			if op.info[0].datetime<=curtime:
				self.curOptions.append(op)
				rops.append(op)
				op.seek(curtime)
		for rop in rops:
			self.notcurOptions.remove(rop)
		self.etf.seek(curtime)

	def joinOptions(self):
		#join new options
		jops=[]
		for op in self.notcurOptions:
			if op.info[0].datetime==self.curtime:
				self.curOptions.append(op)
				jops.append(op)
		for jop in jops:
			self.notcurOptions.remove(jop)

	def removeExpiredOptions(self):
		for op in self.curOptions:
			if(op.curi==len(op.info)-1): self.curOptions.remove(op)

