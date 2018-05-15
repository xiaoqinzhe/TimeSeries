from obj.option import Option
from obj.etfund import ETFund
from ctrl.visual_market import VMarket
from util import util
from datetime import datetime
import time
from ctrl.trade_recorder import TradeRecorder



#asset order
class AssetOrder:
	uid=0
	str_title='id order_time asset_code optype price amount service_charge pay'
	def __init__(self, orderTime, assetCode, optype, curPrice, amount, serviceCharge):
		self.id=AssetOrder.uid
		AssetOrder.uid+=1
		self.orderTime=orderTime       #order time
		self.assetCode=assetCode      #which asset
		self.optype=optype             #operation type: 0 for buy, 1 for sell
		self.curPrice=curPrice         #current asset price
		self.amount=amount             #number of shou option
		self.serviceCharge=serviceCharge    #fu wu fei
		self.totalCost=0.0        #total cost
		if(self.optype==0): self.totalCost=self.amount*self.curPrice+self.serviceCharge
		elif(self.optype==1): self.totalCost=-self.amount*self.curPrice+self.serviceCharge
		self.remainMoney=0.0
		self.curEarning=0.0       #cur earning if sell

	def toString(self):
		return '%d %s %s %d %f %d %f %f' % (self.id, util.strftime(self.orderTime), self.optionCode, \
			self.optype, self.curPrice, self.amount, self.serviceCharge, -self.totalCost)

is_simulated=True
sim_interval=[1, 'd']
sim_st=datetime(2017, 1, 26)

#asset trade
class AssetTrade:
	def __init__(self, decision, tradeinterval=[1, 'd'], curtime=None, sim=True, sim_st=None,):
		self.is_simulated=sim
		self.assets=[]                    #assets
		self.curtime=curtime                     #current time		
		self.interval=tradeinterval   #time interval of trade
		#if(sim): self.curtime=sim_st-self.tradeintercal
		self.market=VMarket(self.interval, self.curtime)              #trade market
		if self.curtime==None: self.curtime=market.curtime
		self.orders=[]                     #asset orders
		self.oldorders=[]
		self.decision=decision             #making decision
		self.serviceCharge=0.0
		self.initmoney=10000.0                 #init investment money
		self.curmoney=self.initmoney                  #current remain money		
		self.profit=0.0                    #profit
		self.roreturn=0.0                  #rate of return
		self.alrProfit=0.0                 #已经得到的利润
		self.inmoneyp=0.0                  #investment money in a period
		self.outmoneyp=0.0                 #return money in a period
		self.sharpeRatio=0.0
		self.recorder=TradeRecorder()      #recoder
		self.initData()

	def initData(self):
		self.recorder.initmoney=self.initmoney

	def run(self):
		while(True):
			#update current time
			if not self.is_simulated: pass
			else:
				day=self.curtime.day
				self.curtime=self.market.curtime
			if self.market.endFlag: break
			#if not market.judgeTradeDay(curtime): continue
			#update option and etf info from market (prices...)
			self.updateFromMarket()
			#calculate period return
			if self.curtime.day!=day: self.periodProfit()
			#make decision and handle
			closeOrders, decOrders=self.decision.decide(self)
			# return format of decide(): [id1, id2, ...], [[code, optype, amount]]
			self.handleTrade(closeOrders, decOrders)
		self.closeAllOrders()
		self.recorder.finalmoney=self.curmoney
		print(self.curtime, self.profit, self.curmoney)
		self.recorder.showOrders()
		self.recorder.showDailyProfits()
		self.recorder.eval()
		self.recorder.showStatistics()

	def periodProfit(self):
		self.recorder.tradeDays+=1
		p=self.calculateTotalProfit()
		self.outmoneyp=p-self.profit
		self.profit=p
		for order in self.orders:
			if order.optype==0: self.inmoneyp+=order.totalCost
			else: self.inmoneyp-=order.totalCost
		#date=datetime()
		self.recorder.addProfit(self.curtime, self.outmoneyp, self.inmoneyp)
		self.inmoneyp=0.0
		self.roreturn=self.profit/self.initmoney

	def closeAllOrders(self):
		i=0
		while i<len(self.orders):
			self.closeOrder(self.orders[i])

	def closeExpOptions(self):
		i=0
		while(i<len(self.options)):
			op=self.options[i]
			if op.expirationDate.year==self.curtime.year and \
			op.expirationDate.month==self.curtime.month and op.expirationDate.day-1==self.curtime.day:
				self.closeOptionOrders(op)
				self.options.remove(op)
				self.expOptions.append(op)
			else: i+=1

	def closeOptionOrders(self, option):
		i=0
		while i<len(self.orders):
			order=self.orders[i]
			if order.optionCode==option.code: 
				self.closeOrder(order)
			else: i+=1

	def isCloseExpDay(self):
		now=datetime.now()
		day=util.getExDay(now.year, now.month)
		if day==now.day-1: return True
		return False

	def handleTrade(self, closeOrders, decOrders):
		for oid in closeOrders:
			order=self.getOrderFId(oid)
			if order: self.closeOrder(order)
		for do in decOrders:
			self.openOrder(do[0], do[1], do[2])
	
	#add order buy
	def openOrder(self, code, optype, amount):
		option=self.getOptionFCode(code)
		oo=OptionOrder(self.curtime, code, optype, option.getLastPrice(), amount, self.serviceCharge)
		if optype==0 and self.curmoney<oo.totalCost: return False
		self.curmoney-=oo.totalCost
		self.orders.append(oo)
		self.recorder.addOrder(oo)
		oo.remainMoney=self.curmoney
		self.recorder.lots+=1
		return True

	# ping cang order
	def closeOrder(self, corder):
		if corder.optype==0: optype=1
		else: optype=0
		option=self.getOptionFCode(corder.optionCode)
		oo=OptionOrder(self.curtime, corder.optionCode, optype, option.getLastPrice(), \
			corder.amount, self.serviceCharge)
		if optype==0 and self.curmoney<oo.totalCost: return False
		self.alrProfit+=(-corder.totalCost-oo.totalCost)
		if self.curtime.hour!=0 and (self.curtime.hour!=9 or self.curtime.minutes!=30): 
			if corder.optype: self.inmoneyp-=corder.totalCost
			else:self.inmoneyp+=corder.totalCost
		self.curmoney-=oo.totalCost
		self.orders.remove(corder)
		self.oldorders.append(corder)
		self.oldorders.append(oo)
		self.recorder.addOrder(oo)
		oo.remainMoney=self.curmoney
		if (-corder.totalCost-oo.totalCost)>0: self.recorder.profitTime+=1
		else: self.recorder.lossTime+=1
		return True

	def getOrderFId(self, oid):
		for o in self.orders:
			if o.id==oid:
				return o
		return None

	#get next new data
	def updateFromMarket(self):
		mops, me=self.market.getData(self.curtime)
		if self.etf==None: self.etf=me
		else: self.etf.info.append(me.info[0])
		for op in mops:
			if self.getExpOptionFCode(op.code): continue
			re=self.getOptionFCode(op.code)
			if re:
				re.info.append(op.info[0])
			else: self.assets.append(op)	

	def getAssetFCode(self, code):
		for ass in self.assets:
			if ass.code==code:
				return ass
		return None
	
	def getExpOptionFCode(self, code):
		for op in self.expOptions:
			if op.code==code:
				return op
		return None
	
	def calculateTotalProfit(self):
		profits=0
		for order in self.orders:
			cost, _=self.orderProfit(order)
			profits+=cost
		profits=profits+self.curmoney-self.initmoney
		return profits

	def orderProfit(self, order):
		if order.optype==0: optype=1
		else: optype=0
		option=self.getOptionFCode(order.optionCode)
		if optype==0: 
			cost=option.getLastPrice()*order.amount*OptionOrder.hand2gu+self.serviceCharge
		else: cost=-option.getLastPrice()*order.amount*OptionOrder.hand2gu+self.serviceCharge
		#print('**', order.optionCode, order.totalCost, cost, '**')
		order.earning=order.totalCost+cost

		return -cost, order.earning
