from model.basic_decision import BasicDecision
from ctrl.option_trade import OptionOrder

class SimpleDecision(BasicDecision):
	def __init__(self):
		super(SimpleDecision, self).__init__()
		self.optionCode='10000845'
		self.buyThreshold=0.01
		self.lossThreshold=600

	def decide(self, optionTrade):
		super(SimpleDecision, self).decide(optionTrade)
		closeOrder, openOrders=[], []
		for order in self.trade.orders:
			if(self.controlLoss(order)): closeOrder.append(order.id)
		actualmoney=self.trade.curmoney+0
		option=self.trade.getOptionFCode(self.optionCode)
		if self.judgeBuy(option):
			n=actualmoney//(option.getLastPrice()*OptionOrder.hand2gu)
			if n: openOrders.append([self.optionCode, 0, n/2])
		return closeOrder, openOrders

	def judgeBuy(self, option):
		if len(option.info)<2: return False
		now=low=len(option.info)-1
		while(low>0 and option.info[low].closingPrice>option.info[low-1].closingPrice):
			low-=1
		if low==0: return False
		if (option.info[now].closingPrice-option.info[low].closingPrice)>=self.buyThreshold: return True
		return False
			