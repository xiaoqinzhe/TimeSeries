import matplotlib as mpl
import matplotlib.pyplot as plt
from obj.option import Option
from obj.etfund import ETFund
import config
import os, operator
import ctrl.datadb as datadb

mpl.rcParams['font.sans-serif'] = ['SimHei']
#mpl.rcParams['xtick.labelsize']=12
#mpl.rcParams['ytick.labelsize']=12

vdatetime_format = '%Y/%m/%d %H:%M:%S'
vdatetime_formats = {60: '%Y%m%d %H%M%S', 300: '%Y%m%d %H%M%S', 24*60*60: '%Y%m%d'}

#plot, set_x/ylabel, x/y_lim, title

prices_default_colors={0: 'g', 1: 'y', 2: 'k', 3: 'r'}
prices_default_labels={0: 'openingPrice', 1: 'topPrice', 2: 'bottomPrice', 3: 'closingPrice'}

class Visualization:
	def __init__(self):
		self.x=[]
		self.y=[]
		self.z=[]

	def axFProduct(self, product, ax, labels=prices_default_labels, colors=None, pricel=[3]):
		x, y = [], [[],[],[],[]]
		#print(self.y)
		fyear, fmonth, fday = 0, 0, 0
		for info in product.info:
			# form = ''
			# if(info.datetime.year > fyear): 
			# 	fyear = info.datetime.year
			# 	form+='%Y'
			# if(info.datetime.month > fmonth): 
			# 	fmonth = info.datetime.month
			# 	form+='%m'
			# if()
			#x.append(int(info.datetime.strftime(vdatetime_formats[product.timeInterval])))
			x.append(info.datetime)
			for i in pricel:
				if(i==0): y[i].append(info.openingPrice)
				elif(i==1): y[i].append(info.topPrice)
				elif(i==2): y[i].append(info.bottomPrice)
				else: y[i].append(info.closingPrice)
		for i in pricel: 
			if(colors): ax.plot(x, y[i], colors[i], label=labels[i])
			else: ax.plot(x, y[i], label=labels[i])
		#ax.plot(x, y[0])
		#ax.set_title(product.name)

	def showFProduct(self, product):
		a, b=plt.subplots()
		self.axFProduct(product, b)
		plt.show()

	def showFProducts(self, products):
		for product in products:
			self.showOption(self, product)

	def showOptionAAsset(self, option, asset):
		a, b=plt.subplots()
		self.axFProduct(option, b, labels={3: option.name}, colors={3: 'r'})
		b.legend(loc=2)
		bw=b.twinx()
		self.axFProduct(asset, bw, labels={3: asset.name}, colors={3: 'b'})
		bw.legend(loc=1)
		plt.show()

	def showOptionsAAsset(self, options, asset):
		#dcolors=('b', 'g', 'c', 'm', 'y', 'k')
		#if(len(options)>len(dcolors)): raise Exception('number of colors is not enough')
		a, b=plt.subplots()
		for i, option in enumerate(options):	
			self.axFProduct(option, b, labels={3: options[i].name}, colors=None)
		b.legend(loc=2)
		bw=b.twinx()
		self.axFProduct(asset, bw, labels={3: asset.name}, colors={3: 'k'})
		bw.legend(loc=1)
		plt.show()

	def saveFileInfo(self, options, etfs):
		options.sort(key=operator.attrgetter('expirationDate'))
		out = open(config.configures['dataPath']+'options_info.dat', 'w')
		for i, option in enumerate(options):
			be,end=option.getRangeTime()
			out.write('%d %s %s %s (%s, %s)\n' % (i+1, option.code, option.name, option.expirationDate.strftime('%Y/%m/%d'), be, end))
		out.close()
		out = open(config.configures['dataPath']+'etfs_info.dat', 'w')
		for i, product in enumerate(products):
			be,end=product.getRangeTime()
			out.write('%d %s %s (%s, %s)\n' % (i+1, product.code, product.name, be, end))
		out.close()

	def printOptionsInfo(self, products, filename = 'options_info.dat'):
		options.sort(key=operator.attrgetter('expirationDate'))
		for i, product in enumerate(products):
			be,end=product.getRangeTime()
			print(i+1, product.code, product.name, product.expirationDate.strftime('%Y/%m/%d'), product.getRangeTime())

	def printETFInfo(self, products, filename = 'etfs_info.dat'):
		for i, product in enumerate(products):
			print(i+1, product.code, product.name, product.getRangeTime())
			be,end=product.getRangeTime()

	def showAllProducts(self, options, etfs):		
		self.printOptionsInfo(options)	
		self.printETFInfo(etfs)

	def compareOptions(self, option1, option2):
		print(option1.name, option2.name)
		difs=[]
		datet=[]
		for i in range(len(option1.info)):
			dif=option1.info[i].closingPrice-option2.info[i].closingPrice
			difs.append(dif)
			datet.append(option1.info[i].datetime)
			print(dif)
		a,b=plt.subplots()
		b.plot(datet, difs, label='dif', color='k')
		b.legend(loc=2)
		c=b.twinx()
		self.axFProduct(option1,c,labels={3:option1.name})
		self.axFProduct(option2,c,labels={3:option2.name})
		c.legend(loc=1)
		plt.show()

	def showPrices(self, option, asset):
		st=option.info[0].datetime
		for i in range(len(asset.info)):
			if asset.info[i].datetime==st: break
		interp=[]
		timep=[]
		x=[]
		for j in range(len(option.info)):
			ip=max(0, asset.info[i].closingPrice-option.strikePrice)
			tp=option.info[j].closingPrice-ip
			#print(asset.info[i].datetime, option.info[j].datetime)
			print(ip, tp)
			interp.append(ip)
			timep.append(tp)
			x.append(option.info[j].datetime)
			i+=1
		a,b=plt.subplots()
		b.plot(x, interp, label='内在价值')
		b.plot(x, timep, label='时间价值')
		self.axFProduct(option, b, labels={3:option.name}, colors={3:'k'})
		b.legend(loc=0)
		plt.show()

	def showDelta(self, option, asset):
		st=option.info[0].datetime
		for i in range(len(asset.info)):
			if asset.info[i].datetime==st: break
		interp=[]
		timep=[]
		x=[]
		rate=[]
		delta=[]
		for j in range(len(option.info)-1):
			ip=asset.info[i+1].closingPrice-asset.info[i].closingPrice
			tp=option.info[j+1].closingPrice-option.info[j].closingPrice
			#print(asset.info[i].datetime, option.info[j].datetime)
			print(asset.info[i+1].datetime, option.info[j+1].datetime, ip, tp)
			interp.append(ip)
			timep.append(tp)
			rate.append(tp-ip)
			if ip: delta.append(tp/ip)
			else: delta.append(ip)
			x.append(option.info[j].datetime)
			i+=1
		a,b=plt.subplots()
		b.plot(x, interp, label='标的价格变化')
		b.plot(x, timep, label='期权价格变化')
		b.plot(x, rate, label='时间价值变化')
		b.legend(loc=2)
		c=b.twinx()
		c.plot(x, delta, label='delta', color='r')
		c.legend(loc=1)
		plt.show()

	def show

def test():
	options=datadb.getOptions()
	options.sort(key=operator.attrgetter('expirationDate'))
	etf=datadb.get50ETF()
	vis=Visualization()
	#vis.showAllProducts(options, etfs)
	selOp = [0,1,10,20,30,31,40,42,45,46,60,70,72,76,77,84,86]
	#selOp = [5,6,11,13,37,38,41,43,51,52,57,61,81,82,87,89]
	vops=[]
	for i in selOp:
		vops.append(options[i])
	vis.showOptionsAAsset(vops, etf)
	#vis.compareOptions(options[0],options[1])
	#vis.showPrices(options[1], etf)
	vis.showDelta(vops[2], etf)