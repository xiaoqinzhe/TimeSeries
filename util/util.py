import calendar, math
import copy

def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))

def arcsigmoid(x):
	return -math.log(1.0/x-1)

def movingAvg(seq, k):
	if len(seq)<k: return []
	res=[]
	for i in range(len(seq)-k+1):
		res.append(sum(seq[i:i+k])/k)
	return res

def scaleAvg(seq, k):
	if len(seq)<k: return []
	res=[]
	for i in range(len(seq)//k):
		res.append(sum(seq[i*k:i*k+k])/k)
	return res

def scaleAvgN(seq):
	res=[]
	n=len(seq[0])
	k=len(seq)
	for i in range(n):
		s=0.0
		for j in range(k):
			s+=seq[j][i]
		res.append(s/k)
	return res