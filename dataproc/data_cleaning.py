import re
from datetime import datetime
import calendar
import os

def getExDay(year, month, no=4, w=3):
	fweek,dayr=calendar.monthrange(year, month)
	tt=0
	if w<fweek: tt=7
	return tt+w-fweek+(no-1)*7

def raw2csv(filename, savedfilename):
	f=open(filename, 'r')
	out=open(savedfilename, 'w')
	out.write("date, opening, high, low, closing, volume, turnover\n")
	line=f.readline().strip()
	line = f.readline().strip()
	line = f.readline().strip()
	while line:
		eles=line.split(',')
		if(len(eles)<=1): break
		out.write(line+'\n')
		line=f.readline().strip()
	f.close()
	out.close()

if __name__=='__main__':
	path="../data/stocks/"
	filenames=os.listdir(path+"raw/")
	for filename in filenames:
		file=path+"raw/"+filename
		if os.path.isfile(file):
			raw2csv(file, path+filename)
			print(filename)
