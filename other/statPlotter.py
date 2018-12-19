#code by Megan Charity
#Plots the energy and time points from project 3

import matplotlib.pyplot as plt
import numpy as np

multi_plot = 1
show_graph = 1

testType = input("Testing type: ")

#setup data
num_lines = 1
data_file = input(("%d line to read in: " % num_lines))

lineDatSet = {}
while(data_file != "-1"):
	#setup data arrays
	epoch, mcc = [], []
	lineDat = {}

	#open and read the file
	file = open(data_file, "r");
	lines = file.readlines();

	#parse the data
	# epoch = first value
	# mcc = fifth value
	print("\tParsing data...")
	for line in lines[1:]:		#skip the first line (label line)
		line.strip()
		dataSet = line.split('\t')
		epoch.append(float(dataSet[0]))
		mcc.append(float(dataSet[5]))

	#add the set to the main data set
	setname = data_file.split("_")[0] + "_" + data_file.split("_")[1] 
	lineDat["name"] = setname
	lineDat["epoch"] = epoch
	lineDat["mcc"] = mcc
	lineDatSet[num_lines] = lineDat

	#get another line
	num_lines+=1
	data_file = input(("%d line to read in: " % num_lines))


print("\n\n\n")

#show the data on the graph
if(multi_plot):
	print("Showing multiple data lines on mini graphs...")

	# Initialize the figure
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	palette = plt.get_cmap('Set1')

	#multiple line plt
	num=0
	for key in lineDatSet.keys():
		#grab sets
		name = lineDatSet[key]["name"]
		x = lineDatSet[key]["epoch"]
		y = lineDatSet[key]["mcc"]

		num+=1
		plt.subplot(12,1, num)
		plt.plot(x, y, color=palette(num), alpha=0.9, label=name)

		#set the limits
		plt.xlim(0,100)
		plt.ylim(0,1)

		# Not ticks everywhere
		if num in range(7) :
			plt.tick_params(labelbottom='off')
		if num not in [1,3,6] :
			plt.tick_params(labelleft='off')
		

		# Add title
		plt.title(name, loc='left', fontsize=12, fontweight=0, color=palette(num) )

	# general title
	plt.suptitle(testType, fontsize=13, fontweight=0, color='black', style='italic')
 
	# Axis title
	#plt.text(0.5, 0.02, 'Epoch', ha='center', va='center')
	#plt.text(0.06, 0.5, 'MCC', ha='center', va='center', rotation='vertical')
	

	plt.legend(loc='upper left')
	plt.savefig((testType + ".png"))

	if(show_graph):
		plt.show()

else:
	print("Showing multiple data lines on same graph...")
	#fig = plt.figure()
	# Initialize the figure
	plt.style.use('seaborn-darkgrid')

	for key in lineDatSet.keys():
		plt.plot(lineDatSet[key]["epoch"], lineDatSet[key]["mcc"], label=lineDatSet[key]["name"])
	
	plt.legend(loc='right')
	plt.savefig((testType + ".png"))

	plt.xlabel('EPOCH')
	plt.ylabel('MCC')
	plt.title("Epoch vs. MCC")

	if(show_graph):
		plt.show()

