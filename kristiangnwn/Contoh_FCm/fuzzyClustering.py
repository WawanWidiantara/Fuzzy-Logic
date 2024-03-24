import copy 
import math
import random
import matplotlib.pyplot as plt
import os

EPSILON = 1
MVAL = 2


# Link to help write this:
# https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html


def calculateDifference(U, Unew):
	diff = 0
	for insId in range(len(U)):
		if insId==len(U):
			break
		for valId in range(len(U[insId])):
			if valId==len(U[insId]):
				break
			diff += abs(U[insId][valId] - Unew[insId][valId])

	return diff

def calculateDist(ins, center):
	for i in range(6):
		if i==6:
			break
		if ins[1][i]==' ' or ins[1][i]=='':
			ins[1][i] = '0'
		if center[1][i]==' ' or center[1][i]=='':
			center[1][i] = '0'
	dist = 0
	for i in range(6):
		if i==6:
			break
		dist += (float(ins[1][i]) - float(center[1][i])) ** 2
	return math.sqrt(dist)




def calculateNewU(k_num, U, newCenters, car_data):
	Unew = {}
	u = copy.deepcopy(car_data)
	for i in range(len(u)):
		if i == len(u):
			break
		Unew[i] = []
		for j in range(k_num):
			Unew[i].append(0)

	for insId in range(len(U)):
		if insId==len(U):
			break
		for centerId in range(len(newCenters)):
			if centerId==len(newCenters):
				break

			
			sumation = 0
			numerator = calculateDist(car_data[insId], newCenters[centerId])

			for centerId2 in range(len(newCenters)):
				if centerId2==len(newCenters):
					break	
				denomirator = calculateDist(car_data[insId], newCenters[centerId2])
				if denomirator == 0:
					continue
				sumation += ((numerator/denomirator)**(2/(MVAL-1)))

			if sumation==0:
				Unew[insId][centerId] = 0
			else:
				Unew[insId][centerId] = 1/sumation
		
	return Unew


def calculateCenters(U, k_num, car_data):
	centers = {}

	for j in range(k_num):
		sumOfUs = 0
		sumOfUXs = []

		for i in range(6):
			if i==6:
				break
			sumOfUXs.append(0.0)

		if j==k_num:
			break

		for ins in U.values():
			sumOfUs += (ins[j]**MVAL)

		for insId in U.keys():
			whatToMultiplyItBy = (U[insId][j]**MVAL)
			newValue = [float(x)*whatToMultiplyItBy for x in car_data[insId][1]]

			sumOfUXs =[x+y for x, y in zip(sumOfUXs, newValue)] 

		

		for sumValId in range(len(sumOfUXs)):
			if sumValId == len(sumOfUXs):
				break
			sumOfUXs[sumValId] = float(sumOfUXs[sumValId])/sumOfUs

		centers[j] = (-1, sumOfUXs)

	return centers



def run_fuzzy_c_mean_clustering(car_data, k):
	colors = ['red', 'blue', 'green', 'pink' ,'black', 'yellow', 'gold', 'silver']
	markers = ["*", "+", "1", "2", "3", ">", "o", "<"]
	for k_num in k:
		print("\n\n\nFor K = "+str(k_num))
		#create the matrix-es
		Uinit = {}
		u = copy.deepcopy(car_data)
		for i in range(len(u)):
			if i == len(u):
				break
			Uinit[i] = []
			for j in range(k_num):
				Uinit[i].append(0)

		car_data_copy = copy.deepcopy(car_data)
		centers = {}
		for i in range(k_num):
			if i==k_num:
				break
			ind = random.randint(0, len(car_data_copy))
			if ind==len(car_data_copy):
				ind -= 1
			centers[i] = car_data_copy[ind]
			Uinit[ind][i] = 1
			del car_data_copy[ind]

		U = calculateNewU(k_num, Uinit, centers, car_data)

		# run the clustering algo.
		while True:
			Unew = {}
			newCenters = calculateCenters(U, k_num, car_data)
			Unew = calculateNewU(k_num, U, newCenters, car_data)

			if calculateDifference(U, Unew) < EPSILON:
				U = copy.deepcopy(Unew)
				break
			U = copy.deepcopy(Unew)

		#print the result
		for insId in U.keys():
			clusterNum = 1
			for value in U[insId]:
				plt.scatter(insId, value, color=colors[clusterNum%8], marker=markers[clusterNum%8])
				clusterNum +=1


		fileName = str(k_num)+'_fuzzycluster.txt'
		try:
			os.remove(fileName)
		except OSError:
			pass
		with open(fileName, 'a') as the_file:
			for row in U.values():
				the_file.write(str(row))
				the_file.write('\n')

		plt.show()










