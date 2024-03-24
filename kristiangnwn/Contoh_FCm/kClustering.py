import random 
import copy 
import sys
import math
import matplotlib.pyplot as plt
import os


ITERATION = 200


def findMeanOfYearForCluster(cluster):
	if len(cluster)==0:
		return 0
	s = 0
	for ins in cluster:
		s += int(ins[1][-1])

	return int(s/len(cluster))

def clusterBasedOnYear(car_data, k):
	colors = ['red', 'blue', 'green', 'pink' ,'black', 'yellow', 'gold', 'silver']
	for k_num in k:
		# create the initial clusters
		car_data_copy = copy.deepcopy(car_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = random.randint(0, len(car_data_copy)-1)
			centers[i] = car_data_copy[ind]
			clusters[i] = []
			clusters[i].append(car_data_copy[ind])
			del car_data_copy[ind]

		for ins in car_data_copy:
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = math.sqrt((int(ins[1][-1]) - int(centers[center][1][-1])) ** 2)
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append(ins)

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = str(findMeanOfYearForCluster(clusters[clusterNum]))
				centers[clusterNum] = (-1, ['0', '0', '0', '0', '0', mean])

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = math.sqrt((int(clusters[clusterNum][insId][1][-1]) - int(centers[center][1][-1])) ** 2)
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]


		# draw the resulting clusters
		printValues = {}
		for clusterNum in clusters.keys():
			printValues[clusterNum] = []
			for ins in clusters[clusterNum]:
				printValues[clusterNum].append((ins[0], ins[1][-1]))
				# print("x = "+str(ins[0])+" y = "+ins[1][-1])
				plt.scatter(ins[0], int(ins[1][-1]), color=colors[clusterNum%8])

		plt.show()

def findMeanOfTimeT060ForCluster(cluster):
	if len(cluster)==0:
		return 0
	s = 0
	for ins in cluster:
		s += int(ins[1][-2])

	return int(s/len(cluster))

def clusterBasedOnTimeTo60(car_data, k):
	colors = ['red', 'blue', 'green', 'pink' ,'black', 'yellow', 'gold', 'silver']
	markers = ["*", "+", "1", "2", "3", ">", "o", "<"]
	for k_num in k:
		# create the initial clusters
		car_data_copy = copy.deepcopy(car_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = random.randint(0, len(car_data_copy)-1)
			centers[i] = car_data_copy[ind]
			clusters[i] = []
			clusters[i].append(car_data_copy[ind])
			del car_data_copy[ind]

		for ins in car_data_copy:
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = math.sqrt((int(ins[1][-2]) - int(centers[center][1][-2])) ** 2)
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append(ins)

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = str(findMeanOfTimeT060ForCluster(clusters[clusterNum]))
				centers[clusterNum] = (-1, ['0', '0', '0', '0', mean, '0'])

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = math.sqrt((int(clusters[clusterNum][insId][1][-2]) - int(centers[center][1][-2])) ** 2)
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]
					
		# draw the resulting clusters
		printValues = {}
		for clusterNum in clusters.keys():
			printValues[clusterNum] = []
			for ins in clusters[clusterNum]:
				printValues[clusterNum].append((ins[0], ins[1][-2]))
				# print("x = "+str(ins[0])+" y = "+ins[1][-1])
				plt.scatter(ins[0], int(ins[1][-2]), color=colors[clusterNum%8], marker=markers[clusterNum%8])

		plt.show()

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

def findMeanOfEverything(cluster):
	if len(cluster)==0:
		return ['0','0', '0', '0', '0', '0']
	s = []
	for fieldId in range(len(cluster[0][1])):
		s.append(0)
		for ins in cluster:
			s[-1] += float(ins[1][fieldId])
		s[-1] /= len(cluster)
	return s

def clusterBasedOnEveryThing(car_data, k):
	resultingCentes = []
	for k_num in k:
		print("For k = "+str(k_num)+":")
		# create the initial clusters
		car_data_copy = copy.deepcopy(car_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = random.randint(0, len(car_data_copy)-1)
			centers[i] = car_data_copy[ind]
			clusters[i] = []
			clusters[i].append(car_data_copy[ind])
			del car_data_copy[ind]

		for ins in car_data_copy:
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = calculateDist(ins, centers[center])
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append(ins)

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = findMeanOfEverything(clusters[clusterNum])
				centers[clusterNum] = (-1, [mean[0], mean[1], mean[2], mean[3], mean[4], mean[5]])

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = calculateDist(clusters[clusterNum][insId], centers[center])
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]

			# cost function 
			cost_func = 0
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break 
					cost_func += (calculateDist(clusters[clusterNum][insId], centers[clusterNum]))**2

			cost_func = cost_func / len(car_data)
			plt.scatter(inter_num, cost_func, color='black')
			# print('Cost : ', cost_func)

		print('Cluster Centers: ')
		print(centers)
		resultingCentes.append(centers)
		# print distances 
		inner_dist = 0
		outer_dist = 0
		for centerId in centers.keys():
			for val in clusters[centerId]:
				inner_dist += calculateDist(val, centers[centerId])

		inner_dist = inner_dist/len(car_data)

		for centerId in centers.keys():
			for val in clusters[centerId]:
				for centerId2 in centers.keys():
					if not centerId2==centerId:
						outer_dist += calculateDist(val, centers[centerId2])

		outer_dist = outer_dist/len(car_data)

		print("Inner dist: "+str(inner_dist))
		print("Outer dist: "+str(outer_dist))

		fileName = str(k_num)+'_'+str(clusterNum)+'_Kcluster.txt'
		try:
			os.remove(fileName)
		except OSError:
			pass
		for clusterNum in clusters.keys():
			with open(fileName, 'a') as the_file:
				for item in clusters[clusterNum]:
					the_file.write(str(item))
					the_file.write('\n')

		plt.show()

	return resultingCentes

def clusteringUsingSklearn(car_data, k):
	from sklearn.cluster import KMeans
	import numpy as np
	centers = []
	x = []
	for data in car_data:
		items = []
		for item in data[1]:
			if item==' ' or item=='':
				item = '0'
			items.append(float(item))
		x.append(items)

	for k_num in k:
		X = np.asarray(x)
		kmeans = KMeans(n_clusters=k_num, random_state=0).fit(X)
		print('Cluster Centers: ')
		print(kmeans.cluster_centers_)
		centers.append(kmeans.cluster_centers_)

	return centers

def run_k_clustering(car_data, k):
	choice = input('Run based on year and time to 60 too? (y/n)')
	if choice=='y':
		print("Running Clustering based on Year")
		clusterBasedOnYear(car_data, k)
		print("Running Clustering based on time-to-60")
		clusterBasedOnTimeTo60(car_data, k)
	
	print("Running Clustering based on all attributes")
	myCenters = clusterBasedOnEveryThing(car_data, k)

	choice = input('Run using Sklearn as well? (y/n)')
	if choice=='y':
		print("Running Clustering based on all attributes Using sklearn")
		libraryCenters = clusteringUsingSklearn(car_data, k)


		print('\n\n\n\n')
		count = 0
		for k_num in k:
			print('Diff for k = ', k_num)
			thisKCentes = myCenters[count]
			thisKCentesLib = libraryCenters[count]
			sum1 = 0
			sum2 = 0
			for item in thisKCentes.values():
				sum1 += (sum(item[1]))

			for item in thisKCentesLib:
				sum2 += (sum(item))

			print(sum1-sum2)
			count += 1


