# Developed By
# Khashayar Azerakhsh
# SMBH - SmbHosseini.ir


import copy
import math
import random
import time
import matplotlib.pyplot as plt
import pandas



"""

    Doc
    m is fuzzy partition matrix exponent for controlling the degree of fuzzy overlap. m > 1 and typically m = 2
    ||*|| Similarity Level (Distance)
    Cluster count as C
    Xk is k'th sample
    V - Center of Cluster
    Uik - Similarity level of i'th sample in k'th cluster
    U is a Matrix from Uik {0, 1} - sum of columns has to equals to 1
    
    
    Algorithm Levels
    1. initialize m, C, U -> guessing first clusters
    2. Calculate Center of Clusters (V)
    3. Calculate U from Clusters in level 2
    4. if UI+1-UI||€e|| End Algorithm else go to level 2

"""



# ===== Mapper =====
def MapData(file):

    """
        import the data into a list form a file name passed as an argument.
        The file should only the data separated by a space.(or change the delimiter as required in split)
    """

    data = []
    f = open(str(file), 'r')
    for line in f:
        current = line.split()
        for j in range(len(current)):
            current[j] = float(current[j])
        data.append(current)
    print("Data Imported")
    return data


def MapExcelData(file):

    """
        Import data from Excel Or CSV Files
    """

    # If Excel had multiple sheets
    # xl = pandas.ExcelFile(data)
    # print(xl.sheet_names)
    # tmp = xl.parse('Sheet1')

    # CSV Files
    # csv = pandas.read_csv(data, header=None, sep=' ', dtype={0:float, 1:float})

    # Excel Files
    # data = pandas.read_excel(file, header=None, dtype={0: float, 1: float})
    data = pandas.read_excel(file, dtype={0: float, 1: float})

    DataList = data.get_values()

    print("Data Imported")
    return DataList


# Shuffle function for Reducer
def ShuffleData(data):

    """
        randomise data, and also keeps record of the order of randomisation.
    """

    order = list(range(len(data)))
    random.shuffle(order)
    new_data = [[] for _ in range(len(data))]
    for index in range(len(order)):
        new_data[index] = data[order[index]]
    return new_data, order


# Sort to Original for Reducer
def DeShuffleData(data, order):
    """
        return the original order of the data,
        pass the order list returned in ShuffleData() as an argument
    """
    new_data = [[] for _ in range(len(data))]
    for index in range(len(order)):
        new_data[order[index]] = data[index]
    return new_data


def EndCondition(U, U_old):
    """
        This is the end conditions, it happens when the U matrix
        stops changing too much with successive iterations.
    """

    # used for end condition
    Epsilon = 0.00001

    for i in range(len(U)):
        for j in range(len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


# Similarity Matrix in cluster numbers range
def Initialise_U(data, cluster_number):
    """
        randomise U such that the rows add up to 1. it requires a global MAX.
    """

    # used for randomising U
    MAX = 10000.0

    U = []
    for _ in range(len(data)):
        current = []
        rand_sum = 0.0
        for _ in range(cluster_number):
            tmp = random.randint(1, int(MAX))
            current.append(tmp)
            rand_sum += tmp
        for j in range(cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def Distance(point, center):
    """
        calculate the Distance between 2 points (taken as a list). We are referring to Euclidean Distance.
    """
    if len(point) != len(center):
        return -1
    tmp = 0.0
    for i in range(len(point)):
        tmp += abs(point[i] - center[i]) ** 2
    return math.sqrt(tmp)


# Set U's members to 1 & 0
def normalise_U(U):
    """
        This de-fuzzifies the U, at the end of the clustering. It would assume that the point
        is a member of the cluster who's membership is maximum.
    """
    for i in range(len(U)):
        maximum = max(U[i])
        for j in range(len(U[0])):
            U[i][j] = 0 if U[i][j] != maximum else 1
    return U


# Graphical show
def color(cluster_number):
    return ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(cluster_number)]


def animate(data, U, cluster_number, colors):
    # plt.close("all")
    cluster = [[] for _ in range(cluster_number)]
    for i in range(len(U)):
        for j in range(cluster_number):
            if U[i][j] == 1:
                cluster[j].append(data[i])

    plt.ion()
    plt.figure()
    for i in range(cluster_number):
        x_list_0 = [x for [x, y] in cluster[i]]
        y_list_0 = [y for [x, y] in cluster[i]]
        plt.plot(x_list_0, y_list_0, colors[i], marker='o', ls='')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('equal')
    plt.show(block=False)
    time.sleep(.5)



def FuzzyReducer(data, cluster_number, m=2):
    """
        This is the main function, it would calculate the required center,
        and return the final normalised membership matrix U.
    """

    # ===== Mapper ======
    # initialise the U matrix:
    U = Initialise_U(data, cluster_number)


    print(f"Cluster Numbers = {str(cluster_number)}")

    # Graphical Show
    colors = color(cluster_number)
    plt.axis([0, 15, 0, 15])
    plt.ion()
    plt.show(block=False)
    # animate(data, normalise_U(copy.deepcopy(U)), cluster_number, colors)

    # --------- Reducer -----------
    # initialise the loop

    while True:

        # create a copy of it, to check the end conditions
        U_old = copy.deepcopy(U)

        # cluster center vector
        C = []
        for j in range(cluster_number):
            current_cluster_center = []
            for i in range(len(data[0])):  # this is the number of dimensions
                _sum_num = 0.0
                _sum_tmp = 0.0
                for k in range(len(data)):
                    _sum_num += (U[k][j] ** m) * data[k][i]
                    _sum_tmp += (U[k][j] ** m)
                current_cluster_center.append(_sum_num / _sum_tmp)
            C.append(current_cluster_center)

        # print('Cluster :', C)

        # creating a Distance vector, useful in calculating the U matrix.
        Distance_matrix = []
        for i in range(len(data)):
            current = [Distance(data[i], C[j]) for j in range(cluster_number)]
            Distance_matrix.append(current)
        # print('Distance Matrix : ', Distance_matrix)

        # update U vector
        for j in range(cluster_number):
            for i in range(len(data)):
                tmp = 0.0
                for k in range(cluster_number):
                    tmp += (Distance_matrix[i][j] / Distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / tmp
        # print('Update U : ', U)

        # Show graphical Change in every single update
        # animate(data, normalise_U(copy.deepcopy(U)), cluster_number, colors)

        if EndCondition(U, U_old):
            print("finished clustering")
            break

    animate(data, normalise_U(copy.deepcopy(U)), cluster_number, colors)
    # animate(data, U, cluster_number, colors)
    # U = normalise_U(U)
    # print("normalised U")
    return U


def Print_Matrix(list):
    """
        Prints the matrix
    """
    for i in range(len(list)):
        print(list[i])



# main section
if __name__ == '__main__':

    # data = MapData("Blobs_Data_set.txt")
    # data = MapData("Circle_Data_set.txt")
    # data = MapData("Moons_Data_set.txt")
    # data = MapData("DG.txt")

    data = MapExcelData("ExcelData.xlsx")


    # Shuffle Data And Save Order
    data, order = ShuffleData(data)

    print('Shuffled Data :')
    Print_Matrix(data)

    start = time.time()


    # call the FuzzyReducer
    final_location = FuzzyReducer(data, 2, 2)

    # DeShuffle Data
    final_location = DeShuffleData(final_location, order)

    Print_Matrix(final_location)

    print("time elapsed=", time.time() - start)
