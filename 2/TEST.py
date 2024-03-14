import numpy as np
import matplotlib.pyplot as plt
import math
import random

# 种群数
population_num = 300
# 改良次数
improve_count = 1000
# 进化次数
iter_count = 3000
# 设置强者的概率，种群前30%是强者，保留每代的强者
retain_rate = 0.3
# 设置弱者的存活概率
live_rate = 0.5
# 变异率
mutation_rate = 0.1
# 起始点
origin = 10

'''
载入数据
'''


def read_data():
    city_name = []
    city_position = []
    with open("city.txt", 'r',encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\n')[0]
            line = line.split('-')
            city_name.append(line[0])
            city_position.append([float(line[1]), float(line[2])])
    city_position = np.array(city_position)

    # plt.scatter(city_position[:, 0], city_position[:, 1])
    # plt.show()

    return city_name, city_position


'''
计算距离矩阵
'''


def distance_Matrix(city_name, city_position):
    global city_count, distance_city
    city_count = len(city_name)
    distance_city = np.zeros([city_count, city_count])
    for i in range(city_count):
        for j in range(city_count):
            distance_city[i][j] = math.sqrt(
                (city_position[i][0] - city_position[j][0]) ** 2 + (city_position[i][1] - city_position[j][1]) ** 2)
    return city_count, distance_city


'''
获取一条路径的总距离
'''


def get_distance(path):
    distance = 0
    distance += distance_city[origin][path[0]]
    for i in range(len(path)):
        if i == len(path) - 1:
            distance += distance_city[origin][path[i]]
        else:
            distance += distance_city[path[i]][path[i + 1]]
    return distance


'''
改良
'''


def improve(path):
    distance = get_distance(path)
    for i in range(improve_count):  # 改良迭代
        u = random.randint(0, len(path) - 1)  # 在[0, len(path)-1]中随机选择交换点下标
        v = random.randint(0, len(path) - 1)
        if u != v:
            new_path = path.copy()
            t = new_path[u]
            new_path[u] = new_path[v]
            new_path[v] = t
            new_distance = get_distance(new_path)
            if new_distance < distance:  # 保留更优解
                distance = new_distance
                path = new_path.copy()


'''
杰出选择
先对适应度从大到小进行排序，选出存活的染色体，再进行随机选择，选出适应度小但是存活的个体
'''


def selection(population):
    # 对总距离进行从小到大排序
    graded = [[get_distance(path), path] for path in population]
    graded = [path[1] for path in sorted(graded)]
    # 选出适应性强的染色体
    retain_length = int(len(graded) * retain_rate)
    parents = graded[: retain_length]  # 保留适应性强的染色体
    for weak in graded[retain_length:]:
        if random.random() < live_rate:
            parents.append(weak)
    return parents


'''
交叉繁殖
'''


def crossover(parents):
    # 生成子代的个数，以此保证种群稳定
    children_count = population_num - len(parents)
    # 孩子列表
    children = []
    while len(children) < children_count:
        male_index = random.randint(0, len(parents) - 1)  # 在父母种群中随机选择父母
        female_index = random.randint(0, len(parents) - 1)
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]
            left = random.randint(0, len(male) - 2)  # 给定父母染色体左右两个位置坐标
            right = random.randint(left + 1, len(male) - 1)
            # 交叉片段
            gen1 = male[left: right]
            gen2 = female[left: right]
            # 通过部分匹配交叉法获得孩子
            # 将male和female中的交叉片段移到末尾
            male = male[right:] + male[:right]
            female = female[right:] + female[:right]
            child1 = male.copy()
            child2 = female.copy()

            for o in gen2:  # 移除male中存在于gen2交换片段上的基因
                male.remove(o)
            for o in gen1:  # 移除female中存在于gen1交换片段上的基因
                female.remove(o)

            # 直接替换child上对应的基因片段
            child1[left:right] = gen2
            child2[left:right] = gen1

            # 调整交换片段两侧的基因
            child1[right:] = male[0: len(child1) - right]  # 将原male交叉片段右侧长度对应的现male片段给child
            child1[:left] = male[len(child1) - right:]  # 将现male靠后的片段是原male的左侧片段

            child2[right:] = female[0: len(child2) - right]
            child2[:left] = female[len(child2) - right:]

            children.append(child1)
            children.append(child2)
    return children


'''
变异: 随机选取两个下标交换对应的城市
'''


def mutation(children):
    for i in range(len(children)):
        if random.random() < mutation_rate:  # 变异
            child = children[i]
            u = random.randint(0, len(child) - 2)
            v = random.randint(u + 1, len(child) - 1)
            tmp = child[u]
            child[u] = child[v]
            child[v] = tmp


'''
返回种群的最优解
'''


def get_result(population):
    graded = [[get_distance(path), path] for path in population]
    graded = sorted(graded)
    return graded[0][0], graded[0][1]  # 返回种群的最优解


'''
遗传算法
'''


def GA_TSP():
    city_name, city_position = read_data()
    city_count, distance_city = distance_Matrix(city_name, city_position)
    list = [i for i in range(city_count)]
    list.remove(origin)
    population = []
    for i in range(population_num):
        # 随机生成个体
        path = list.copy()
        random.shuffle(path)
        improve(path)
        population.append(path)
    every_gen_best = []  # 存储每一代最好的
    distance, result_path = get_result(population)
    for i in range(iter_count):
        # 选择繁殖个体群
        parents = selection(population)
        # 交叉繁殖
        children = crossover(parents)
        # 变异
        mutation(children)
        # 更新种群，采用杰出选择
        population = parents + children

        distance, result_path = get_result(population)
        every_gen_best.append(distance)

    print("最佳路径长度为：", distance)
    result_path = [origin] + result_path + [origin]
    print("最佳路线为：")
    for i, index in enumerate(result_path):
        print(city_name[index] + "(" + str(index) + ")", end=' ')
        if i % 9 == 0:
            print()

    X = []
    Y = []
    for i in result_path:
        X.append(city_position[i, 0])
        Y.append(city_position[i, 1])

    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.subplot(211)
    plt.plot(X, Y, '-o')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title("GA_TSP")
    for i in range(len(X)):
        plt.annotate(city_name[result_path[i]], xy=(X[i], Y[i]),
                     xytext=(X[i] + 0.1, Y[i] + 0.1))  # xy是需要标记的坐标，xytext是对应的标签坐标

    plt.subplot(212)
    plt.plot(range(len(every_gen_best)), every_gen_best)

    plt.show()


if __name__ == '__main__':
    GA_TSP()

