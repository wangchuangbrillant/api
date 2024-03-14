import numpy as np
import random
import copy
import matplotlib.pyplot as plt


class City: #城市类 x纬度 y经度
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


def distance(ca, cb):  # 通过经度纬度信息，用两点间距离计算距离
    dx = abs(ca.x - cb.x)
    dy = abs(ca.y - cb.y)
    distance = np.sqrt((dx ** 2) + (dy ** 2))
    return distance


def init_pop(city_list, popSize):  # 初始化种群，把城市数量以及种群个数初始化
    pop = []
    for i in range(popSize):#100个
        new_city_list = random.sample(city_list, len(city_list))   #将城市内部顺序打乱
        pop.append(new_city_list)
    return pop


def fitness(pop):  # 适应度评价  距离的倒数
    dis_citys = distance_citys(pop)
    return 1.0 / dis_citys


def distance_citys(pop):  # 计算个体的总距离，以便评价最优个体
    temp_dis = 0
    for i in range(len(pop) - 1):
        temp_dis += distance(pop[i], pop[i + 1])  #每个和下一个的距离
    temp_dis += distance(pop[len(pop) - 1], pop[0])  #最后一个和第一个的距离
    return temp_dis


def rank(poplulation):  # 按照适应度排序
    rankPop_dic = {}
    for i in range(len(poplulation)):
        fit = fitness(poplulation[i])
        rankPop_dic[i] = fit  # 每个个体的适应度
        # print(poplulation[i])
    return sorted(rankPop_dic.items(), key=lambda x: x[1], reverse=True)  #返回适应度最大的  返回的是一个列表 列表的每一项是一个键值对 key是第几个个体 value是适应度大小


def select(pop, pop_rank, eliteSize):  # 适应度比例选择适应个体  选择前20个
    select_pop = []
    for i in range(eliteSize):
        select_pop.append(pop[pop_rank[i][0]]) # pop_rank[i][0]找到个体号 pop[]根据个体号找到那个个体 先将前20个适应度大的放进
    cumsum = 0
    cumsum_list = []
    temp_pop = copy.deepcopy(pop_rank) #复制
    for i in range(len(temp_pop)):
        cumsum += temp_pop[i][1]  #将适应度值加在一起
        cumsum_list.append(cumsum)  #记录每次加完之后 他所对应的值
    for i in range(len(temp_pop)):
        cumsum_list[i] /= cumsum   #把这个列表的每一项都变成0-1的数
    for i in range(len(temp_pop) - eliteSize):  # 轮盘赌，剩下的80个
        rate = random.random()   #生成随机数
        for j in range(len(temp_pop)):  #找到值比rate大的
            if cumsum_list[j] > rate:
                select_pop.append(pop[pop_rank[i][0]])
                break
    return select_pop


def breed(pop, eliteSize):  # 交叉互换，先选择，再交叉，最后将没有的补回去
    breed_pop = []
    for i in range(eliteSize):#先将前20个放进去
        breed_pop.append(pop[i])
    i = 0
    while i < (len(pop) - eliteSize):#剩下的80个
        a = random.randint(0, len(pop) - 1)  #随机数找到两个个体
        b = random.randint(0, len(pop) - 1)
        if a != b:
            fa, fb = pop[a], pop[b]  #将这两个个体拿出来
            genea, geneb = random.randint(0, len(pop[a]) - 1), random.randint(0, len(pop[b]) - 1) #随机产生两个基因
            startgene = min(genea, geneb) #这两个基因小的作为开始 大的作为结束
            endgene = max(genea, geneb)
            child1 = []
            for j in range(startgene, endgene):#两个基因中间的直接放到child1里
                child1.append(fa[j])
            child2 = []
            for j in fb:#在b中找到child1里没有的就加到child1里
                if j not in child1:
                    child2.append(j)
            breed_pop.append(child1 + child2)
            i = i + 1
    return breed_pop


def mutate(pop, mutationRate):  # 按照一定概率变异  0。005概率变异
    mutation_pop = []
    for i in range(len(pop)):  #对于所有个体
        for j in range(len(pop[i])): #对于个体的所有基因
            rate = random.random()
            if rate < mutationRate:  #产生的随机数小于变异概率
                a = random.randint(0, len(pop[i]) - 1) #再生成一个随机数选择一个基因
                pop[i][a], pop[i][j] = pop[i][j], pop[i][a]  #两个基因进行交换
        mutation_pop.append(pop[i])
    return mutation_pop


def next_pop(population, eliteSize, mutationRate):  # 产生下一代种群
    pop_rank = rank(population)  # 按照适应度排序  返回是一个列表 [(种群序号,适应度)......]
    select_pop = select(population, pop_rank, eliteSize)  # 适应度比例选择适应个体策略，加上轮盘赌选择
    breed_pop = breed(select_pop, eliteSize)  # 交叉互换
    next_generation = mutate(breed_pop, mutationRate)  # 变异
    return next_generation


# 画出路线图的动态变化
def GA_plot_dynamic(city_list, popSize, eliteSize, mutationRate, generations):
    plt.figure('Map')
    plt.ion() #打开交互模式
    population = init_pop(city_list, popSize)
    # print("初始距离:{}".format(1.0 / (rank(population)[0][1])))
    for i in range(generations):
        plt.cla() #清除原来的
        population = next_pop(population, eliteSize, mutationRate)
        idx_rank_pop = rank(population)[0][0]
        best_route = population[idx_rank_pop]
        city_x = []
        city_y = []
        for j in range(len(best_route)):
            city = best_route[j]
            city_x.append(city.x)
            city_y.append(city.y)
        city_x.append(best_route[0].x)
        city_y.append(best_route[0].y)
        plt.scatter(city_x, city_y, c='r', marker='o', s=200, alpha=0.5)
        plt.plot(city_x, city_y, "b", ms=20)
        plt.pause(0.1)#暂停0.1秒
    plt.ioff()  #关闭交互模式
    plt.show()
    print("最终距离:{}".format(1.0 / (rank(population)[0][1])))
    bestRouteIndex = rank(population)[0][0]
    bestRoute = population[bestRouteIndex]
    return bestRoute



def GA(city_list, popSize, eliteSize, mutationRate, generations):
    population = init_pop(city_list, popSize)  # 初始化种群  列表 每一项是一个个体 一个种群是100个城市
    process = []
    print("初始距离:{}".format(1.0 / (rank(population)[0][1])))
    for i in range(generations): #迭代1000次
        population = next_pop(population, eliteSize, mutationRate)  # 产生下一代种群
        process.append(1.0 / (rank(population)[0][1])) #每次距离记录下来
        print((i+1).__str__()+" 轮距离:{}".format(1.0 / (rank(population)[0][1])))#输出每一轮的距离
    plt.figure(1)  #定位创建第一个画板
    print("最终距离:{}".format(1.0 / (rank(population)[0][1]))) #将不同迭代后距离画下来
    plt.plot(process)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.savefig(str(generations) + '_' + str(1.0 / (rank(population)[0][1])) + '_' + str(mutationRate) + '_process.jpg')
    plt.figure(2) #创建第二个画板
    idx_rank_pop = rank(population)[0][0] #最优的个体号
    best_route = population[idx_rank_pop] #最后的个体
    city_x = []
    city_y = []
    city_name = []
    for j in range(len(best_route)):#确定 x,y 坐标
        print("第"+(j+1).__str__()+"个城市  ",end='')
        print(best_route[j].name,end='')
        print(best_route[j])
        city = best_route[j]
        city_name.append(best_route[j].name)
        city_x.append(city.x)
        city_y.append(city.y)
    city_x.append(best_route[0].x)
    city_y.append(best_route[0].y)
    plt.scatter(city_x, city_y, c='r', marker='o', s=200, alpha=0.5) #画散点图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    for k in range(len(best_route)):
        plt.text(city_x[k], city_y[k], city_name[k])
    plt.plot(city_x, city_y, "b", ms=20)   #连线
    plt.savefig(str(generations) + '_' + str(mutationRate) + '_route.jpg')
    plt.show()


city_list = []


with open('city.txt', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.replace('\n', '')
    city = line.split('-')
    city_list.append(City(float(city[1]), float(city[2]), city[0]))

GA(city_list, 100, 20, 0.005, 1000)
GA_plot_dynamic(city_list, 100, 20, 0.005, 1000) # 100个种群个体 进化过程中选择前20个效果好的 变异概率0.005 迭代1000次

#1.路线图是如图所示的结果
#2.根据适应度函数 多次迭代之后的曲线
#3.动态展示路线生成的结果