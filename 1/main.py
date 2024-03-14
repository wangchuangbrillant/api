import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 函数f1 x和y的取值范围
x_bound = [-5.12, 5.12]
y_bound = [-5.12, 5.12]

X_precision = 5  # 决策变量的精度
y_precision = 6  # 目标变量的精度

pop_size = 30  # 种群个体数
crossover_rate = 0.75  # 交叉算子
mutation_rate = 0.01  # 变异算子
max_generation = 1000  # 最多迭代的进化次数
min_generation = 100  # 最少需迭代的进化次数
without_optim_tolerate = 200  # 如果持续200代最优值改善很小（1e-6）的话，提前终止迭代


# 利用决策变量的精度和取值区间计算基因数
def cal_DNA_size(x_bound, y_bound, X_precision):
    x_size, y_size = 1, 1
    discriminant_X = (x_bound[1] - x_bound[0]) * 10 ** X_precision
    discriminant_Y = (y_bound[1] - y_bound[0]) * 10 ** X_precision
    while 2 ** (x_size - 1) < discriminant_X:
        if discriminant_X < 2 ** x_size - 1:
            break
        x_size = x_size + 1

    while 2 ** (y_size - 1) < discriminant_Y:
        if discriminant_Y < 2 ** y_size - 1:
            break
        y_size = y_size + 1

    return x_size, y_size


x_size, y_size = cal_DNA_size(x_bound, y_bound, X_precision)
DNA_size = x_size + y_size


# 适应度计算，即函数值，函数表达式乘-1
def get_fitness(x, y):
    return -1 * (x ** 2 + y ** 2)


# 二进制转十进制，变换到决策变量取值区间
def binary_to_decimal(pop, bound, size):
    return np.around(bound[0] + pop.dot(2 ** np.arange(size)[::-1]) * (bound[1] - bound[0]) / (2 ** x_size - 1),
                     decimals=X_precision)


# 模拟自然选择，适应度越大，越大概率被保留
def select(pop, fitness):
    idx = np.random.choice(np.arange(pop_size), size=pop_size - 1, replace=True,
                           p=(fitness + (abs(min(fitness)) if min(fitness) < 0 else 0)) / (
                                       fitness.sum() + abs(min(fitness) if min(fitness) < 0 else 0) * pop_size))
    return pop[idx]


# 模拟交叉，生成新的子代
def crossover(parent, pop):
    if np.random.rand() < crossover_rate:
        i_ = np.random.randint(0, pop_size, size=1)
        cross_points = np.random.randint(0, DNA_size, size=1)
        cross_points = cross_points[0]
        return np.append(parent[0:cross_points], pop[i_, cross_points:])
    return parent


# 模拟变异
def mutate(child):
    for point in range(DNA_size):
        if np.random.rand() < mutation_rate:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(pop_size, DNA_size))

plt.ion()  # 打开plt交互模式
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(x_bound[0], x_bound[1], (x_bound[1] - x_bound[0]) / 50)
Y = np.arange(y_bound[0], y_bound[1], (y_bound[1] - y_bound[0]) / 50)
X, Y = np.meshgrid(X, Y)
Z = -1 * get_fitness(X, Y)

# 绘制3D曲面
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm')

best_F = float('-inf')
worse_F = float('inf')
best_x, best_y = 0, 0
actual_generation = 0

cur_best_F_list = []
best_F_list = []

# 核心代码，进行迭代进化
for i in range(max_generation):
    x = binary_to_decimal(pop[:, 0:x_size], x_bound, x_size)
    y = binary_to_decimal(pop[:, x_size:], y_bound, y_size)
    fitness = np.around(get_fitness(x, y), decimals=y_precision)

    cur_best_x = binary_to_decimal(pop[np.argmax(fitness), 0:x_size], x_bound, x_size)
    cur_best_y = binary_to_decimal(pop[np.argmax(fitness), x_size:], y_bound, y_size)

    print("generation:", i + 1)
    print("most fitted DNA: ", pop[np.argmax(fitness), :])
    print("var corresponding to most fitted DNA: ", cur_best_x, cur_best_y)
    print("F_values corresponding to DNA: ", -1 * fitness[np.argmax(fitness)])

    if fitness[np.argmax(fitness)] > best_F:
        best_F = fitness[np.argmax(fitness)]
        best_x = cur_best_x
        best_y = cur_best_y

    if fitness[np.argmax(fitness)] < worse_F:
        worse_F = fitness[np.argmax(fitness)]

    # 判断是否需要提前终止迭代
    if i + 1 > min_generation and (
            best_F - best_F_list[i - (without_optim_tolerate if i > without_optim_tolerate else i)]) < 10 ** (
    -y_precision):
        actual_generation = i + 1
        break

    # 逐代绘制，绘制前先清除上一代的
    if 'sca' in globals():
        sca.remove()
    sca = ax.scatter(best_x, best_y, fitness[np.argmax(fitness)], s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.001)

    # 精英保留
    pop = np.vstack((select(pop, fitness), pop[np.argmax(fitness), :]))
    pop_copy = pop.copy()  # parent会被child替换，所以先copy一份pop
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child

    cur_best_F_list.append(fitness[np.argmax(fitness)])
    best_F_list.append(best_F)

# 迭代完输出最优值和对应的决策变量
print("best F_value is", -1 * best_F)
print("var corresponding to best F_value is", best_x, best_y)

plt.ioff()
plt.show()

# 绘制进化代数和最优解的关系图
best_F_list = -1 * np.array(best_F_list)
cur_best_F_list = -1 * np.array(cur_best_F_list)
plt.plot(range(actual_generation - 1), best_F_list, label='best solution so far', color='red')
plt.plot(range(actual_generation - 1), cur_best_F_list, label='best solution of current generation', color='green')

# 每隔50代标注一下最优值
l = [i for i in range(actual_generation) if i % 50 == 0]
for x, y in zip(l, best_F_list[l]):
    print(x + 1, y)
    plt.text(x, y + 0.001, f'%.{y_precision}f' % y, ha='center', va='bottom', fontsize=12)

# 参数标注
plt.text(actual_generation * 0.5, (best_F + worse_F) / -2,
         "crossover_rate=%.2f, mutation_rate=%.2f, actual_generation=%d" % (
         crossover_rate, mutation_rate, actual_generation), fontdict={'size': 15, 'color': 'black'})
plt.legend(loc='best', fontsize=20)
plt.title('generation vs. F_value of f1', fontsize=25, color='black')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

fig = plt.gcf()
fig.set_size_inches(18, 10)

plt.savefig('f1_generation_F_value.png')
plt.show()
