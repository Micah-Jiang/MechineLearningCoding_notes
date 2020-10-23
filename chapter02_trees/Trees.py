from math import log


def createDataSet():
    # 数据集
    '''
    函数说明：创建测试数据集
    :return: dataSet - 数据集
    labels - 分类属性
    :others:
    年龄：0代表青年，1代表中年，2代表老年；
    有工作：0代表否，1代表是；
    有自己的房子：0代表否，1代表是；
    信贷情况：0代表一般，1代表好，2代表非常好；
    类别(是否给贷款)：no代表否，yes代表是。
    '''
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, labels  # 返回数据集和分类属性


def calcShannonEnt(dataSet):
    '''
    函数说明：计算给定数据集的经验熵（又称香浓熵）
    :param dataSet: 数据集
    :return:
    '''
    # 计算数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签(Label)出现次数的字典
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 提取标签(Label)信息
        currentLabel = featVec[-1]
        # 如果标签(Label)没有放入统计次数的字典,添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # Label计数
        labelCounts[currentLabel] += 1
    # 经验熵(香农熵)
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # 选择该标签(Label)的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob * log(prob, 2)
    # 返回经验熵(香农熵)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''
    函数说明：按照给定特征划分数据集
    :param dataSet: 待划分数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    '''
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 这个和下面那个的主要目的就是为了去掉featVec中axis所对应的值。
            reducedFeatVec = featVec[:axis]
            # 这里要区分一下extend和append。例如a=[1,2,3],b=[3,4,5],a.append则为[1,2,3,[4,5,6]]而extend的值为[1,2,3,4,5,6]
            reducedFeatVec.extend(featVec[axis + 1:])
            # 这个是为了把变化后的每个小列表添加到retDataSet中
            retDataSet.append(reducedFeatVec)
    # 返回划分后的数据集
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    函数说明：选择最优特征，信息增益越大，就越是最优特征
    :param dataSet:
    :return:
    '''
    # 特征数量
    numFeatures = len(dataSet[0]) - 1   # 有一个元素不是特征，是评估结果，所以减1。
    print("特征数量是：" + str(numFeatures))  # 4个
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)  # 香浓熵的计算公式是：shannonEnt -= prob * log(prob, 2) ，其中prob是 选择该特征标签的概率。
    # 信息增益
    bestInfoGain = 0.0  # 信息增益是：香浓熵 - 条件熵
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]  # 重要，稍后研究
        # 创建set集合{}, 元素不可重复
        uniqueVals = set(featList)  # Set集合的一个特点就是集合中的元素不重复
        # 经验条件熵
        newEntropy = 0.0  # 经验条件熵的计算公式为：newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        # 计算信息增益
        if (infoGain > bestInfoGain):
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i
    # 返回信息增益最大的特征的索引值
    return bestFeature


if __name__ == '__main__':
    dataSet, features = createDataSet()
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    print(dataSet)
    print(calcShannonEnt(dataSet))
