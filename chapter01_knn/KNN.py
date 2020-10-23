# K - 近邻算法描述
from numpy import *
import operator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# 重新加载文件使用reload命令，如: reload(KNN)


def createDataSet():
    """
    创建数据集，A表示爱情篇，B表示动作片
    :return:
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


# 下面对输入进行基本介绍：
# inx: 投入的数据
# dataSet: 原始的数据集
# labels: 原始数据集的标签
# k: 选择最近的数据个数
def classify0(inX, dataSet, labels, K):
    """
    函数说明：KNN函数分类器
    :param inX:
    :param dataSet:
    :param labels:
    :param K:
    :return:
    """
    dataSetSize = dataSet.shape[0]  # 这里shape函数返回的是dataSet的格式大小，而shape[0]表示的是dataSet的行数，这里行数代表的是数据集的数目。
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 将待分类的向量，复制成 dataSetSize行，1列的矩阵，然后和dataSet样本集中的点，相减，求差值。
    sqDiffMat = diffMat ** 2  # 求差向量的横坐标的平方，纵坐标的平方
    sqDistance = sqDiffMat.sum(axis=1)  # 将差向量横坐标的平方，纵坐标的平方相加起来
    distances = sqDistance ** 0.5  # 开平方得距离
    sortedDistIndicies = distances.argsort()  # 对距离长短进行排序
    classCount = {}

    for i in range(K):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    函数说明：将文本记录转换为Numpy的解析程序
    :param filename:
    :return:
    """
    # 打开数据集文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOnLines = fr.readlines()
    # 计算文本包含的行数
    numberOfLines = len(arrayOnLines)
    # 创建以0填充的矩阵，矩阵是三列，行数是文本包含的行数
    returnMat = zeros((numberOfLines, 3))
    # 分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    # 逐行遍历
    for line in arrayOnLines:
        # 截取掉所有的回车字符
        line = line.strip()
        # 使用tab字符\t将上一步得到的整行数据分割成一个元素列表
        listFromLine = line.split('\t')
        # 选取前三个元素将其存入特征矩阵之中
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


def showdatas(datingDataMat, datingLabels):
    """
    函数说明：数据可视化
    :param datingDataMat:
    :param datingLabels:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # 以第二列和第三列为x,y轴画出散列点，给予不同的颜色和大小
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    """
    函数说明：归一化特征值
    :param dataSet:
    :return:
    """
    # 获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals


def datingClassTest():
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    # 取所有数据的百分之十
    hoRatio = 0.10
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


def classifyPerson():
    """
    函数说明： 输入一个人的三围特征，进行分类输出
    """
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([precentTats, ffMiles, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 20]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)
    datingClassTest()
    classifyPerson()
