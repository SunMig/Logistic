from math import exp
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat=[];labelMat=[];
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    print(dataMat)
    print(labelMat)
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))
#梯度上升法，求取优化系数的函数
def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    print(dataMatrix)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights


#画出边界
def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#随机梯度上升法，减少算法复杂度,改进方法是一次只使用一个样本点来更新回归系数
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatirx,classLabels,numIter=150):
    m,n=shape(dataMatirx)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))#python3中range返回的是range对象，要想得到list数组需要转化一下
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatirx[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatirx[randIndex]
            del(dataIndex[randIndex])#这里删除的是list的数据，上面的代码对range对象进行了转化
    return weights

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:return 1.0
    else:return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        curline=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(curline[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curline[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1

    errorRate=(float(errorCount)/numTestVec)
    print("the error rate is :",errorRate)
    return errorRate

def multiTest():
    numTests=10;errorSum=0.0;
    for k in range(numTests):
        errorSum+=colicTest()

    print("after "+str(numTests)+" iterations the average error rate is :"+str(errorSum/float(numTests)))