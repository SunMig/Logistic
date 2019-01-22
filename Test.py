from functions import *
from numpy import *

dataMat,labelMat=loadDataSet()
CalculateWeights=gradAscent(dataMat,labelMat)
print(CalculateWeights)
plotBestFit(CalculateWeights.getA())
#随机梯度上升法的测试
weights=stocGradAscent1(array(dataMat),labelMat)
plotBestFit(weights)

#预测病马死亡
multiTest()
