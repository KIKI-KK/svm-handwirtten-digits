from __future__ import division
from pre_process import MNIST
from sklearn import svm
from sklearn import metrics

# 导入数据
mndata = MNIST('./Datasets')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

trainingImagesCount = len(trainingImages)
testingImagesCount = len(testImages)

# 调用svm不同内核进行比较

clf = svm.SVC(kernel='poly', degree = 2)
#clf = svm.SVC(kernel='linear')
#clf = svm.SVC(kernel='rbf')
clf.fit(trainingImages[:1000], trainingLabels[:1000])
#clf.fit(trainingImages[:60000], trainingLabels[:60000])

predictionRes = clf.predict(testImages)

# Calculation of the success of the test phase via metrics
print (metrics.classification_report(testLabels.tolist(), predictionRes, digits=4))

# Manual calculation of the success of the test phase
#rightClassifiedTestImages = 0

'''
for x in range(testingImagesCount):
   if(testLabels[x]==clf.predict(testImages[x])[0]):
        rightClassifiedTestImages+=1

print (rightClassifiedTestImages)
print ("Success: %f" %(rightClassifiedTestImages/testingImagesCount))
'''