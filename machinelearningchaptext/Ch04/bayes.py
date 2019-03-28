'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],#0
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],#1
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],#0
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],#1
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],#0
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]#1
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets 两个集合的并集
    return list(vocabSet)

    #集合操作： |取并集   &取交集   -取差集
    
    
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)   #创建一个len(vocabList)维向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec
#Python index() 方法检测字符串中是否包含子字符串 str ，如果指定 beg（开始） 和 end（结束） 
#范围，则检查是否包含在指定范围内，该方法与 python find()方法一样，
#只不过如果str不在 string中会报一个异常。
#str.index(str, beg=0, end=len(string))
#如果包含子字符串返回开始的索引值，否则抛出异常。

def trainNB0(trainMatrix, trainCategory):#文档矩阵， 每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix)
    #print("numTrainDocs: ",numTrainDocs)#6
    numWords = len(trainMatrix[0])
    #print("numWords: ", numWords)#6

    #文档属于侮辱类性文档概率
    pAbusive = sum(trainCategory)/float(numTrainDocs) #向量加和（羞辱词的数量）/总数量=

    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:#侮辱类
            p1Num += trainMatrix[i]
            #print("p1Num: ",p1Num)
            p1Denom += sum(trainMatrix[i])
            #print("p1Denom: ",p1Denom)
        else:
            p0Num += trainMatrix[i]
            #print("p0Num",p0Num)
            p0Denom += sum(trainMatrix[i])
            #print("p0Denom: ",p0Denom)
    p1Vect = log(p1Num/p1Denom)         #change to log()

    p0Vect = log(p0Num/p0Denom)         #change to log()
    return p0Vect,p1Vect,pAbusive
'''
def trainNB0_Old(trainMatrix,trainCategory):#文档矩阵， 每篇文档类别标枪所构成的向量
    numTrainDocs = len(trainMatrix)
    print("numTrainDocs: ",numTrainDocs)
    numWords = len(trainMatrix[0])
    print("numWords: ", numWords)

    #文档属于侮辱类性文档概率
    pAbusive = sum(trainCategory)/float(numTrainDocs) #向量加和（羞辱词的数量）/总数量=

    p0Num = zeros(numWords); p1Num = zeros(numWords)      #change to ones()
    p0Denom = 0.0; p1Denom = 0.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            print("p1Num: ",p1Num)
            p1Denom += sum(trainMatrix[i])
            print("p1Denom: ",p1Denom)
        else:
            p0Num += trainMatrix[i]
            print("p0Num",p0Num)
            p0Denom += sum(trainMatrix[i])
            print("p0Denom: ",p0Denom)
    p1Vect = p1Num/p1Denom         #change to log()
    p0Vect = p0Num/p0Denom         #change to log()
    return p0Vect,p1Vect,pAbusive
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):#pClass1侮辱类的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    print("pAb:", pAb)
    print("p0V:", p0V)
    print("p1V:", p1V)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc: ', thisDoc)
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print('thisDoc: ', thisDoc)
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
  
    
#文档词袋准备
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
    
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

#字符串前面加上 r 表示原生字符串
#str.split(str="", num=string.count(str))
#分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
#分割次数。默认为 -1, 即分隔所有
#.   匹配任意1个字符（除了\n）
#[ ] 匹配[ ]中列举的字符
#\d  匹配数字，即0-9
#\D  匹配非数字，即不是数字
#\s  匹配空白，即 空格，tab键
#\S  匹配非空白
#\w  匹配非特殊字符，即a-z、A-Z、0-9、_、汉字
#\W  匹配特殊字符，即非字母、非数字、非汉字、非_


def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)#按照对象插入，
        fullText.extend(wordList)#按照内容插入
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        
    vocabList = createVocabList(docList)          #create vocabulary
    
    trainingSet = range(50); testSet=[]           #create test set
    #trainingSet = (0, 50)
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        #random.uniform(a,b)：用于生成一个指定范围内的随机浮点数，两格参数中，其中一个是上限，一个是下限。
        #如果a>b，则生成的随机数n，即b<=n<=a；如果a>b，则a<=n<=b。
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print( "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])

def main():
    # listOPosts,listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # trainMat=[]
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    # print("p0V: ", p0V)
    # print("p1V: ", p1V)
    # print("pAb: ", pAb)
    testingNB()
if __name__ == '__main__':
    main()
 