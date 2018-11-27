---



layout: post



title: "distance/reconstruction-based ND "



comments: true



description: "Distance_Reconstruction-based ND"



keywords: "Business Analytics"



---







이번글은 distance/reconstruction-based novelty detection(ND)의 방법론인 k-Nearest Neighbor-based ND, Clustering기반 ND, PCA기반 ND에 대하여 살펴보겠습니다. 글을 시작하기전 아래글은 강필성 교수님 강의를 기반으로 만들어졌음을 먼저 밝히겠습니다.









novelty detection를 간단하게 설명하자면, 다수의 범주 데이터를 활용하여 이상치(outlier)가 아닌 영역을 구분하는 것이라고 볼 수 있습니다.(figure1참조) novelty detection의 방법론에는 밀도기반, 거리기반 등 다양한 방법론이 있습니다.









![](https://github.com/almond-hater/almond-hater.github.io/blob/master/1.png?raw=true)









#1.k-nearest neighbor-based novelty detection







먼저 살펴볼 k-nearest Neighbor-based Approach라는 거리 기반의 방법론을 살펴보겠습니다. 이 방법론은 쉽게 말해 **k개의 이웃들의 거리 정보를 기반으로 novelty 점수를 매기는 것입니다.** 계산된 novelty 점수가 다른 데이터들과 비교하였을때 높게 나온다면 이상치일 확률이 높다고 할 수 있습니다.(figure2참조) 이때, 다수의 범주의 사전 확률 분포는 고려하지 않습니다.









![](https://github.com/almond-hater/almond-hater.github.io/blob/master/2.png?raw=true)









**그렇다면 거리기반의 novelty 점수를 계산하는 방법을 살펴보도록 하겠습니다.(figure3참조)**









![](https://github.com/almond-hater/almond-hater.github.io/blob/master/3.png?raw=true)







-maximum distance to the k-th nearest neighbor은 최대거리를 기반 





-average distance to k-nearest neighbors는 거리의 평균을 기반





-distance to the mean of the k-nearest neighbors는 이웃들간의 평균을 기반 





한 것입니다.









위의 세가지 점수 지표를 비교하는 아래의 그림을 살펴보도록 하겠습니다. 





먼저, figure4의 (a)와 (b)를 비교해  본다면 maximum distance는 동일하지만, average distance는 (a)가 더 높음을 알 수 있습니다.





y가 x에 비해 더 밀집되어 있는데 이러한 정보는 maximum distance에는 포함 할 수 있지 않지만, average distance에는 데이터의 밀집력 정보도 함께 내포 할 수 있습니다.





그리고 (c)와 (d)를 비교해 본다면 average distance는 동일하지만, distance to the mean은 (c)가 더 높음을 알 수 있습니다.





distance to the mean의 정보는 밀집력뿐만 아니라 상대적 위치에 관한 정보를 함께 담고있다는 것을 알 수 있습니다.







![](https://github.com/almond-hater/almond-hater.github.io/blob/master/4.png?raw=true)







*** result ***







figure5는 앞서 공부한 distance 기반의 novelty score를 구한 예시입니다. 동그라미가 일반적으로 생각하는 novelty이고 삼각형이 일반 데이터
입니다. 보시면 a에서는 average distance에서만 제대로 novelty detection을 하였고, b에서는 어떠한 알고리즘도 novelty detection을하지 못하였습니다.







![](https://github.com/almond-hater/almond-hater.github.io/blob/master/5.png?raw=true)









**이러한 문제점을 보완하기 위하여 기존 계산 방법에 convex hull까지 거리를 고려하였습니다.**









이 개념은 "나는 이웃으로부터 재구축 되어야 한다"로 부터 시작합니다. convex hull이란 주어진 점이나 영역을 포함하는 가장 작은 볼록 집합입니다. 여기서는 깊게 다룰 내용이 아니지만 반드시 이러한 사항만은 알아야 합니다. 이웃들끼리 연결했을 때 그 안에 있으면 거리가 0, 그 밖에 있으면 거리가 0 이상이 된다는 것입니다.(figure6 참조) 즉, 이웃들과의 convex combination과의 거리를 계산하겠다는 뜻입니다. 수식으로 나타내면 아래와 같습니다. 









![](https://github.com/almond-hater/almond-hater.github.io/blob/master/6.png?raw=true)









![](https://github.com/almond-hater/almond-hater.github.io/blob/master/6-1.JPG?raw=true)









** 이제 위에서 설명한 average distance와 convex distance를 합쳐보도록 하겠습니다.**











![](https://github.com/almond-hater/almond-hater.github.io/blob/master/6-2.JPG?raw=true)







명심해야 할 것은, 



![](https://github.com/almond-hater/almond-hater.github.io/blob/master/6-3.JPG?raw=true)









의 값은 [1,2)사이의 값을 가집니다. 이 말은 convex밖에 있다면 최대 2배의 패널티를 주겠다는 말과 같습니다.





hybrid novelty score의 장점은 계산량은 앞선 distance based 방법론과 비슷하지만 성능이 좋다는 장점이 있습니다.





알고리즘의 계산 절차는 figure7과 같습니다.









![](https://github.com/almond-hater/almond-hater.github.io/blob/master/7.png?raw=true)









*** result ***





figure8을 보신다면, hybrid novelty score에서는 모두 적합하게 hybrid novelty detection한다는 것을 알 수 있습니다. 또한 figure9의 hybrid novelty detection을 본다면, 완벽하게 3개의 그룹을 형성하고 있다는 것을 알고 있습니다. 그리고 왼쪽위에는 바나나 모양, 아래와 오른쪽 위는 구모양을 형성하고 있음을 볼 수 있습니다. 







![](https://github.com/almond-hater/almond-hater.github.io/blob/master/8.png?raw=true)





![](https://github.com/almond-hater/almond-hater.github.io/blob/master/9.png?raw=true)







figure10와 같이 실제 데이터에도 적용해 보았는데 21개 실제데이터중 12개에서 가장 좋은 성능을 보였습니다.(figure11 참조)





![](https://github.com/almond-hater/almond-hater.github.io/blob/master/10.png?raw=true)



![](https://github.com/almond-hater/almond-hater.github.io/blob/master/11.png?raw=true)

*** hybrid novelty detection 코드***
```python
import math
import numpy as np
import operator
from collections import namedtuple 
import matplotlib.pyplot as plt  
```
필요한 패키지를 로드합니다.
그리고 convex의 거리를 구하기 위해 Convex hull의 좌표를 구합니다.(아래 코드 참조)convex hull에 대한 자세한 설명은 생략하도록 하겠습니다. (reference 참고)
```python
Point = namedtuple('Point', 'x y')

class ConvexHull(object):  

    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is clockwise of p2.
        :param p1:
        :param p2:
        :return: integer
        '''
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:
            
            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1
 
            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2


            self._hull_points.append(far_point)
            point = far_point


    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)

        plt.title('Convex Hull')
        plt.show()
 
```
앞서 이웃들과의 Convex hull의 좌표를 계산했다면, convex combination과의 거리를 계산합니다. 
```python
def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def getconvexdist(arrtestSet,getConvexHullPoint):

    pnt2lineset=[]
    for i in range(arrtestSet.shape[0]):
        for j in range(getConvexHullPoint.shape[0]-1):
        #pnt2line(pnt, getConvexHullPoint[i], getConvexHullPoint[i+1])
            result=pnt2line(arrtestSet[i], getConvexHullPoint[j], getConvexHullPoint[j+1])
            pnt2lineset.append(result[0])

    pnt2lineset=np.reshape(pnt2lineset,(arrtestSet.shape[0],getConvexHullPoint.shape[0]-1))
    convex_dist=np.min(pnt2lineset,axis=1)
    print('pnt2lineset:',pnt2lineset)

    return convex_dist

def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)
```
유클리디안 거리를 활용하여 average distance를 계산합니다. 그리고 hybrid novelty detection socre를 수식을 활용하여 표현합니다.
```python
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors


def main():  
    ch = ConvexHull()


    trainingSet = [[1,1,'s'],[3,1,'s'],[2,1,'s'],[1.5,2,'s'],[2.5,2,'s']]
    testSet = [[1.15,1.2,'a'],[2,2.2,'a']]

    arrtrainingSet=np.array(trainingSet)
    arrtestSet=np.array(testSet)

    arrtrainingSet=arrtrainingSet[:,:-1].astype(np.float)
    arrtestSet=arrtestSet[:,:-1].astype(np.float)
  
    for i in range(arrtrainingSet.shape[0]):
        ch.add(Point(arrtrainingSet[i,0], arrtrainingSet[i,1]))



    getConvexHullPoint=np.array(ch.get_hull_points())
    ch.display()

    print('ConvexHullPoint:',getConvexHullPoint)

    convex_dist=getconvexdist(arrtestSet,getConvexHullPoint)

    print('convex_dist:',convex_dist)
        
    k = 5
    neighbors_inform=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        neighbors_inform.append(neighbors)
        
    neighbors_dist=np.array(neighbors_inform)[:,:,1]
    knn_dist_avg=np.mean(neighbors_dist,axis=1)
    
    
    print('knn_dist_avg:',knn_dist_avg)
    
    knn_dist_hybrid=knn_dist_avg*2/(1+np.exp(-convex_dist))
    print('knn_dist_hybrid:',knn_dist_hybrid)
    
    
main()
```


#2.K-means clustering-based novelty detection

2-1. 절대적 거리기반의 K-means clustering-based novelty detection
clustering기반의 novelty score 방법론의 소개할 이론은 K-means clustering-based novelty detection입니다.

K-means clustering-based novelty score는 가장 가까운 중심에 대한 거리 정보를 기초하여 계산됩니다. 여기에서도 마찬가지로 다수의 범주의 사전 확률 분포는 고려하지 않습니다. K-NN에서의 K의 갯수는 이웃의 개수이지만, K-means의 K는 중심의 개수를 의미합니다. 각 군집에서 멀수록 이상치라고 판단하는 것이 기본 매커니즘 입니다.

![](https://github.com/pilsung-kang/Business-Analytics/blob/master/03%20Novelty%20Detection/Tutorial%2009%20-%20Distance%20and%20reconstruction-based%20novelty%20detection/KMEANS/kmeans_image.JPG?raw=true)


k-means클러스터링을 하는 방법은 아래와 같습니다.

1단계 랜덤하게 중심점(centroid) 설정
2단계 설정된 중심을 기반으로 객체 할당
3단계 각각 구해진 영역에 대해 중심점(centroid)을 구함
4단계 이 중심점(centroid)를 기반으로 객체 할당
5단계 이것을 변하지 않을때까지 반복
![](https://github.com/pilsung-kang/Business-Analytics/blob/master/03%20Novelty%20Detection/Tutorial%2009%20-%20Distance%20and%20reconstruction-based%20novelty%20detection/KMEANS/kmeans_image2.JPG?raw=true)


**마지막 단계에서 정해진 중심점 좌표와 데이터 셋의 instance와의 거리를 비교하여 최근접 군집 중심좌표와의 거리가 novelty score가 되는 것입니다.**

*** 절대적 거리기반의 K-means clustering-based novelty detection 코드***
```python
import numpy as np
import random
import csv
import matplotlib.pyplot as plt

#파일로 데이터 입력(Column Name없이), 파일명 : ex_kmc.csv
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])  
  
#중심점이 변하지 않을때까지 클러스터링 실행       
def kmeans(X,k,maxIt):
    numPoints,numDim=X.shape

    dataSet=np.zeros((numPoints,numDim+1))
    dataSet[:,:-1]=X
    centroids=dataSet[np.random.randint(numPoints,size=k),:]
    centroids[:,-1]=range(1,k+1)
    #print("centroids:",centroids)

    iterations=0;
    oldCentroids=None
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        oldCentroids=np.copy(centroids)
        iterations+=1
        updateLabels(dataSet, centroids)
        centroids=getCentroids(dataSet, k)
    return dataSet, centroids

#중심점이 변하지 않으면 EM알고리즘 STOP!
def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)

def updateLabels(dataSet,centroids):
    numPoints,numDim=dataSet.shape
    for i in range(numPoints):
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1], centroids)

def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]
            #print("label:"+str(label))
    return label

def getCentroids(dataSet,k):       
    result=np.zeros((k,dataSet.shape[1]))
    #print("result:",result)
    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        #print("cluster:",oneCluster)
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1,-1]=i
    #print("result:",result)
    return result

#데이터와 중심점(군집별로 다함)의 차이를 중 minDist에 저장하여 거기에 최소값을 뽑아냄(가장 가까이 있는 군집과 비교해야 하므로)
def getNoveltyScore(dataSetRow,centroids,k):
    minDist=[]    
    for i in range(k):
        Dist=np.linalg.norm(dataSetRow[:,:-1]-centroids[i,:-1],axis=1)   
        minDist.append(Dist)        
    NoveltyScore=np.min(minDist,axis=0)
    #print(minDist)
    #print(NoveltyScore)
    return NoveltyScore


def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 1
    random.seed(100)
    loadDataset('ex_kmc2.csv', split, trainingSet, testSet)
    trainX=np.array(trainingSet)
    
    # k=군집개수, max_iter=반복 회수 제한(Hyper parameter)   
    k=2
    max_iter=100
    final_result=kmeans(trainX[:,:-1],k,max_iter)
    
    kmeans_result=final_result[0]
    centroid_result=final_result[1]
    Score=getNoveltyScore(kmeans_result,centroid_result,k)
    
    x=kmeans_result[:,0]
    y=kmeans_result[:,1]
    colors=kmeans_result[:,2]
    
    plt.figure(figsize=(7, 3), dpi=80)
    plt.scatter(x, y, s=(Score**3)*3, c=colors);
    plt.xticks(np.arange(-1,8,2))
    plt.yticks(np.arange(-1.5,1.6,0.5))

main()
```

2-2. 상대적 거리기반의 K-means clustering-based novelty detection

아래의 왼쪽과 오른쪽의 절대적인 거리는 같아도 밀도가 높은곳에서 조금이라도 벗어나게 되면 상대적 distance를 크게 만들어 줄 필요가 있습니다. 상대적 distance를 만들기 위하여 밀도의 개념을 반영하였습니다. 질량은 군집안의 instance의 개수, 부피는 군집의 centroid와 군집 객체와의 최대거리로 하였습니다. 이렇게 하여 구해진 밀도를 절대적인 거리에 곱하여 밀도가 높은곳은 상대적인 거리가 길어지도록 만들었습니다. 

![](https://github.com/pilsung-kang/Business-Analytics/blob/master/03%20Novelty%20Detection/Tutorial%2009%20-%20Distance%20and%20reconstruction-based%20novelty%20detection/KMEANS/kmeans_image9.JPG?raw=true)

그러나 이러한 알고리즘의 단점은 outlier에 취약합니다. 아무래도 부피를 계산할때 군집의 centroid와 군집 객체와의 최대거리로 하였기 때문에 outlier가 존재하면 부피가 왜곡되는 현상이 발생합니다. 이를 방지하기 위해서는 k-means clustering을 통해 outlier의 제거가 필요합니다. 

*** 상대적 거리기반의 K-means clustering-based novelty detection 코드***
아래코드는 1번의 k-means clustering를 통하여 outlier를 제거하는 과정이 포함되어 있습니다.
```python
%matplotlib inline
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


#파일로 데이터 입력(Column Name없이), 파일명 : ex_kmc.csv
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])  
  
         
def kmeans(X,k,maxIt):
    numPoints,numDim=X.shape

    dataSet=np.zeros((numPoints,numDim+1))
    dataSet[:,:-1]=X
    centroids=dataSet[np.random.randint(numPoints,size=k),:]
    centroids[:,-1]=range(1,k+1)
    #print("centroids:",centroids)

    iterations=0;
    oldCentroids=None
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        oldCentroids=np.copy(centroids)
        iterations+=1
        updateLabels(dataSet, centroids)
        centroids=getCentroids(dataSet, k)
    return dataSet, centroids


def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)

def updateLabels(dataSet,centroids):
    numPoints,numDim=dataSet.shape
    for i in range(numPoints):
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1], centroids)

def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]
            #print("label:"+str(label))
    return label


def getCentroids(dataSet,k):       
    result=np.zeros((k,dataSet.shape[1]))
    #print("result:",result)
    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        #print("cluster:",oneCluster)
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1,-1]=i
    #print("result:",result)
    return result

#이부분이 위에서 설명한 상대적 novelty score설명하는 부분(밀도기반)
def getNoveltyScore(dataSetRow,centroids,k):
    minDist=[]  
    maxDist=[]
    for i in range(k):
        Dist=np.linalg.norm(dataSetRow[:,:-1]-centroids[i,:-1],axis=1)   
        minDist.append(Dist)     
    
    NoveltyScore=np.min(minDist,axis=0)
    min_index=np.argmin(minDist,axis=0)
    
    for i in range(k):
        clusterDist=NoveltyScore[np.where(min_index==i)]   
        maxDist.append(np.max(clusterDist))
    
#    a=centroids[min_index,-1]
#    print(a)
    #print(NoveltyScore)
    return NoveltyScore,maxDist


def getModNoveltyScore(kmeans_result,centroid_result,maxDist,k):
    minDist=[]
    countcluster=[]    
    for i in range(k):
        Dist=np.linalg.norm(kmeans_result[:,:-1]-centroid_result[i,:-1],axis=1)
        minDist.append(Dist)
        count=np.sum(kmeans_result[:,-1]==i+1)
        countcluster.append(count)           

    NoveltyScore=np.min(minDist,axis=0)
    weight=(countcluster/np.min(countcluster))/maxDist
    scale_weight=weight/np.min(weight)
    mod_NoveltyScore = NoveltyScore*np.repeat(scale_weight,countcluster,axis=0)

#    print(np.repeat(scale_weight,countcluster,axis=0))
#    print(countcluster)
#    print(np.min(countcluster))
#    print(scale_weight)
#    print(minDist)
#    print(MaxDist)
#    print(NoveltyScore)
   
    return mod_NoveltyScore


def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 1
    random.seed(100)
    loadDataset('ex_kmc2.csv', split, trainingSet, testSet)
    trainX=np.array(trainingSet)
    
    #print('Train set: ' ,repr(len(trainingSet)))
    #print('Test set: ' + repr(len(testSet)))
    #print('Train set: ' ,trainingSet[0:10])
    #a=trainX[:,:-1].astype(np.float)
    #plt.scatter(a[:,0], a[:,1]);
    
    # k=군집개수, max_iter=반복 회수 제한(Hyper parameter)   
    k=2
    max_iter=100
    final_result=kmeans(trainX[:,:-1],k,max_iter)

    kmeans_result=final_result[0]
    centroid_result=final_result[1]
    
    NoveltyScore_result=getNoveltyScore(kmeans_result,centroid_result,k)  
    Score=NoveltyScore_result[0]
    maxDist=NoveltyScore_result[1]
    
    print('final centroid:',centroid_result)
    print('maxDist:',maxDist)
    print('Novelty Score:',Score)


    #Outlier제거
    outlier_percentage=0.98
    Distsorting=np.argsort(Score)
    outlier_point=int(Distsorting.shape[0]*outlier_percentage)    
    cutoff_index=Distsorting[0:outlier_point]
    trainX=trainX[cutoff_index]
    
    #두번째 K-Means (새로운 Centroid와 군집의 크기(MaxDist)를 구함)
    final_result_second=kmeans(trainX[:,:-1],k,max_iter)
    kmeans_result_second=final_result_second[0]
    centroid_result_second=final_result_second[1]
    
    NoveltyScore_result_second=getNoveltyScore(kmeans_result_second,centroid_result_second,k)  
#    Score_second=NoveltyScore_result_second[0]
    maxDist_second=NoveltyScore_result_second[1]

    #Relative Distance기반의 Novelty Score    
    RelativeScore=getModNoveltyScore(kmeans_result,centroid_result_second,maxDist_second,k)

    print('final centroid_second:',centroid_result_second)
    print('maxDist_second:',maxDist_second)
    print('Relative Novelty Score:',RelativeScore)    
   
    x=kmeans_result[:,0]
    y=kmeans_result[:,1]
    colors=kmeans_result[:,2]
    
    plt.figure(figsize=(7, 3), dpi=80)
    plt.scatter(x, y, s=(RelativeScore**2)*5, c=colors,alpha=0.5);
    plt.xticks(np.arange(-1,8,2))
    plt.yticks(np.arange(-1.5,1.6,0.5))
    
main()
```

#3.PCA-based novelty detection

이 방법론은 reconstruction기반의 novelty detection입니다. 원래의 2차원 데이터의 분산을 잘반영하여 1차원으로 줄이는 것인뒤, 다시 2차원으로 reconstruct하여, 본래의 2차원 데이터와 비교하여 novelty score를 계산합니다.  **원래의 데이터와 멀리있을 수록 novelty score를 크게하고 가까이 있을수록 novelty score를 작게하여야 합니다.**


![](https://github.com/pilsung-kang/Business-Analytics/blob/master/03%20Novelty%20Detection/Tutorial%2009%20-%20Distance%20and%20reconstruction-based%20novelty%20detection/PCA/PCA2.JPG?raw=true)

아래의 그림은 reconstruction된 그래프인데 1번의 점은 novelty score가 높고, 2번 점은 novelty score가 낮다고 판단 할 수 있습니다.
![](https://github.com/almond-hater/almond-hater.github.io/blob/master/pca.JPG?raw=true)

*** PCA-based novelty detectio 코드***
pca패키지를 사용하지 않은 코드입니다.
```python
%matplotlib inline
import numpy as np
import random
import csv
import matplotlib.pyplot as plt

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

                
def pca(X,n): 

    #1st Step : subtract mean (x에 x의 average를 빼줌)
    avg = np.mean(X,axis=0)
    avg = np.tile(avg,(X.shape[0],1)) 
    X -= avg; 
    #print(avg)

    #2nd Step : covariance matrix 
    C = np.dot(X.transpose(),X)/(X.shape[0]-1)

    #3rd Step : Eigen Value, Eigen Vector 
    eig_values,eig_vecs = np.linalg.eig(C)

    #4rd Step : Select n개의 PC  (eigenvalue의 가장 왼쪽에 있는 수가 큰 eigenvalue가 되도록 정렬)
    idx = np.argsort(eig_values)[-n:][::-1]
    eig_values = eig_values[idx] 
    eig_vecs = eig_vecs[:,idx] 
    
    #print(eig_values.argsort())
    #print(eig_values.argsort()[-n:])
    #print(idx)
    #print(eig_values)
    #print(eig_vecs)
 
    #5th Step : new coordinate in new space (새로운 feature로 변형(2차원>1차원))
    Y = np.dot(X,eig_vecs) 

    #6th Step : reconstruction (1차원>2차원)
    rec=np.dot(eig_vecs,Y.transpose())
    
    #Get NovetyScore
    Score=np.linalg.norm(X.transpose()-rec,axis=0)     
    
    #print(rec)
    #print(Score)
    
    return (X.transpose(), rec, Score.transpose(), eig_vecs, eig_values)

```
```python
def main():

	# prepare data
    trainingSet=[]
    testSet=[]
    split = 0.8
    random.seed(100)
    loadDataset('ex_pca5.csv', split, trainingSet, testSet)
    #print('Train set: ' + repr(len(trainingSet)))
    #print('Test set: ' + repr(len(testSet)))

    # n=PC개수 (Hyper parameter)  
    n=1
    trainX=np.array(trainingSet)
    pca_result=pca(trainX[:,:-1].astype(np.float),n)

   # print('pca result:',pca_result)

    x=pca_result[0][0]
    y=pca_result[0][1]
    Score=pca_result[2]*100

    print('Eigen Value : ', pca_result[4])
    print('Eigen Vector : ', pca_result[3])    
    print('Data X : ', np.transpose(pca_result[0])[:10])
    print('Reconstruction : ', np.transpose(pca_result[1])[:10])
    print('Novelty Score : ', np.transpose(pca_result[2])[:10])
        
    x_rec=pca_result[1][0]
    y_rec=pca_result[1][1]
    
    plt.figure(figsize=(6, 6), dpi=80)
    plt.scatter(x,y,s=Score);
    plt.scatter(x_rec,y_rec,s=20);
    plt.xticks(np.arange(-3,3,0.5))
    plt.yticks(np.arange(-3,3,0.5))
         
main()

```
** 위의 이론들의 데이터는 https://almond-hater.github.io에 첨부하였습니다.**
























reference: 



[1]Kang, P. and Cho, S. (2009). A hybrid novelty score and its use in keystroke dynamics-based user authentication. Pattern Recognition 42(11): 3115-3127

[2]https://jayhey.github.io/novelty%20detection/2018/01/29/Novelty_detection_KNN/

[3]https://ratsgo.github.io/machine%20learning/2017/04/19/KC/

[4]https://www.youtube.com/watch?v=3-fp2_mmUHs&index=9&list=PLetSlH8YjIfXHbqJmguPdw1H7BmZPy6SS








