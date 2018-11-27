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


#2.K-means clustering-based novelty detection

2-1. 절대적 거리기반의 K-means clustering-based novelty detection
clustering기반의 novelty score 방법론의 소개할 방법론으로 K-means clustering-based novelty detection입니다.

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


2-2. 상대적 거리기반의 K-means clustering-based novelty detection

아래의 왼쪽과 오른쪽의 절대적인 거리는 같아도 밀도가 높은곳에서 조금이라도 벗어나게 되면 상대적 distance를 크게 만들어 줄 필요가 있습니다. 상대적 distance를 만들기 위하여 밀도의 개념을 반영하였습니다. 질량은 군집안의 instance의 개수, 부피는 군집의 centroid와 군집 객체와의 최대거리로 하였습니다. 이렇게 하여 구해진 밀도를 절대적인 거리에 곱하여 밀도가 높은곳은 상대적인 거리가 길어지도록 만들었습니다. 

![](https://github.com/pilsung-kang/Business-Analytics/blob/master/03%20Novelty%20Detection/Tutorial%2009%20-%20Distance%20and%20reconstruction-based%20novelty%20detection/KMEANS/kmeans_image9.JPG?raw=true)

그러나 이러한 알고리즘의 단점은 outlier에 취약합니다. 아무래도 부피를 계산할때 군집의 centroid와 군집 객체와의 최대거리로 하였기 때문에 outlier가 존재하면 부피가 왜곡되는 현상이 발생합니다. 이를 방지하기 위해서는 k-means clustering을 통해 outlier의 제거가 필요합니다. 


#3.PCA-based novelty detection

이 방법론은 reconstruction기반의 novelty detection입니다. 원래의 2차원 데이터의 분산을 잘반영하여 1차원으로 줄이는 것인뒤, 다시 2차원으로 reconstruct하여, 본래의 2차원 데이터와 비교하여 novelty score를 계산합니다.  **원래의 데이터와 멀리있을 수록 novelty score를 크게하고 가까이 있을수록 novelty score를 작게하여야 합니다.**


![](https://github.com/pilsung-kang/Business-Analytics/blob/master/03%20Novelty%20Detection/Tutorial%2009%20-%20Distance%20and%20reconstruction-based%20novelty%20detection/PCA/PCA2.JPG?raw=true)

아래의 그림은 reconstruction된 그래프인데 1번의 점은 novelty score가 높고, 2번 점은 novelty score가 낮다고 판단 할 수 있습니다.
![](https://github.com/almond-hater/almond-hater.github.io/blob/master/pca.JPG?raw=true)


** 위의 이론들의 코드는 https://almond-hater.github.io에 첨부하였습니다.**
























reference: 



[1]Kang, P. and Cho, S. (2009). A hybrid novelty score and its use in keystroke dynamics-based user authentication. Pattern Recognition 42(11): 3115-3127

[2]https://jayhey.github.io/novelty%20detection/2018/01/29/Novelty_detection_KNN/

[3]https://ratsgo.github.io/machine%20learning/2017/04/19/KC/

[4]https://www.youtube.com/watch?v=3-fp2_mmUHs&index=9&list=PLetSlH8YjIfXHbqJmguPdw1H7BmZPy6SS








