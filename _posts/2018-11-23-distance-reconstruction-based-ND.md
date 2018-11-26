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




#k-nearest neighbor-based novelty detection



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




#clustering-based approach


지금까지는 거리 기반의 novelty detection을 살펴보았습니다. 이번에는 clustering기반의 novelty detection을 해보도록 하겠습니다. 

##K-means clustering-based novelty detection


clustering기반의 novelty score 방법론의 소개할 첫번째 방법론으로 K-means clustering-based novelty detection입니다.

















reference: 

[1]Kang, P. and Cho, S. (2009). A hybrid novelty score and its use in keystroke dynamics-based user authentication. Pattern Recognition 42(11): 3115-3127

[2]https://jayhey.github.io/novelty%20detection/2018/01/29/Novelty_detection_KNN/



