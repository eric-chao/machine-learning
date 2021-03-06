
## ***贝叶斯模型***
样本集
<img src="https://latex.codecogs.com/gif.latex?$$X$$">，
每个样本有
<img src="https://latex.codecogs.com/gif.latex?$$n$$">
种可能出现的特征
<img src="https://latex.codecogs.com/gif.latex?$$X_i%20=%20\{x_1,%20x_2,%20x_3%20...%20x_n\}$$">
和与之相对应的类别
<img src="https://latex.codecogs.com/gif.latex?$$C_i(i%20\in%20k)$$">。

贝叶斯公式：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?P(C_k|X)%20=%20\frac{P(C_k)P(X|C_k)}{P(X)}">

由条件概率公式：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?$$(P(A|B)=\frac{P(AB)}{P(B)}$$">

公式变换(乘法公式)：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?$$P(AB)=P(A|B)\cdot{P(B)}=P(B|A)\cdot{P(A)}$$">

可知贝叶斯公式中的分子就是联合概率分布
<img src="https://latex.codecogs.com/gif.latex?$$P(C_k,X)$$">：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?$$P(C_k,X)=P(x_1|x_2,...,x_n,C_k)P(x_2|x_3,...,x_n,C_k)P(x_n|,C_k)P(C_k)$$">：

由贝叶斯假设(假设
<img src="https://latex.codecogs.com/gif.latex?$$x_1,x_2,x_3...x_n$$">
互斥且构成一个完全事件)，可以得到：

&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?$$P(C_k,X)=P(x_1)P(x_2)...P(x_n)P(C_k)$$">

建模的目标是从每一个分类的概率
<img src="https://latex.codecogs.com/gif.latex?$$P_i(i\ink)$$">
中取最值：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?$$P_i(C_k|x_1,x_2,x_3...x_n)$$">

所以，在计算中贝叶斯公式中的分母可以去除。

综上，计算模型可描述为如下公式：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?\arg_{k\in\{1,...k\}}\max%20P(C_k)\prod_{i=1}^{n}P(x_i|C_k)">

## ***样例DEMO***

有总体样本数1000，转化样本数200。

其中，

&ensp;&ensp;&ensp;&ensp;
特征A在总体样本中出现200次，转化样本中出现80次；

&ensp;&ensp;&ensp;&ensp;
特征B在总体样本中出现800次，转化样本中出现120次。

转化概率用
<img src="https://latex.codecogs.com/gif.latex?$$P(C)$$">
表示，非转化概率用
<img src="https://latex.codecogs.com/gif.latex?$$P(-C)$$">
表示，则有：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?P(C)=200/1000=0.2,P(-C)=(1000-200)/1000=0.8">

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?P(A)=200/1000=0.2,P(A|C)=80/200=0.4,">
<img src="https://latex.codecogs.com/gif.latex?P(A|-C)=(200-80)/800=0.15">

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?P(B)=800/1000=0.8,P(B|C)=120/200=0.6,">
<img src="https://latex.codecogs.com/gif.latex?P(B|-C)=(800-120)/800=0.85">

可得：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?P(C|A,B)=P(C)*P(A|C)*P(B|C)=0.2*0.4*0.6=0.048">

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?P(-C|A,B)=P(-C)*P(A|-C)*P(B|-C)=0.8*0.15*0.85=0.102">

归一化处理：

&ensp;&ensp;&ensp;&ensp;
<img src="https://latex.codecogs.com/gif.latex?P(C|A,B):P(-C|A,B)=0.048:0.102=0.32:0.68">

## ***数据处理***
+ 数据去重，去除原始数据中出现的重复数据
+ 特征数据标准化，得到各个特征的字典值
+ 数据转换，根据特征字典值将原始数据转换为对应浮点型数据格式
+ 数据处理类库：pandas, numpy
+ 数据建模类库：sklearn.naive_bayes, sklearn.metrics

## ***代码实现***
<pre><code>

	from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
	from sklearn.model_selection import KFold
	import sklearn.metrics as mtx
	import pandas as pd
	import numpy as np

	# fit
	nb = GaussianNB()
	nb.fit(x_train_features, x_train_target)

	# predict
	pred_prob = nb.predict_proba(x_test_features)
	predicted = nb.predict(x_test_features)
	auc = mtx.roc_auc_score(x_test_target, pred_prob[:, 1])
	recall = mtx.recall_score(x_test_target, predicted)
	output = 'AUC: %(auc).4f  Recall: %(recall).4f'%{'auc':auc, 'recall':recall}
	print(output)

</code></pre>


## ***统计结果***


## ***评估结果***


## ***参考文献***
1. [Naive Bayes classifiers in TensorFlow](https://nicolovaligi.com/naive-bayes-tensorflow.html "Naive Bayes classifiers in TensorFlow")

2. 《统计学关我什么事：生活中的极简统计学》[小岛宽之]
