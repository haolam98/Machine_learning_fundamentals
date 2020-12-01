import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Unlike previous algorithm, k mean clustering is the unsupervised learning algorithm
# we're going to give machine the data, but we are not going tell machine what the data is

#load data
digits = load_digits()
# since our data is going to be huge, we scale the data down to save time on the computation
# scale down into a range between -1 to 1
# .data parts is all our features
data = scale(digits.data)

# get labels
y = digits.target

#set the amount of centroid to make
k = len(np.unique(y))

# define how many samples and features we have by getting the data set shape
# - get the amount of instances - amount of number we have that  are going to be classified,
# - and, get the amount of features that go along with that data
samples,features = data.shape

#use provided fucntion from sklearn for scoring
# -we give the classifier(estimator) that fit our data, then it essentially will use different things to score it
# -we don't have to split test to train data, it's going to generate automaticaly y value for every single test data
# point that we given
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

#euclidean distance = a distance between 2 points
# init -> how points are set initially
# n_init = 10 -> run 10 times and get the best
clf = KMeans(n_clusters=k, init= "random", n_init=10)

bench_k_means(clf,"1",data)
# it will print out all the accuracies when the function is called