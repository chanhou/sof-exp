import time
#from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from sof.sofgd import sof
from sof.util import rand_score 
from sof.data import iris, pendigits, ecoli


def train(times, X, y,c, lea, ep1, ep2, lamda1, lamda2 ):
    t0 = time.time()
#     times = 1
    # for lea in [0.0001, 0.00001, 0.000001]:
#    lea = .00001
    print 'learn={}, ep1={}, ep2={}, la1={}, la2={}'.format(lea, ep1, ep2, lamda1, lamda2)
    ari,ri,accu = [], [], []
    for ddd in range(times):
        y_pred_old = sof(X, y, k=len(np.unique(y)), c=1, 
                                 lamda1=lamda1,lamda2=lamda2, mu=2, 
                                 gamma=lea, ep1=ep1, ep2=ep2 )
        row, col = linear_sum_assignment(-confusion_matrix(y, y_pred_old))
        y_pred = np.copy(y_pred_old)
        for i, q in enumerate(col):
            y_pred[y_pred_old==q] = i
        ari.append( adjusted_rand_score(y,y_pred) )
        ri.append(rand_score(y, y_pred))
        accu.append(accuracy_score(y,y_pred))
        print '\taccu={}, RI={}'.format(accuracy_score(y,y_pred),rand_score(y, y_pred))
    # print 'ARI: ', adjusted_rand_score(y,y_pred)
    # print 'RI: ', rand_score(y, y_pred)
    # print 'Accu: ', accuracy_score(y,y_pred)
            

    print confusion_matrix(y, y_pred)
    # print y_pred
    print 'time, ', time.time()-t0
    print 'title\tmax\tmean\tstd'
    print 'ARI, ', np.array(ari).max(), np.array(ari).mean(), np.array(ari).std()
    print 'RI, ', np.array(ri).max(), np.array(ri).mean(), np.array(ri).std()
    print 'Accu, ', np.array(accu).max(), np.array(accu).mean(), np.array(accu).std()
    print ''

def main():
#    iris = datasets.load_iris()
#    X = iris.data
#    y = iris.target
   
    #train(20, X, y)
    # train(times, X, y, lea, ep1, ep2, lamda1, lamda2 )
    times = 20
#    train(times, X, y, 0.00001, .0001, .0001, 10, 10 )
#    train(times, X, y, 0.00001, .0001, .0001, 1, 1 )
#    train(times, X, y, 0.00001, .0001, .0001, 0.1, 1 )
#    train(times, X, y, 0.00001, .0001, .0001, 1, 0.1 )
#    train(times, X, y, 0.00001, .0001, .0001, 0.1, 0.1 )
#    train(times, X, y, 0.00001, .0001, .0001, 0.01, 0.01 )
#    train(times, X, y, 0.00001, .0001, .0001, 0.001, 0.001 )
#    train(times, X, y, 0.00001, .0001, .0001, 0.0001, 0.0001 )
#    train(times, X, y, 0.0001, .0001, .0001, 10, 10 )
#    train(times, X, y, 0.0001, .0001, .0001, 1, 1 )
#    train(times, X, y, 0.0001, .0001, .0001, 0.1, 1 )
#    train(times, X, y, 0.0001, .0001, .0001, 1, 0.1 )
#    train(times, X, y, 0.0001, .0001, .0001, 0.1, 0.1 )
#    train(times, X, y, 0.0001, .0001, .0001, 0.01, 0.01 )
#    train(times, X, y, 0.0001, .0001, .0001, 0.001, 0.001 )
#    train(times, X, y, 0.0001, .0001, .0001, 0.0001, 0.0001 )

#    for leaa in [0.0001, 0.00001]:
#        for la1 in ([10,1,0.1,0.01,0.001,0.0001]):
#            for la2 in ([10,1,0.1,0.01,0.001,0.0001]):
#                train(times, X, y, leaa, .0001, .0001, la1, la2 )

    #xPen, yPen = pendigits()
    xPen, yPen = ecoli()
    train(1, xPen, yPen, 1, 0.00001, .0001, .0001, 10, 1 )
    # 0.0001, 0.1, 0.1 -> 
    #ARI,  0.436456539995 0.436456539995 0.0
    #RI,  0.808066808813 0.808066808813 0.0
    #Accu,  0.619047619048 0.619047619048 0.0

    #0.00001, .0001, .0001, 10, 0.1
    
    # eco, seed 102
    # 1, 0.00001, .0001, .0001, 10, 0.1 )
    # max RI,  0.836425017768
    # max Accu,  0.696428571429

