from sof.util import findMost
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sof.util import rand_score
import cPickle as pickle
import time

np.random.seed(102)
# 102 0.696428571429, 0.836425017768

class sof_c():
    def __init__(self, k=2, c=1, lamda1=1, lamda2=1, mu=2, gamma=0.00001, ep1=0.001, ep2=0.001):
        self.k = k
        self.c = c
        self.la1 = lamda1
        self.la2 = lamda2
        self.mu = mu
        self.ga = gamma
        self.ep1 = ep1
        self.ep2 = ep2

    def fit(x):
        N = x.shape[0]

        D1 = euclidean_distances(x,x)
        vec = findMost(D1,10)
        D1 = D1/np.sqrt(vec.reshape(1,N))/np.sqrt(vec.reshape(N,1))
    
        P = np.exp(-self.c*D1) # NxN
        w0 = np.random.rand(N, self.k) # NxK
        wt = np.copy(w0)
    
        grad = 4*(( w0.dot(w0.T)-P )).dot(w0)
        tmp = np.ones((N,self.k))
        tmp[w0>=0] = 0
        grad += -lamda1*(tmp*(np.ones((N, self.k))))
        grad += 2*lamda2*(w0.dot(np.ones((self.k,self.k)))-np.ones((N,self.k)))
        wt = wt - self.ga*grad
    
        count0 = 0
        while(self.la1 < 1./self.ep2 and self.la2 < 1./self.ep2):
            count1 = 0
#        print np.linalg.norm(wt-w0,1)
#        print np.linalg.norm(wt-w0,2)
            while( np.linalg.norm(wt-w0, 2) > self.ep1):
                
                w0 = np.copy(wt)
                grad = 4*(( w0.dot(w0.T)-P )).dot(w0)
                tmp = np.ones((N,self.k))
                tmp[w0>=0] = 0
                grad += -lamda1*(tmp*(np.ones((N,self.k))))
                grad += 2*lamda2*(w0.dot(np.ones((self.k,self.k)))-np.ones((N,self.k)))
                wt = wt - gamma*grad
                count1 += 1
                #print count1
                if count1 % 10==0:
                    print '\t',count1, np.linalg.norm(wt-w0,2)
            self.la1 = self.la1*self.mu
            self.la2 = self.la2*self.mu
            self.ga = self.ga/self.mu
            print count0, count1, self.la, self.la2, self.ga
            count0 += 1
            if count1==0:
                print count0, self.la1, self.la2, self.ga
                break
        self.wt = wt

    def predict():
        return np.argmax(wt,axis=1)

    def result(y):
        return 'nan'
        

def sof(I, y, k, c, lamda1, lamda2, mu, gamma, ep1, ep2):
    N = I.shape[0]
    
    D1 = euclidean_distances(I,I)
    vec = findMost(D1,10)
    D1 = D1/np.sqrt(vec.reshape(1,N))/np.sqrt(vec.reshape(N,1))
    
    P = np.exp(-c*D1) # NxN
    wt = np.random.rand(N, k) # NxK
    #wt = np.copy(w0)
    
    max_accu = 0
    max_ri = 0

    count0 = 0
    while(lamda1 < 1./ep2 and lamda2 < 1./ep2):
        count1 = 0
        w0 = np.copy(wt)
        grad = 4*(( w0.dot(w0.T)-P )).dot(w0)
        tmp = np.ones((N,k))
        tmp[w0>=0] = 0
        grad += -lamda1*(tmp*(np.ones((N,k))))
        grad += 2*lamda2*(w0.dot(np.ones((k,k)))-np.ones((N,k)))
        wt = wt - gamma*grad
#        print np.linalg.norm(wt-w0,1)
#        print np.linalg.norm(wt-w0,2)
        t0 = time.time()
        while( np.linalg.norm(wt-w0, 2) > ep1):
            
            w0 = np.copy(wt)
            grad = 4*(( w0.dot(w0.T)-P )).dot(w0)
            tmp = np.ones((N,k))
            tmp[w0>=0] = 0
            grad += -lamda1*(tmp*(np.ones((N,k))))
            grad += 2*lamda2*(w0.dot(np.ones((k,k)))-np.ones((N,k)))
            wt = wt - gamma*grad
            count1 += 1
            #print count1

            y_pred_old = np.argmax(wt,axis=1)
            row, col = linear_sum_assignment(-confusion_matrix(y, y_pred_old))
            y_pred = np.copy(y_pred_old)
            for i, q in enumerate(col):
                y_pred[y_pred_old==q] = i

            if accuracy_score(y,y_pred) > max_accu:
                max_accu = accuracy_score(y,y_pred)
                #print count1, 'accu,' , max_accu
            if rand_score(y, y_pred) > max_ri:
                max_ri = rand_score(y, y_pred)
                #print count1, 'RI, ', max_ri

            if count1 % 500==0:
                print '\t',count1, np.linalg.norm(wt-w0,2)
            if count1 % 500 ==0:
            #    with open('./result'+str(count1)+'.pkl','wb') as w:
            #        pickle.dump(wt,w)
#                y_pred_old = np.argmax(wt,axis=1)
#                row, col = linear_sum_assignment(-confusion_matrix(y, y_pred_old))
#                y_pred = np.copy(y_pred_old)
#                for i, q in enumerate(col):
#                    y_pred[y_pred_old==q] = i
          
                print 'absolute sum', np.sum(np.sum(wt,axis=1))
                print confusion_matrix(y, y_pred)
                print 'time, ', time.time()-t0
                # print 'ARI, ', adjusted_rand_score(y,y_pred)
                print 'max RI, ', max_ri #rand_score(y, y_pred)
                print 'max Accu, ', max_accu #accuracy_score(y,y_pred)
                print 'RI, ', rand_score(y, y_pred)
                print 'Accu, ', accuracy_score(y,y_pred)
                print ''

        lamda1 = lamda1*mu
        lamda2 = lamda2*mu
        gamma = gamma/mu
        print count0, count1, lamda1, lamda2, gamma
        count0 += 1
        if count1==0:
            print count0, lamda1, lamda2, gamma
            break
        #break
    return np.argmax(wt,axis=1)

