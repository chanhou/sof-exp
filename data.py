import os
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler

path = os.path.dirname('/data/clustering.disambiguation/sof/data/')

def pendigits():
    # path = os.path.dirname('/data/clustering.disambiguation/sof/data/pendigits/')
    xPen = []
    yPen = []
    with open(path+'pendigits/pendigits.tra','r')as f:
        for line in f:
            line = line.rsplit('\n')
            line = line[0].split(',')
            yPen.append(int(line[-1]))
            tmp = []
            for l in line[:-1]:
                tmp.append(int(l))
            xPen.append(tmp)
    with open(path+'pendigits/pendigits.tes.txt','r')as f:
        for line in f:
            line = line.rsplit('\n')
            line = line[0].split(',')
            yPen.append(int(line[-1]))
            tmp = []
            for l in line[:-1]:
                tmp.append(float(l))
            xPen.append(tmp)
    #print len(xPen)
    #print len(yPen)
    xPen = np.array(xPen)
#    a = StandardScaler()
#    xPen = a.fit_transform(xPen)
    yPen = np.array(yPen)

    return xPen, yPen

def iris():
    iris1 = datasets.load_iris()
    X = iris1.data
    y = iris1.target

    return X,y

def ecoli():
    xeco = []
    yeco = []
    qq = {}
    count = 0
    with open(path+'/ecoli/ecoli.data.txt','r')as f:
        for line in f:
            line = line.rsplit('\n')
            line = line[0].split()
            if line[-1] not in qq:
                qq[line[-1]]=count
                count += 1
            yeco.append(qq[line[-1]])
            tmp = []
            for l in line[1:-1]:
                tmp.append(float(l))
            xeco.append(tmp)

    xeco = np.array(xeco)
    yeco = np.array(yeco)
    
    return xeco, yeco
