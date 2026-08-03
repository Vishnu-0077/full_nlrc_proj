import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def gls_neuron_gen(X,b): #skew binary method
    '''
    Docstring for gls_neuron_gen
    
    :param X: the single box of a 2d input, (single float value between 0 and 1)
    :param b: threshold
    :param ss: symbolic sequence where the binary values are appended
    '''
    if X>=1:
        X = 1-10e-5
    elif X<=0:
        X = 10e-5
    if 1>X>=b:
        return (1-X)/(1-b)
    return X/b

def iterate_gls_neuron(x,target_neighbor,eps):
    ss = [x]
    max_iter = 1000
    k = 0
    while abs(x-target_neighbor)>eps and k<max_iter-1:
        x = gls_neuron_gen(x,0.5)
        ss.append(x)
        k+=1
    return ss

def ss_to_binary(ss,thres):
    return (np.array(ss)>thres).astype(int)

def firing_rate(ss,thres=0.5):
    ss_bin = ss_to_binary(ss,thres)
    return np.count_nonzero(ss_bin)/len(ss_bin)

def entropy(ss,thres=0.5):
    ss_bin = ss_to_binary(ss,thres)
    p = np.count_nonzero(ss_bin)/len(ss_bin)
    range = 1e-10
    p = np.clip(p,range,1-range)
    return -(p*np.log2(p)) - ((1-p)*np.log2(1-p))

def energy(ss):
    if len(ss) ==0:
        return 0
    ss = np.array(ss)
    ss = np.clip(ss,10e-5,10)
    return np.mean(np.square(ss))
    
def firing_time(ss):
    max_iter = 1000
    return len(ss)/max_iter

def variance(ss):
    if len(ss) ==0:
        return 0
    return np.var(ss)

def extract_featues(x,target_neighbor_list,eps):
    (m,n) = x.shape
    c = len(target_neighbor_list)
    ttss = np.zeros((m,5*n*c))
    for i in range(m):
        for j in range(n):
            for k in range(c):
                ss = iterate_gls_neuron(x[i][j],target_neighbor_list[k],eps)
                ttss[i][(k*5)+(j*c*5)+0] = firing_rate(ss,0.5)
                ttss[i][(k*5)+(j*c*5)+1] = entropy(ss,0.5)
                ttss[i][(k*5)+(j*c*5)+2] = energy(ss)
                ttss[i][(k*5)+(j*c*5)+3] = firing_time(ss)
                ttss[i][(k*5)+(j*c*5)+4] = variance(ss)
    return ttss


def model_train(X_train,y_train): #train
    pipe = SVC(kernel='linear')
    pipe.fit(X_train, y_train)
    return pipe

X,y = load_iris(return_X_y=True)
mm = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)
print(f'shape of the input is {X.shape}')
def main():
    
    eps = 0.1
    target_neighbor_list = np.linspace(0.1,0.9,10)
    print(f'number of target neighbors is {len(target_neighbor_list)}')

    
    train_ttss = extract_featues(X_train,target_neighbor_list,eps)
    print(f'shape of the train data is {X_train.shape}')
    print(f'shape of the tranformed train data is {train_ttss.shape}')
    test_ttss = extract_featues(X_test,target_neighbor_list,eps)
    print(f'shape of the test data is {X_test.shape}')
    print(f'shape of transformed test data is {test_ttss.shape}')


    model = model_train(train_ttss,y_train)
    y_pred = model.predict(test_ttss)
    return accuracy_score(y_test,y_pred)


print(main())