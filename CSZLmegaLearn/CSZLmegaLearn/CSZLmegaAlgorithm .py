#coding=utf-8

#EM算法
SIGMA = 6

EPS = 0.0001

def my_EM(X):
    k = 1
    N = len(X)
    Miu = np.random.rand(k,1)
    Posterior = np.mat(np.zeros((N,2)))
    dominator = 0
    numerator = 0
    #先求后验概率
    for iter in range(1000):
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0/(2.0*SIGMA**2) * (X[i] - Miu[j])**2)
                #print dominator,-1/(2*SIGMA**2) * (X[i] - Miu[j])**2,2*SIGMA**2,(X[i] - Miu[j])**2
                #return
            for j in range(k):
                numerator = np.exp(-1.0/(2.0*SIGMA**2) * (X[i] - Miu[j])**2)
                Posterior[i,j] = numerator/dominator			
        oldMiu = copy.deepcopy(Miu)
        #最大化	
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i,j] * X[i]
                dominator = dominator + Posterior[i,j]
            Miu[j] = numerator/dominator
        print ((abs(Miu - oldMiu)).sum())

        if (abs(Miu - oldMiu)).sum() < EPS:
            print (Miu,iter)
            break
    return Miu[0][0]