import numpy as np
import scipy as sp
import scipy.spatial as ss
from scipy.special import beta,betaln,digamma,gamma
from math import log,pi,exp
import itertools
import time

def extend_torus(x):
	N = len(x)
	d = len(x[0])
	sign = (x>0.5)*2-1
	ret = []
	for i in range(N):
		for elem in itertools.product([0,1], repeat=d):
			ret.append(x[i]+elem*sign[i])
	return np.array(ret)

def log_regression(N_list, MSE_list):
	logN = np.array(map(log,N_list))
	logMSE = np.array(map(log,MSE_list))
	A = np.vstack([logN, np.ones(len(logN))]).T
	return np.linalg.lstsq(A, logMSE)[0]

def entropy(x, x_torus, k):
	N = len(x)
	d = len(x[0])

	tree = ss.cKDTree(x_torus)
	knn_dis = [tree.query(point,k+1,p=2)[0][k] for point in x]
	ans = -digamma(k)+log(N)+0.5*d*log(pi)-log(gamma(1+0.5*d))
	return ans + d*np.mean(map(log,knn_dis))

def experiment(d, s, k, N_list, T): 
	gt = d*(betaln(s+1,s+1)-2*s*digamma(s+1)+2*s*digamma(2*s+2))
	ret = []
	for n in N_list:
		MSE = 0.0
		for t in range(T):
			x = np.random.beta(s+1,s+1,size=(n,d))
			x_torus = extend_torus(x)
			est = entropy(x, x_torus, k)
			MSE += (est-gt)**2/T
		ret.append(np.sqrt(MSE))
	return ret

f = open("result.dat", 'w')
f.write("#Nearest Neighbor Entropy Estimator. X~Beta(s+1,s+1)^d. Average over 500 tries\n")
f.write("#d	s	k	N	sqMSE	slope	intercept\n")
start_time = time.time()

for d in [1,2,3,4]:
	for s in [0.5,1,1.5,2]:
		for k in [1,2,3,4,5,6,7,8,9,10]:
			N_list = [32,64,128,256,512,1024,2048]
			MSE_list = experiment(d, s, k, N_list, 512)
			m, c = log_regression(N_list, MSE_list)
			print("d=%d, s=%.1f, k=%d Finished! slope=%.4f, Intercept=%.4f, Time=%.4f"%(d,s,k,m,c,time.time()-start_time))
			for i in range(len(N_list)):
				f.write("%d	%.1f	%d	%d	%.4f	%.4f	%.4f\n"%(d,s,k,N_list[i],MSE_list[i],m,c))

