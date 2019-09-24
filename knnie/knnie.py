#Written by Weihao Gao from UIUC

import scipy.spatial as ss
import scipy.stats as sst
from scipy.special import digamma,gamma
from sklearn.neighbors import KernelDensity
from math import log,pi,exp
import numpy.random as nr
import numpy as np
import random
import time
import matplotlib.pyplot as plt
#from cvxopt import matrix,solvers


#Usage Functions
def kraskov_mi(x,y,k=5):

    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    dx = len(x[0])
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans_xy = -digamma(k) + digamma(N) + (dx+dy)*log(2)#2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
    ans_x = digamma(N) + dx*log(2)
    ans_y = digamma(N) + dy*log(2)
    for i in range(N):
        ans_xy += (dx+dy)*log(knn_dis[i])/N
        ans_x += -digamma(len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dx*log(knn_dis[i])/N
        ans_y += -digamma(len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dy*log(knn_dis[i])/N

    return ans_x+ans_y-ans_xy


# def revised_mi(x,y,k=5,q=float('inf')):
# 	'''
# 		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
# 		Using *REVISED* KSG mutual information estimator (see arxiv.org/abs/1604.03006)
#
# 		Input: x: 2D list of size N*d_x
# 		y: 2D list of size N*d_y
# 		k: k-nearest neighbor parameter
# 		q: l_q norm used to decide k-nearest distance
#
# 		Output: one number of I(X;Y)
# 	'''
#
# 	assert len(x)==len(y), "Lists should have same length"
#    	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
# 	N = len(x)
#    	dx = len(x[0])
# 	dy = len(y[0])
# 	data = np.concatenate((x,y),axis=1)
#
#    	tree_xy = ss.cKDTree(data)
# 	tree_x = ss.cKDTree(x)
# 	tree_y = ss.cKDTree(y)
#
#    	knn_dis = [tree_xy.query(point,k+1,p=q)[0][k] for point in data]
# 	ans_xy = -digamma(k) + log(N) + vd(dx+dy,q)
# 	ans_x = log(N) + vd(dx,q)
# 	ans_y = log(N) + vd(dy,q)
# 	for i in range(N):
# 		ans_xy += (dx+dy)*log(knn_dis[i])/N
# 		ans_x += -log(len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=q))-1)/N+dx*log(knn_dis[i])/N
# 		ans_y += -log(len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=q))-1)/N+dy*log(knn_dis[i])/N
# 	return ans_x+ans_y-ans_xy
#
#
# def kraskov_multi_mi(x,y,z,k=5):
# 	'''
# 		Estimate the multivariate mutual information I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y,Z)
# 		of X, Y and Z from samples {x_i, y_i, z_i}_{i=1}^N
# 		Using KSG mutual information estimator
#
# 		Input: x: 2D list of size N*d_x
# 		y: 2D list of size N*d_y
# 		z: 2D list of size N*d_z
# 		k: k-nearest neighbor parameter
#
# 		Output: one number of I(X;Y;Z)
# 	'''
#
# 	assert len(x)==len(y), "Lists should have same length"
# 	assert len(x)==len(z), "Lists should have same length"
#    	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
# 	N = len(x)
#    	dx = len(x[0])
# 	dy = len(y[0])
# 	dz = len(z[0])
# 	data = np.concatenate((x,y,z),axis=1)
#
#    	tree_xyz = ss.cKDTree(data)
# 	tree_x = ss.cKDTree(x)
# 	tree_y = ss.cKDTree(y)
# 	tree_z = ss.cKDTree(z)
#
#    	knn_dis = [tree_xyz.query(point,k+1,p=float('inf'))[0][k] for point in data]
# 	ans_xyz = -digamma(k) + digamma(N) + (dx+dy+dz)*log(2)#2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
# 	ans_x = digamma(N) + dx*log(2)
# 	ans_y = digamma(N) + dy*log(2)
# 	ans_z = digamma(N) + dz*log(2)
# 	for i in range(N):
# 		ans_xyz += (dx+dy+dz)*log(knn_dis[i])/N
# 		ans_x += -digamma(len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dx*log(knn_dis[i])/N
# 		ans_y += -digamma(len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dy*log(knn_dis[i])/N
# 		ans_z += -digamma(len(tree_z.query_ball_point(z[i],knn_dis[i]-1e-15,p=float('inf'))))/N+dz*log(knn_dis[i])/N
#
# 	return ans_x+ans_y+ans_z-ans_xyz
#
#
# def revised_multi_mi(x,y,z,k=5,q=float('inf')):
#
# 	'''
# 		Estimate the multivariate mutual information I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y,Z)
# 		of X, Y and Z from samples {x_i, y_i, z_i}_{i=1}^N
# 		Using *REVISED* KSG mutual information estimator (see arxiv.org/abs/1604.03006)
#
# 		Input: x: 2D list of size N*d_x
# 		y: 2D list of size N*d_y
# 		z: 2D list of size N*d_z
# 		k: k-nearest neighbor parameter
# 		q: l_q norm used to decide k-nearest neighbor distance
#
# 		Output: one number of I(X;Y;Z)
# 	'''
# 	assert len(x)==len(y), "Lists should have same length"
# 	assert len(x)==len(z), "Lists should have same length"
#    	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
# 	N = len(x)
#    	dx = len(x[0])
# 	dy = len(y[0])
# 	dz = len(z[0])
# 	data = np.concatenate((x,y,z),axis=1)
#
#    	tree_xyz = ss.cKDTree(data)
# 	tree_x = ss.cKDTree(x)
# 	tree_y = ss.cKDTree(y)
# 	tree_z = ss.cKDTree(z)
#
#    	knn_dis = [tree_xyz.query(point,k+1,p=q)[0][k] for point in data]
# 	ans_xyz = -digamma(k) + log(N) + vd(dx+dy+dz,q)
# 	ans_x = log(N) + vd(dx,q)
# 	ans_y = log(N) + vd(dy,q)
# 	ans_z = log(N) + vd(dz,q)
# 	for i in range(N):
# 		ans_xyz += (dx+dy+dz)*log(knn_dis[i])/N
# 		ans_x += -log(len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=q))-1)/N+dx*log(knn_dis[i])/N
# 		ans_y += -log(len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=q))-1)/N+dy*log(knn_dis[i])/N
# 		ans_z += -log(len(tree_z.query_ball_point(z[i],knn_dis[i]+1e-15,p=q))-1)/N+dz*log(knn_dis[i])/N
# 	return ans_x+ans_y+ans_z-ans_xyz

#
# #Auxilary functions
# def vd(d,q):
# 	# Compute the volume of unit l_q ball in d dimensional space
# 	if (q==float('inf')):
# 		return d*log(2)
# 	return d*log(2*gamma(1+1.0/q)) - log(gamma(1+d*1.0/q))
#
# def entropy(x,k=5,q=float('inf')):
# 	# Estimator of (differential entropy) of X
# 	# Using k-nearest neighbor methods
#    	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
# 	N = len(x)
#    	d = len(x[0])
# 	thre = 3*(log(N)**2/N)**(1/d)
#    	tree = ss.cKDTree(x)
#    	knn_dis = [tree.query(point,k+1,p=q)[0][k] for point in x]
# 	truncated_knn_dis = [knn_dis[s] for s in range(N) if knn_dis[s] < thre]
# 	ans = -digamma(k) + digamma(N) + vd(d,q)
# 	return ans + d*np.mean(map(log,knn_dis))
#
# def kde_entropy(x):
# 	# Estimator of (differential entropy) of X
# 	# Using resubstitution of KDE
# 	N = len(x)
#    	d = len(x[0])
# 	local_est = np.zeros(N)
# 	for i in range(N):
# 		kernel = sst.gaussian_kde(x.transpose())
# 		local_est[i] = kernel.evaluate(x[i].transpose())
# 	return -np.mean(map(log,local_est))
#
#








