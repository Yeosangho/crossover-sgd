import numpy as np
import math
from copy import copy, deepcopy
def select_layer(rseed_per_rank, world_size, roulettes):
	select_rank = []
	copy_roulettes = deepcopy(roulettes)
	for i in range(0,world_size):
		for j in select_rank :
				copy_roulettes[i][j] = 0
		roulette_sum = sum(copy_roulettes[i])
		copy_roulettes[i] = [ r/roulette_sum for r in copy_roulettes[i]]
		if((world_size-2 == i) and ((world_size-1) not in select_rank)):
			select_rank.append(world_size -1)
		else:
			select_rank.append(rseed_per_rank[0].choice(world_size, p=copy_roulettes[i]))
	return select_rank
def make_mixing_matrix(world_size, select_rank):
	mat = np.zeros((world_size,world_size))
	for i in range(0, world_size):
		mat[i][i] = 1/2
		mat[i][select_rank[i]] = 1/2
	return mat
def make_directional_exponential_graph(iter_num, world_size, select_rank):
	mat = np.zeros((world_size, world_size))
	iter_num = iter_num + math.log(world_size,2)
	for i in range(0, world_size):
		mat[i][i] = 1/2
		c = (i+(2**(iter_num%(math.log(world_size,2)))))%world_size
		b = (i-(2**(iter_num%(math.log(world_size,2)))))%world_size
		print(f"rank {i} send {c} recv from {b} ")
		c = int(c)
		mat[i][c] = 1/2
	return mat 
if __name__ == '__main__':
	world_size = 32
	roulettes = [] * world_size
	rseed_per_rank = []
	for i in range(world_size +1):
		rseed_per_rank.append(np.random.RandomState(i+3))
	for i in range(0, world_size):
		roulette_except_rank = [1./(world_size -1) for i in range(0, world_size)]
		roulette_except_rank[i] = 0
		roulettes.append(roulette_except_rank)
	select_rank = select_layer(rseed_per_rank, world_size, roulettes)
	mixing_matrix = make_mixing_matrix(world_size, select_rank)
	for i in range(1, 32):	
		select_rank = select_layer(rseed_per_rank, world_size, roulettes)
		mixing_matrix_op = make_mixing_matrix(world_size, select_rank)	
		#mixing_matrix_op = mixing_matrix
		mixing_matrix = mixing_matrix_op.dot(mixing_matrix)
		u, s, v = np.linalg.svd(mixing_matrix)
		#print(i)
		#print(mixing_matrix)
		print(s[1])
		#print(mixing_matrix)
	mixing_matrix = make_directional_exponential_graph(0, world_size, select_rank)
	print("------------------------")
	for i in range(1, 16):
		mixing_matrix_op = make_directional_exponential_graph(i, world_size, select_rank)
		mixing_matrix = mixing_matrix_op.dot(mixing_matrix)
		#print(mixing_matrix)
		u, s, v = np.linalg.svd(mixing_matrix)
		print(s[1])

	A = np.ones((world_size, world_size)) * 1/16
	A = A.dot(A)
	u, s, v = np.linalg.svd(A)
	print(s[1])

	#print(A)

