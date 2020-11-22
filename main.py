import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
# K-means
# best_seed = np.zeros(4, dtype=np.int)
# for k in range(1, 5):
#     best_cost = np.inf
#     for seed in range(0, 5):
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = kmeans.run(X, mixture, post)
#         if cost < best_cost:
#             best_cost = cost
#             best_seed[k-1] = seed
#     # import pdb; pdb.set_trace()
#     print(f'Cost at k = {k} is {best_cost}')
#     mixture, post = common.init(X, k, best_seed[k-1])
#     mixture, post, cost = kmeans.run(X, mixture, post)
#     common.plot(X, mixture, post, f"K-means for K={k}")
#
# # EM Algorithm
# best_seed = np.zeros(4, dtype=np.int)
# for k in range(1, 5):
#     best_cost = np.inf
#     for seed in range(0, 5):
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = naive_em.run(X, mixture, post)
#         if cost < best_cost:
#             best_cost = cost
#             best_seed[k-1] = seed
#     # import pdb; pdb.set_trace()
#     print(f'Cost at k = {k} is {best_cost}')
#     mixture, post = common.init(X, k, best_seed[k-1])
#     mixture, post, cost = naive_em.run(X, mixture, post)
#     common.plot(X, mixture, post, f"EM for K={k}")

# Best K according to BIC
# best_seed = np.zeros(4, dtype=np.int)
# for k in range(1, 5):
#     best_cost = np.inf
#     for seed in range(0, 5):
#         mixture, post = common.init(X, k, seed)
#         mixture, post, cost = naive_em.run(X, mixture, post)
#         cost = common.bic(X, mixture, cost)
#         if cost < best_cost:
#             best_cost = cost
#             best_seed[k-1] = seed
#     # import pdb; pdb.set_trace()
#     print(f'Cost at k = {k} is {best_cost}')
#     # mixture, post = common.init(X, k, best_seed[k-1])
#     # mixture, post, cost = naive_em.run(X, mixture, post)
#     # common.plot(X, mixture, post, f"EM for K={k}")

# EM for collaborative filtering
# X = np.loadtxt("netflix_incomplete.txt")
# k = [1, 12]
# best_seed = np.zeros(2, dtype=np.int)
# for j in range(2):
#     best_cost = -np.inf
#     for seed in range(0, 5):
#         mixture, post = common.init(X, k[j], seed)
#         mixture, post, cost = em.run(X, mixture, post)
#         # cost = common.bic(X, mixture, cost)
#         if cost > best_cost:
#             best_cost = cost
#             best_seed[j] = seed
#     # import pdb; pdb.set_trace()
#         print(f'Cost at k = {k[j]} with seed = {best_seed} is {best_cost}')
#     print(f'Best Cost at k = {k[j]} with seed = {best_seed} is {best_cost}')

# RMSE error with complete data
seed = 1
k = 12
X = np.loadtxt("netflix_incomplete.txt")
mixture, post = common.init(X, k, seed)
mixture, post, cost = em.run(X, mixture, post)
X_pred = em.fill_matrix(X, mixture)
X_gold = np.loadtxt("netflix_complete.txt")
rmse = common.rmse(X_pred, X_gold)
print(f'RMSE = {rmse}')
