import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv(r'd:\thuchanh\train.csv')
# Sample a smaller subset for speed
data_small = data.iloc[:5, :10]
obs_flat = data_small.astype(str).values.flatten()
symbols, inv = np.unique(obs_flat, return_inverse=True)
obs_seq = inv.reshape(data_small.shape)

# Negative log-likelihood function
def neg_log_likelihood(params, obs, K=2):
    M = len(symbols)
    idx = 0
    pi_scores = params[idx:idx+K]; idx += K
    A_scores  = params[idx:idx+K*K].reshape(K, K); idx += K*K
    B_scores  = params[idx:idx+K*M].reshape(K, M)
    def softmax(x, axis=None):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)
    pi = softmax(pi_scores)
    A  = softmax(A_scores, axis=1)
    B  = softmax(B_scores, axis=1)
    
    N, T = obs.shape
    log_alpha = np.zeros((N, K, T))
    for i in range(N):
        log_alpha[i,:,0] = np.log(pi) + np.log(B[:, obs[i,0]])
    for t in range(1, T):
        for i in range(N):
            prev = log_alpha[i,:,t-1][:, None] + np.log(A)
            log_alpha[i,:,t] = np.log(B[:, obs[i,t]]) + np.logaddexp.reduce(prev, axis=0)
    return -np.sum(np.logaddexp.reduce(log_alpha[:,:, -1], axis=1))

# Niching PSO implementation
def niching_pso(obs, dim, swarm_size=30, num_species=3, iters=50, cluster_freq=5):
    w, c1, c2 = 0.5, 1.5, 1.5
    X = np.random.randn(swarm_size, dim)
    V = np.zeros_like(X)
    pbest = X.copy()
    pbest_scores = np.array([neg_log_likelihood(x, obs) for x in X])
    
    # initial species determination
    labels = KMeans(n_clusters=num_species, n_init=5).fit_predict(X)
    species_best = np.zeros((num_species, dim))
    species_score = np.full(num_species, np.inf)
    for s in range(num_species):
        idxs = np.where(labels == s)[0]
        best_idx = idxs[np.argmin(pbest_scores[idxs])]
        species_best[s] = pbest[best_idx]
        species_score[s] = pbest_scores[best_idx]
    
    # PSO loop with niching
    for it in range(iters):
        if it % cluster_freq == 0:
            labels = KMeans(n_clusters=num_species, n_init=5).fit_predict(X)
            for s in range(num_species):
                idxs = np.where(labels == s)[0]
                best_idx = idxs[np.argmin(pbest_scores[idxs])]
                species_best[s] = pbest[best_idx]
                species_score[s] = pbest_scores[best_idx]
        for i in range(swarm_size):
            s = labels[i]
            V[i] = w*V[i] + c1*np.random.rand()*(pbest[i] - X[i]) + c2*np.random.rand()*(species_best[s] - X[i])
            X[i] += V[i]
            score = neg_log_likelihood(X[i], obs)
            if score < pbest_scores[i]:
                pbest[i] = X[i].copy()
                pbest_scores[i] = score
                if score < species_score[s]:
                    species_best[s] = X[i].copy()
                    species_score[s] = score
    return species_best, species_score

# Run niching PSO
K = 2
dim = K + K*K + K*len(symbols)
species_params, species_scores = niching_pso(obs_seq, dim)

# Decode results
pis = []
for params in species_params:
    idx = 0
    pi = np.exp(params[idx:idx+K])
    pi /= pi.sum()
    pis.append(pi)

# Plot 1: Negative Log-Likelihood per Species
plt.figure()
plt.bar(range(len(species_scores)), species_scores)
plt.xlabel('Species')
plt.ylabel('Negative Log-Likelihood')
plt.title('NegLogLikelihood for each Species')
plt.show()

# Plot 2: Initial State Probabilities per Species
labels = ['Species ' + str(i) for i in range(len(pis))]
pi_matrix = np.array(pis)
x = np.arange(len(pis))
width = 0.35

plt.figure()
plt.bar(x - width/2, pi_matrix[:, 0], width)
plt.bar(x + width/2, pi_matrix[:, 1], width)
plt.xlabel('Species')
plt.ylabel('Probability')
plt.title('Initial State Probabilities (π) per Species')
plt.xticks(x, labels)
plt.legend(['π1', 'π2'])
plt.show()
