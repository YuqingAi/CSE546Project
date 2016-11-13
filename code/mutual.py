import numpy as np

def MI(features, labels, Q=12):
    edges = np.zeros(len(features[0]), Q+1)
    for k in range(features[0]):
        minval = min(features[:, k])
        maxval = max(features[:, k])
        if minval == maxval:
            continue
        quantlevels = mslice[minval:(maxval - minval) / 500:maxval]
        N = histc(features(mslice[:], k), quantlevels)
        totsamples = len(features)
        N_cum = cumsum(N)
        edges[k, 1].lvalue = -Inf
        stepsize = totsamples / Q
        for j in mslice[1:Q - 1]:
            a = find(N_cum > j *elmul* stepsize, 1)
            edges(k, j + 1).lvalue = quantlevels(a)
        edges(k, j + 2).lvalue = Inf

    S = np.zeros(len(features),l(features))
    for k in mslice[1:len(S[0])]:
        S(mslice[:], k).lvalue = quantize(features(mslice[:], k), edges(k, mslice[:])) + 1

    I = np.zeros(len(features[0]), 1)
    for k in mslice[1:size(features, 2)]:
        I(k).lvalue = computeMI(S(mslice[:], k), labels, 0)

    [weights, features] = sort(I, mstring('descend'))


def computeMI(seq1, seq2, lag=0):
    if (length(seq1) != length(seq2)):
        error(mstring('Input sequences are of different length'))

	lambda1 = max(seq1)
    symbol_count1 = zeros(lambda1, 1)

    for k in mslice[1:lambda1]:
        symbol_count1(k).lvalue = sum(seq1 == k)
    end

    symbol_prob1 = symbol_count1 /eldiv/ sum(symbol_count1) + 0.000001

    lambda2 = max(seq2)
    symbol_count2 = zeros(lambda2, 1)

    for k in mslice[1:lambda2]:
        symbol_count2(k).lvalue = sum(seq2 == k)
    end

    symbol_prob2 = symbol_count2 /eldiv/ sum(symbol_count2) + 0.000001

    M = zeros(lambda1, lambda2)
    if (lag > 0):
        for k in mslice[1:length(seq1) - lag]:
            loc1 = seq1(k)

            loc2 = seq2(k + lag)

            M(loc1, loc2).lvalue = M(loc1, loc2) + 1
        end
    else:
        for k in mslice[abs(lag) + 1:length(seq1)]:
            loc1 = seq1(k)

            loc2 = seq2(k + lag)

            M(loc1, loc2).lvalue = M(loc1, loc2) + 1
        end
    end

    SP = symbol_prob1 * symbol_prob2.cT

    M = M /eldiv/ sum(M(mslice[:])) + 0.000001

    I = sum(sum(M *elmul* log2(M /eldiv/ SP)))

def quantize(x, q):
    x = x(mslice[:])
    nx = length(x)
    nq = length(q)
    y = sum(repmat(x, 1, nq) > repmat(q, nx, 1), 2)