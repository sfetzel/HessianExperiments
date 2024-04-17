from torch import eye, zeros


def S_matrix(n):
    n_2 = n**2
    S = zeros((n_2,n_2))

    for k in range(n_2):
        q = k // n
        r = k % n

        l = r*n + q

        S[k, l] = 1
    return S

print(S_matrix(2))
print(S_matrix(3))