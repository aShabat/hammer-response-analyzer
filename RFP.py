import numpy as np
import scipy as sp

# RFP is an algorithm from ./math/global_frequency_and_damping_estimates_from_frm.pdf, wich aplies algorithm from ./math/parameter_estimation_from_frm_using_rfp.pdf
# to multiple measurements.


def orthogonal_polynomials(omega, q, k):
    # k up to 30, or error becomes too big
    q = np.real(q)

    def qdot(left, right):
        return np.sum(q * left * right)

    f = np.array([np.ones(omega.shape), omega], dtype=np.double)
    s = np.zeros((2, k + 1), dtype=np.double)
    s[0, 0] = 1
    s[1, 1] = 1
    for i in range(2, k + 1):
        alpha = qdot(f[i - 2], omega * f[i - 1]) / qdot(f[i - 2], f[i - 2])
        f = np.append(f, [omega * f[i - 1] - alpha * f[i - 2]], axis=0)
        s = np.append(s, [np.roll(s[i - 1], 1) - alpha * s[i - 2]], axis=0)

    for i in range(k + 1):
        gamma = np.sqrt(qdot(f[i], f[i]))
        f[i] = f[i] / gamma
        s[i] = s[i] / gamma

    f = (1j) ** (np.arange(k + 1)) * f.transpose()
    for i in range(k + 1):
        s[i] = s[i] * np.real(1j**i / (1j ** np.arange(k + 1)))
    s = s.transpose()

    # f -- orthogonal polynomials for all omega, s -- transformation matrix
    return f, s


# omega - array of frequencies, Hs = 3d array with first two dimensions relating to points where we apply and record signal. Along the third dimension we have recorded FRF.
# m and n are assumed degrees of denominator and divisor of FRF. Number of modes = 2 * n. m >= n
# Most matrix variable names are copied from ./math/parameter_estimation_from_frm_using_rfp.pdf.
def rfp(omega, Hs, m, n):
    omegaMax = np.max(omega)
    omega = omega / omegaMax
    Utotal = np.zeros((0, n + 1))
    Vtotal = np.zeros(0)
    Cs = np.zeros(Hs.shape[:-1] + (m + 1,))
    Ks = np.zeros(Hs.shape[:-1] + (m + 1, n + 1))
    As = np.zeros(Hs.shape[:-1] + (m + 1,))
    for I in range(Hs.shape[0]):
        for J in range(Hs.shape[1]):
            H = Hs[I, J]
            if np.any(np.isnan(H)):
                continue
            P, Sa = orthogonal_polynomials(omega, np.ones_like(omega), m)
            T, Sb = orthogonal_polynomials(omega, np.conj(H) * H, n)
            Sbi = np.linalg.inv(Sb)
            W = H * T[:, n]
            T = H[:, np.newaxis] * T[:, :n]

            X = -np.real(np.conj(P.transpose()) @ T)
            Ud = np.identity(n) - X.transpose() @ X
            U = np.append(Ud @ Sbi[:n], [Sbi[n]], axis=0)
            V = np.append(-X.transpose() @ np.real(np.conj(P.transpose()) @ W), [1])

            Ks[I, J] = -Sa @ X @ Sbi[:n]
            Cs[I, J] = Sa @ np.real(np.conj(P.transpose()) @ W)

            Utotal = np.append(Utotal, U, axis=0)
            Vtotal = np.append(Vtotal, V)

    B = np.linalg.lstsq(Utotal, Vtotal)[0]
    for I in range(Hs.shape[0]):
        for J in range(Hs.shape[1]):
            As[I, J] = Ks[I, J] @ B + Cs[I, J]

    return As, B, omegaMax


# used for gradient descent when approximating amplitudes. isn't used now
def rfp_error(omega, Hs, m, n):
    As, B, omegaMax = rfp(omega, Hs, m, n)
    rfpHs = np.sum(
        (1j * omega[:, np.newaxis, np.newaxis, np.newaxis] / omegaMax)
        ** np.arange(m + 1)
        * As,
        axis=-1,
    ) / np.sum(
        (1j * omega[:, np.newaxis, np.newaxis, np.newaxis] / omegaMax)
        ** np.arange(n + 1)
        * B,
        axis=-1,
    )
    return np.sum(np.nan_to_num(abs(np.moveaxis(rfpHs, 0, -1) - Hs) ** 2))


# main function. gives poles of FRF which correspond to modes.
def rfp_poles(omega, Hs, m, n):
    omegaMax = np.max(omega)
    _, B, _ = rfp(omega, Hs, m, n)
    poles = np.polynomial.polynomial.Polynomial(B).roots() * omegaMax
    poles = np.intersect1d(poles, np.conj(poles))
    poles = poles[(np.imag(poles) > 0) & (np.real(poles) <= 0)]
    return poles


# WIP. attempt at extracting amplitudes algebraically
def rfp_shapes(omega, Hs, m, n):
    omegaMax = np.max(omega)
    As, B, _ = rfp(omega, Hs, m, n)
    poles = np.roots(B[::-1])
    poles = poles[np.argsort(np.imag(poles))]
    poles = omegaMax * poles[(np.imag(poles) > 0) & (np.real(poles) <= 0)]
    N = poles.shape[0]
    Ni, No = Hs.shape[:-1]
    R = np.zeros((Ni, No, N))
    for I in range(Ni):
        for O in range(No):
            residues, p, _ = sp.signal.residue(As[I, O, ::-1], B[::-1])
            residues = omegaMax * residues
            p = omegaMax * p
            S = np.argsort(np.imag(p))
            S = S[(np.imag(p[S]) > 0) & (np.real(p[S]) <= 0)]
            p = p[S]
            if np.any(p != poles):
                raise Exception("poles from residues are different to poles from roots")
            residues = residues[S]
            # figure out better way. may be allow complex shapes?
            R[I, O] = -2 * np.imag(p) * np.imag(residues)

    in_shapes = np.zeros((N, Ni))
    out_shapes = np.zeros((N, No))
    for r in range(N):
        U, Sigma, V = sp.linalg.svd(R[:, :, r])
        u = U[:, 0]
        v = V[0]
        sigma = Sigma[0]
        in_shapes[r] = sigma * u
        out_shapes[r] = v

    return in_shapes, out_shapes
