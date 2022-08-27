# Extended vision supports
# This file is mangled due to the inclusion of assignment codes
import warnings
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

pUSiVyMn = np.std
pUSiVyMb = np.mean
pUSiVyMX = np.arange
pUSiVyMu = np.zeros_like
pUSiVyMc = np.sum
pUSiVyMG = np.dot
pUSiVyMj = np.floor
pUSiVycf = np.exp
pUSiVycJ = np.log
pUSiVycO = np.abs
pUSiVycN = np.finfo
pUSiVycx = np.log10
pUSiVycB = np.linspace
pUSiVycP = np.imag
pUSiVycA = np.real
pUSiVycQ = np.fft
pUSiVyck = np.pi
pUSiVych = np.cos
pUSiVycY = np.pad
pUSiVycK = np.concatenate
pUSiVycL = np.ndarray
pUSiVycd = np.transpose
pUSiVycs = np.matmul
pUSiVyca = np.linalg
pUSiVyct = np.zeros
pUSiVycl = np.array
pUSiVyMI = range
pUSiVyMR = UserWarning
pUSiVyMr = list
pUSiVyME = len
pUSiVyMm = Exception
pUSiVyMC = None
pUSiVyMq = False
pUSiVyMe = print
pUSiVyMT = str
pUSiVyMF = int
pUSiVyMD = True
pUSiVyMg = float
pUSiVyMw = isinstance
pUSiVyMz = tuple
pUSiVycW = warnings.warn
pUSiVyjG = pUSiVyMI
pUSiVyjc = pUSiVyMR
pUSiVyjM = pUSiVyMr
pUSiVyju = pUSiVyME
pUSiVyjX = pUSiVyMm
pUSiVyjb = pUSiVycW
pUSiVyMn = np.std
pUSiVyMb = np.mean
pUSiVyMX = np.arange
pUSiVyMu = np.zeros_like
pUSiVyMc = np.sum
pUSiVyMG = np.dot
pUSiVyMj = np.floor
pUSiVycf = np.exp
pUSiVycJ = np.log
pUSiVycO = np.abs
pUSiVycN = np.finfo
pUSiVycx = np.log10
pUSiVycB = np.linspace
pUSiVycP = np.imag
pUSiVycA = np.real
pUSiVycQ = np.fft
pUSiVyck = np.pi
pUSiVych = np.cos
pUSiVycY = np.pad
pUSiVycK = np.concatenate
pUSiVycL = np.ndarray
pUSiVycd = np.transpose
pUSiVycs = np.matmul
pUSiVyca = np.linalg
pUSiVyct = np.zeros
pUSiVycl = np.array
pUSiVyjn = pUSiVycl
pUSiVyjo = pUSiVyct
pUSiVyjI = pUSiVyca
pUSiVyjR = pUSiVycs
pUSiVyjr = pUSiVycd
pUSiVyjE = pUSiVycL
pUSiVyMo = wavfile.read
pUSiVyjs = 0.97
pUSiVyjd = 25
pUSiVyjL = 10
pUSiVyjK = 0.5
pUSiVyjY = 26
pUSiVyjh = 20
pUSiVyjk = 6000
pUSiVyjQ = 13
pUSiVyjA = 3
pUSiVyjP = 3


def pUSiVyGf(OO0O0OOO000OO0O0O: pUSiVyjE):
    pUSiVyjm = pUSiVyjr(OO0O0OOO000OO0O0O, (1, 0))
    pUSiVyjC = pUSiVyjR(pUSiVyjm, OO0O0OOO000OO0O0O)
    pUSiVyjq, pUSiVyje = pUSiVyjI.eig(pUSiVyjC)
    pUSiVyjT = 0
    for pUSiVyjF in pUSiVyjG(pUSiVyjq.shape[0]):
        if pUSiVyjq[pUSiVyjF] < pUSiVyjq[pUSiVyjT]:
            pUSiVyjT = pUSiVyjF
    if pUSiVyjq[pUSiVyjT] >= 1e-6:
        pUSiVyjb("Error", pUSiVyjc)
    return pUSiVyjq[pUSiVyjT], pUSiVyje[:, pUSiVyjT]


def pUSiVycj(O0OOO0OO0O00OOOOO):
    pUSiVyjD = pUSiVyjo((3, 1))
    pUSiVyjD[0, 0] = O0OOO0OO0O00OOOOO[0]
    pUSiVyjD[1, 0] = O0OOO0OO0O00OOOOO[1]
    pUSiVyjD[2, 0] = 1
    return pUSiVyjD


def pUSiVycG(O0O0000000O0000O0):
    return [O0O0000000O0000O0[0, 0] / O0O0000000O0000O0[2, 0], O0O0000000O0000O0[1, 0] / O0O0000000O0000O0[2, 0]]


def pUSiVycM(OOOO00000O0O0O0OO, OOO0000O0O0OOOO00):
    return (OOOO00000O0O0O0OO[0] - OOO0000O0O0OOOO00[0]) ** 2 + (OOOO00000O0O0O0OO[1] - OOO0000O0O0OOOO00[1]) ** 2


def pUSiVycX(subplot_id):
    plt.subplots_adjust(left=pUSiVyMC, bottom=pUSiVyMC, right=pUSiVyMC, top=pUSiVyMC, wspace=0.5, hspace=0.5)
    plt.subplot(pUSiVyjP, pUSiVyjA, subplot_id + 1)


def pUSiVycb(filename, normalize=pUSiVyMq):
    pUSiVyjx, pUSiVyjN = pUSiVyMo(filename)
    pUSiVyMe(pUSiVyjx, pUSiVyME(pUSiVyjN))
    if normalize:
        pUSiVyjN = pUSiVyjN / 32767
    if pUSiVycl(pUSiVyjN.shape).shape[0] == 2:
        pUSiVyMe("")
        pUSiVyjO = pUSiVyct((pUSiVyjN.shape[0]))
        pUSiVyjJ = pUSiVyjN.shape[0]
        pUSiVyjf = pUSiVyjN.shape[1]
        for i in pUSiVyMI(pUSiVyjJ):
            if i % 100000 == 0:
                pUSiVyMe("" + pUSiVyMT(i / pUSiVyjJ * 100) + "%")
            for k in pUSiVyMI(pUSiVyjf):
                pUSiVyjO[i] += pUSiVyjN[i, k] / pUSiVyjf
        pUSiVyjN = pUSiVyjO
    pUSiVyMe("done")
    return pUSiVyjx, pUSiVyjN


def pUSiVycn(pUSiVyjN, preemphasis_factor=pUSiVyjs):
    pUSiVyGj = pUSiVycK((pUSiVyjN[1:pUSiVyME(pUSiVyjN)], [0]), 0)
    pUSiVyGc = pUSiVyGj - pUSiVyjN * preemphasis_factor
    pUSiVyMe("")
    return pUSiVyGc, pUSiVyjN


def pUSiVyco(pUSiVyjN, pUSiVyjx, windowing_factor=pUSiVyjK, frame_shift_ms=pUSiVyjL, frame_size_ms=pUSiVyjd):
    pUSiVyGM = pUSiVyME(pUSiVyjN)
    pUSiVyGu = pUSiVyMF(frame_size_ms * pUSiVyjx / 1000)
    pUSiVyGX = pUSiVyMF(frame_shift_ms * pUSiVyjx / 1000)
    pUSiVyGb = (pUSiVyGu - pUSiVyGM) % pUSiVyGX
    pUSiVyGn = pUSiVycY(pUSiVyjN, (0, pUSiVyGb), mode="constant", constant_values=(0, 0))
    pUSiVyGo = (pUSiVyME(pUSiVyGn) - pUSiVyGu) // pUSiVyGX + 1
    pUSiVyGI = pUSiVyct((pUSiVyGo, pUSiVyGu))
    pUSiVyMe("")
    pUSiVyMe(pUSiVyME(pUSiVyGn))
    for i in pUSiVyMI(pUSiVyGo):
        for j in pUSiVyMI(pUSiVyGX * i, pUSiVyGX * i + pUSiVyGu):
            t = j - pUSiVyGX * i
            pUSiVyGI[i, t] = pUSiVyGn[j] * (
                    (1 - windowing_factor) - windowing_factor * pUSiVych(2 * pUSiVyck * t / (pUSiVyGu - 1)))
    pUSiVyGR = pUSiVyGI.copy()
    pUSiVyMe("")
    return pUSiVyGR, pUSiVyGo, pUSiVyGu


def pUSiVycu(OOO00OOO00O0OOO0O: pUSiVyjM[pUSiVyjE], O0000OO000O0O0O0O: pUSiVyjM[pUSiVyjE]):
    pUSiVyjg, pUSiVyjw = pUSiVyju(OOO00OOO00O0OOO0O), pUSiVyju(O0000OO000O0O0O0O)
    if pUSiVyjg != pUSiVyjw:
        raise pUSiVyjX("Error")
    if pUSiVyjg < 2 ** 2:
        raise pUSiVyjX("Error")
    pUSiVyjz = pUSiVyjg
    pUSiVyjv = pUSiVyjo((pUSiVyjz * 2, 9))
    for pUSiVyjH in pUSiVyjG(pUSiVyjz):
        pUSiVyjv[2 * pUSiVyjH, :] = pUSiVyjn(
            [-OOO00OOO00O0OOO0O[pUSiVyjH][0], -OOO00OOO00O0OOO0O[pUSiVyjH][1], -1, 0, 0, 0,
             O0000OO000O0O0O0O[pUSiVyjH][0] * OOO00OOO00O0OOO0O[pUSiVyjH][0],
             O0000OO000O0O0O0O[pUSiVyjH][0] * OOO00OOO00O0OOO0O[pUSiVyjH][1], O0000OO000O0O0O0O[pUSiVyjH][0]])
        pUSiVyjv[2 * pUSiVyjH + 1, :] = pUSiVyjn(
            [0, 0, 0, -OOO00OOO00O0OOO0O[pUSiVyjH][0], -OOO00OOO00O0OOO0O[pUSiVyjH][1], -1,
             O0000OO000O0O0O0O[pUSiVyjH][1] * OOO00OOO00O0OOO0O[pUSiVyjH][0],
             O0000OO000O0O0O0O[pUSiVyjH][1] * OOO00OOO00O0OOO0O[pUSiVyjH][1], O0000OO000O0O0O0O[pUSiVyjH][1]])
    pUSiVyjW, pUSiVyjl = pUSiVyGf(pUSiVyjv)
    pUSiVyjt = pUSiVyjo((3, 3))
    for pUSiVyjH in pUSiVyjG(3):
        for pUSiVyja in pUSiVyjG(3):
            pUSiVyjt[pUSiVyjH, pUSiVyja] = pUSiVyjl[pUSiVyjH * 3 + pUSiVyja]
    pUSiVyjt = pUSiVyjt / pUSiVyjt[2, 2]
    return pUSiVyjt


def pUSiVycI(pUSiVyGI, pUSiVyGo):
    pUSiVyGR = []
    pUSiVyGr = 1
    while pUSiVyMD:
        if pUSiVyGr < pUSiVyGI.shape[1]:
            pUSiVyGr = pUSiVyGr * 2
        else:
            break
    pUSiVyGr = pUSiVyGI.shape[1]
    for i in pUSiVyMI(pUSiVyGo):
        pUSiVyGE = pUSiVycQ.rfft(pUSiVyGI[i], pUSiVyGr)
        pUSiVyGm = pUSiVycA(pUSiVyGE)
        pUSiVyGC = pUSiVycP(pUSiVyGE)
        pUSiVyGq = (pUSiVyGC * pUSiVyGC + pUSiVyGm * pUSiVyGm)
        pUSiVyGR.append(pUSiVyGq)
    pUSiVyMe("DFT: done")
    return pUSiVycl(pUSiVyGR), pUSiVyGr


def pUSiVycR(pUSiVyGR, pUSiVyGu, sampling_freq, subplot_id=2):
    pUSiVyGe = pUSiVyGR[:, :pUSiVyGR.shape[1]]
    yt = pUSiVycB(0, pUSiVyGu, 10)
    xt = pUSiVycB(0, pUSiVyGR.shape[0], 10)
    s = sampling_freq // 2 / pUSiVyGu * yt
    r = pUSiVyjL * pUSiVycB(0, pUSiVyGR.shape[0], 10)
    r = r.astype("int")
    s = s.astype("int")
    pUSiVyGT = pUSiVycx(pUSiVyGe.T + pUSiVycN("float").eps) * 10
    pUSiVycX(subplot_id)
    plt.pcolormesh(pUSiVyGT)
    plt.yticks(yt, s)
    plt.ylabel("")
    plt.xticks(xt, r)
    plt.xlabel("")
    pUSiVyGF = plt.colorbar()
    pUSiVyGF.ax.set_ylabel("")
    plt.title("")


def pUSiVycr(audio_sequence, pUSiVyjx, title=""):
    pUSiVyGE = pUSiVycO(pUSiVycQ.rfft(audio_sequence, 512)) ** 2
    pUSiVyGD = 512
    st = [pUSiVyjx // 2 / pUSiVyGD * i for i in pUSiVyMI(pUSiVyGE.shape[0])]
    pUSiVycX(0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Accumulated Energy (dB)")
    plt.title(title)
    pUSiVyGE = pUSiVycx(pUSiVyGE + pUSiVycN("float").eps) * 10
    plt.plot(st, pUSiVyGE)


def pUSiVycE(normal_freq):
    return 1125 * pUSiVycJ(1 + normal_freq / 700)


def pUSiVycm(m_freq):
    return 700 * (pUSiVycf(m_freq / 1125) - 1)


def pUSiVycC(sampling_freq, fft_size, filters=pUSiVyjY, low_freq=pUSiVyjh, high_freq=pUSiVyjk):
    pUSiVyGg = pUSiVycl([pUSiVycE(low_freq) + i * (pUSiVycE(high_freq) - pUSiVycE(low_freq)) / (filters + 1) for i in
                         pUSiVyMI(filters + 2)])
    pUSiVyGw = pUSiVycm(pUSiVyGg)
    pUSiVyGz = pUSiVyMj((fft_size / sampling_freq) * pUSiVyGw)
    pUSiVyGv = pUSiVyct((filters, fft_size // 2 + 1))
    for i in pUSiVyMI(filters):
        for j in pUSiVyMI(fft_size):
            if pUSiVyGz[i + 1] >= j > pUSiVyGz[i]:
                pUSiVyGv[i, j] = (j - pUSiVyGz[i]) / (pUSiVyGz[i + 1] - pUSiVyGz[i])
            elif pUSiVyGz[i + 1] < j < pUSiVyGz[i + 2]:
                pUSiVyGv[i, j] = (pUSiVyGz[i + 2] - j) / (pUSiVyGz[i + 2] - pUSiVyGz[i + 1])
    pUSiVyMe("")
    return pUSiVyGv


def pUSiVycq(stft_matrix, mel_filter_banks, log_energy=pUSiVyMD):
    pUSiVyGH = pUSiVyMG(stft_matrix, mel_filter_banks.T)
    pUSiVyGW = pUSiVyMc(stft_matrix, 1)
    for i in pUSiVyMI(pUSiVyME(pUSiVyGW)):
        if pUSiVyGW[i] == 0:
            pUSiVyGW[i] = pUSiVycN(pUSiVyMg).eps
    if log_energy:
        pUSiVyGW = pUSiVycJ(pUSiVyGW)
    pUSiVyMe("")
    return pUSiVyGH, pUSiVyGW


def pUSiVyce(mel_filter_bank):
    pUSiVyGr = mel_filter_bank.shape[1]
    pUSiVyGv = mel_filter_bank.shape[0]
    pUSiVyGl = pUSiVycl([i for i in pUSiVyMI(pUSiVyGr)])
    pUSiVycX(3)
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    for i in pUSiVyMI(pUSiVyGv):
        plt.plot(pUSiVyGl, mel_filter_bank[i, :])


def pUSiVycT(pUSiVyGH):
    pUSiVyGt = pUSiVyGH[:, :pUSiVyGH.shape[1]]
    xt = pUSiVycB(0, pUSiVyGH.shape[0], 10)
    r = pUSiVyjL * pUSiVycB(0, pUSiVyGH.shape[0], 10)
    r = r.astype("int")
    pUSiVyGT = pUSiVyGt.T
    pUSiVycX(4)
    plt.pcolormesh(pUSiVyGT)
    plt.ylabel("")
    plt.xticks(xt, r)
    plt.xlabel("")
    plt.colorbar()
    plt.title("")


def pUSiVycF(pUSiVyGH, mfcc_nums=pUSiVyjQ):
    pUSiVyGa = pUSiVyct((pUSiVyGH.shape[0], mfcc_nums))
    for i in pUSiVyMI(pUSiVyGH.shape[0]):
        for j in pUSiVyMI(mfcc_nums):
            for k in pUSiVyMI(pUSiVyGH.shape[1]):
                pUSiVyGa[i, j] += pUSiVyGH[i, k] * pUSiVych((k + 0.5) * j * pUSiVyck / pUSiVyGH.shape[1])
    pUSiVyMe("")
    pUSiVyMe(pUSiVyGa.shape)
    return pUSiVyGa


def pUSiVycD(pUSiVyGH, subplot_id=5, title=""):
    pUSiVyGt = pUSiVyGH[:, :pUSiVyGH.shape[1]]
    xt = pUSiVycB(0, pUSiVyGH.shape[0], 5)
    r = pUSiVyjL * pUSiVycB(0, pUSiVyGH.shape[0], 5)
    r = r.astype("int")
    pUSiVyGT = pUSiVyGt.T
    pUSiVycX(subplot_id)
    plt.pcolormesh(pUSiVyGT)
    plt.ylabel("")
    plt.xticks(xt, r)
    plt.xlabel("")
    plt.colorbar()
    plt.title(title)


def pUSiVycg(pUSiVyGH, subplot_id=5, title="Feature Map"):
    pUSiVyGt = pUSiVyGH[:, :pUSiVyGH.shape[1]]
    xt = pUSiVycB(0, pUSiVyGH.shape[0], 5)
    r = pUSiVyjL * pUSiVycB(0, pUSiVyGH.shape[0], 5)
    r = r.astype("int")
    pUSiVyGT = pUSiVyGt.T
    pUSiVycX(subplot_id)
    plt.pcolormesh(pUSiVyGT)
    plt.ylabel("")
    plt.xticks(xt, r)
    plt.xlabel("")
    plt.colorbar()
    plt.title(title)


def pUSiVycw(audio_sequence, title, subplot_id):
    pUSiVycX(subplot_id)
    pUSiVyGl = pUSiVycl([i for i in pUSiVyMI(pUSiVyME(audio_sequence))])
    plt.plot(pUSiVyGl, audio_sequence)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title)


def pUSiVycz(pUSiVyGH, n):
    pUSiVyGs = 2 * pUSiVyMc([i * i for i in pUSiVyMI(1, n + 1)])
    pUSiVyGd = pUSiVycY(pUSiVyGH, ((n, n), (0, 0)), mode="edge")
    pUSiVyGL = pUSiVyMu(pUSiVyGH)
    pUSiVyGK = pUSiVyMX(-n, n + 1)
    for i in pUSiVyMI(pUSiVyGH.shape[0]):
        pUSiVyGL[i] = pUSiVyMG(pUSiVyGK, pUSiVyGd[i:i + 2 * n + 1]) / pUSiVyGs
    pUSiVyMe("")
    return pUSiVyGL


def pUSiVycv(pUSiVyGW, n):
    pUSiVyGs = 2 * pUSiVyMc([i * i for i in pUSiVyMI(1, n + 1)])
    pUSiVyGY = pUSiVycY(pUSiVyGW, (n, n))
    pUSiVyGh = pUSiVyMu(pUSiVyGW)
    pUSiVyGK = pUSiVyMX(-n, n + 1)
    for i in pUSiVyMI(pUSiVyGW.shape[0]):
        pUSiVyGh[i] = pUSiVyMG(pUSiVyGK, pUSiVyGY[i:i + 2 * n + 1]) / pUSiVyGs
    pUSiVyMe("")
    return pUSiVyGh


def pUSiVycH(feature):
    if pUSiVyMw(feature, pUSiVyMz) or pUSiVyMw(feature, pUSiVyMr):
        pUSiVyGk = pUSiVycl(feature)
    else:
        pUSiVyGk = feature
    pUSiVyGQ = pUSiVyGk.shape
    pUSiVyGA = pUSiVyGQ[0]
    pUSiVyGP = []
    for i in pUSiVyMI(pUSiVyGA):
        pUSiVyGP.append(pUSiVyGk[i])
    pUSiVyGB = pUSiVyMz(pUSiVyGP)
    pUSiVyGx = pUSiVycK(pUSiVyGB, axis=1)
    pUSiVyGN = pUSiVyGx.shape[1]
    for i in pUSiVyMI(pUSiVyGN):
        pUSiVyGO = pUSiVyMb(pUSiVyGx[:, i])
        pUSiVyGJ = pUSiVyMn(pUSiVyGx[:, i], ddof=1)
        pUSiVyGx[:, i] = (pUSiVyGx[:, i] - pUSiVyGO) / pUSiVyGJ
    return pUSiVyGx
