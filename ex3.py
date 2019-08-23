import cv2
import numpy as np
import matplotlib.pyplot as plt


def padding(img):
    cv2.imshow('before', img)
    cv2.waitKey(0)
    new_img = np.pad(img, ((18, 18), (18, 18)), 'constant')
    cv2.imshow('after', new_img)
    cv2.waitKey(0)
    return new_img


def derivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray):
    ker1 = np.array([[-1, 0, 1]])
    ker2 = np.array([[-1], [0], [1]])
    Ix = cv2.filter2D(inImage, -1, ker1)
    Iy = cv2.filter2D(inImage, -1, ker2)
    Ix_n = cv2.normalize(Ix.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    Iy_n = cv2.normalize(Iy.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    return Ix_n, Iy_n


def newIm1(im1: np.ndarray, uv: np.ndarray) -> np.ndarray:
    new_im1 = im1.copy()
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            new_im1[i][j] = im1[i+int(uv[0][0])][j+int(uv[1][0])]
    return new_im1


# def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
#     Ix, Iy = derivative(im2.copy())
#     print("Ix=", Ix)
#     print("Iy=", Iy)
#     It = im2 - im1
#     uv = np.array([[0.0],
#                    [0.0]])
#     IxIx, IxIy, IyIx, IyIy, IxIt, IyIt = (0, 0, 0, 0, 0, 0)
#     for ind in range(0, 1000):
#         It = im2 - newIm1(im1, uv)
#         print(It)
#         for i in range(0, im2.shape[0]):
#             for j in range(0, im2.shape[1]):
#                 IxIx += Ix[i][j]*Ix[i][j]
#                 IxIy += Ix[i][j]*Iy[i][j]
#                 IyIx += Iy[i][j]*Ix[i][j]
#                 IyIy += Iy[i][j]*Iy[i][j]
#                 IxIt += Ix[i][j]*It[i][j]
#                 IyIt += Iy[i][j]*It[i][j]
#         ATA = np.array([[IxIx, IxIy],
#                         [IyIx, IyIy]])
#         print(ATA)
#         ATb = (-1)*np.array([[IxIt],
#                              [IyIt]])
#         res = np.dot(np.linalg.pinv(ATA), ATb)
#         uv = res
#     print(uv)
#     return None


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    return None


def findTranslationLKPython():
    im1 = cv2.imread('1.png')
    im2 = cv2.imread('2.png')
    old_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    old_points = np.array([[17.4597, 233.321]], dtype=np.float32)
    now_points = np.array([[258.744, 233.321]], dtype=np.float32)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    while True:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, now_points, **lk_params)
        # old_gray = gray_frame.copy()
        old_points = new_points
        print(new_points)


#  Q2.1 - find Gaussian Pyramid
def GaussianPyramid(im: np.ndarray, maxLevels: int, filterSize: int):
    kernel = getgaussiankernel(filterSize)
    ans = [None]*maxLevels
    new_im = im.copy()
    for i in range(maxLevels):
        new_im = cv2.normalize(new_im.astype('double'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)  # Convert to normalized floating point
        ans[i] = new_im
        cv2.imshow('IMAGE', new_im)
        cv2.waitKey(0)
        new_im = cv2.filter2D(new_im, -1, kernel)
        new_im = cv2.filter2D(new_im, -1, kernel.T)
        # new_im = np.convolve(new_im, kernel)
        # new_im = np.convolve(new_im, kernel.T)
        new_im = reduce(new_im)
    return ans, kernel


def getgaussiankernel(filterSize: int) -> np.ndarray:
    base = np.array([1, 1])
    kernel = np.array([1, 1])
    for i in range(1, filterSize-1):
        kernel = np.convolve(kernel, base)
    n_kernel = cv2.normalize(kernel.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    n_kernel = n_kernel.T
    return n_kernel


def reduce(im: np.ndarray):
    new_im = np.zeros((im.shape[0]//2, im.shape[1]//2))
    for i in range(im.shape[0]-1):
        for j in range(im.shape[1]-1):
            if i%2 == 0 and j%2 == 0:
                new_im[i//2][j//2] = im[i][j]
    return new_im


#   Q2.1 - find Laplacian Pyramid
def LaplacianPyramid(im: np.ndarray, maxLevels: int, filterSize: int):
    new_im = im.copy()

    # Example to 1 level pyramid
    # LG0, kernel = GaussianPyramid(new_im, 2, 5)
    # G1 = LG0[1]
    # G0 = LG0[0]
    # L0 = G0 - expand(G1, kernel)
    # real = expand(G1, kernel) + L0
    # cv2.imshow('Example: expand+laplacian', real)
    # cv2.waitKey(0)

    # Implementation
    ans = [None]*maxLevels
    Gs, kernel = GaussianPyramid(new_im, maxLevels+1, filterSize)
    for i in range(1, maxLevels+1):  # Li = Gi - expand(Gi+1)
        Gi = Gs[i-1]
        cv2.imshow('Gi', Gi)
        cv2.waitKey(0)
        Gi_1 = Gs[i]
        cv2.imshow('Gi+1', expand(Gi_1, kernel))
        cv2.waitKey(0)
        Li = Gi - expand(Gi_1, 0)
        ans[i-1] = Li
        cv2.imshow('implement laplacian', Li)
        cv2.waitKey(0)


def expand(im: np.ndarray, kernel: np.ndarray):
    ex_im = np.zeros((im.shape[0]*2, im.shape[1]*2))
    ans = ex_im.copy()

    # Init the matrix with zero padding and values
    for i in range(ex_im.shape[0]):
        for j in range(ex_im.shape[1]):
            if j%2 == 1:
                ex_im[i][j] = im[i//2][j//2]

    # Cross Coralletion
    for i in range(ex_im.shape[0]):
        for j in range(1, ex_im.shape[1]-1):
            ans[i][j] = 0.5*ex_im[i][j-1] + 1.0*ex_im[i][j] + 0.5*ex_im[i][j+1]
    return ans




# a1 = np.array([[1.0, 1.0]])
# a2 = np.array([[0.0, 1.0]])
a1 = np.array([[1.0, 1.0, 1.0],
               [1.0, 2.0, 1.0],
               [3.0, 1.0, 3.0]])
a2 = np.array([[0.0, 1.0, 1.0],
               [0.0, 1.0, 2.0],
               [0.0, 3.0, 1.0]])
# findTranslationLK(a1, a2)
# findTranslationLKPython()
# t1 = np.array([1, 1])
# t2 = np.convolve(t1, t1)
# t3 = np.convolve(t2, t1)
# t4 = np.convolve(t3, t1)
# print(t4)
img = cv2.imread('lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = padding(img)
# GaussianPyramid(img, 5, 5)
LaplacianPyramid(img, 5, 5)
