import numpy as np
import cv2
import time


SHOW_STEREOMATCHING_DISPARITY_IMG           = True

STEREOMATCHING_SGBM_MIN_DISPARITY           = 0
    # normally, it's 0 but sometimes rectification algo can shift images, so this param needs to be adjusted accordingly
# 最小视差值。通常设置为0，但有时校正算法可能会移动图像，因此需要相应调整此参数。
STEREOMATCHING_SGBM_N_DISPARITIES			= 512#1024
    # in the current implementation, this param must be divisible by 16
# 视差搜索范围，必须是16的倍数。这个参数决定了算法搜索匹配块的最大视差值。
STEREOMATCHING_SGBM_MATCHING_WIN_SIZE		= 19
    # it must be an odd number >= 1. Normally, it should be somewhere in the 3..11 range
# 匹配窗口大小，必须是大于等于1的奇数。通常在3到11范围内选择一个值。
STEREOMATCHING_SGBM_DISPARITY_SMOOTH_P1		= 0		# Thuy check: max 19								# FIXED PARAM
    # control disparity smoothness. Larger values -> smoother disparity.
    # P1 is the penalty on disparity change by + or - 1 between neighbor pixels
# 这两个参数控制视差图的平滑程度。P1是相邻像素视差改变±1时的惩罚项，P2是相邻像素视差改变超过1时的惩罚项，P2应大于P1。
STEREOMATCHING_SGBM_DISPARITY_SMOOTH_P2		= 0		# Thuy check: max 95								# FIXED PARAM
    # control disparity smoothness. Is the penalty on disparity change by more than 1 between neighbor pixels. P2 > P1
STEREOMATCHING_SGBM_MAX_DISPARITY_CHECK		= 0															# FIXED PARAM
    # max allowed diff (in pixel) in the left-right disparity check. Set it to a non-positive value to disable the check
# 左右视差检查的最大允许差异（以像素为单位）。设置为非正值可以禁用此检查。
STEREOMATCHING_SGBM_PREFILTER_CAP			= 0															# FIXED PARAM
    # truncation value for the prefiltered image pixels. Algo first computes x-derivative at each pixel & clips its
    # value by [-preFilterCap, preFilterCap] interval. Result values are passed to Birchfield-Tomasi pixel cost func
# 预滤波截断值。算法首先计算每个像素的x-导数，并将其值限制在[-preFilterCap, preFilterCap]区间内。结果值传递给像素成本函数。
STEREOMATCHING_SGBM_UNIQUENESS_RATIO		= 15	# good range: 5 -> 15								# FIXED PARAM
    # margin in percent by which the best (min) computed cost func value should 'win' the 2nd best value to consider
    # the found match correct. A value within the 5-15 range is good enough
# 用于确定找到的匹配是否正确的最小成本函数值的百分比“保证金”。在5到15的范围内选择一个值通常就足够好。
STEREOMATCHING_SGBM_SPECKLE_WIN_SIZE		= 200	# 0 to disable. Otherwise, good range: 50 -> 200	# FIXED PARAM
    # max size of smooth disparity regions to consider their noise speckles and invalidate. Set to 0 to disable
    # speckle filtering. Otherwise, set in the 50-200 range
STEREOMATCHING_SGBM_SPECKLE_RANGE   		= 2		# good range: 1 -> 2								# FIXED PARAM
    # max disparity variation within each connected component. If do speckle filtering, set the param to a positive
    # value, it'll be implicitly x 16. Normally, 1 or 2 is good enough
# 这两个参数用于斑点过滤。
STEREOMATCHING_SGBM_MODE					= cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # STEREO_SGBM_MODE_HH to run the full-scale 2-pass dynamic programming algo. It consumes O(W*H*numDisparities)
    #       bytes (large for 640x480 and huge for HD-size). Mutual info cost func is not implemented. Instead, a simpler
    #       Birchfield-Tomasi sub-pixel metric is used. Though, COLOR images are supported as well.
    # STEREO_SGBM_MODE_SGBM: does not support COLOR images
    # STEREO_SGBM_MODE_SGBM_3WAY: 2-3 times faster than MODE_SGBM with minimal degradation in quality and uses universal
    #       HAL intrinsics. Does not support COLOR images
	# Pre- & post- processing steps from K. Konolige algorithm StereoBM::operator() are included, for example:
    #       pre-filtering (CV_STEREO_BM_XSOBEL type) and post-filtering (uniqueness check, quadratic interpolation and
    #       speckle filtering)
# SGBM模式选择。cv2.STEREO_SGBM_MODE_SGBM_3WAY是一种比MODE_SGBM快2-3倍的模式，以最小的质量退化为代价使用通用HAL内在函数，不支持彩色图像。

STEREOMATCHING_PLY_FILTER_RANGE_Z_FROM		= 0
STEREOMATCHING_PLY_FILTER_RANGE_Z_TO		= 30
STEREOMATCHING_PLY_FILTER_RANGE_XY			= 5000												# FIXED PARAM
STEREOMATCHING_PLY_FILTER_BLACK_COLOR_THR	= 30


# ************************************************************************************************
# **********                 FUNCTIONS: DISPLAY                                         **********
# ************************************************************************************************

def ShowDisparityImageSGBM(disparityImg):
    disImg = ResizeImage(disparityImg, 0.25)

    # ref: https://rdmilligan.wordpress.com/2016/05/23/disparity-of-stereo-images-with-python-and-opencv/
    disImg = disImg.astype(np.float32) / 16.0
    disImg = (disImg - STEREOMATCHING_SGBM_MIN_DISPARITY) / STEREOMATCHING_SGBM_N_DISPARITIES
    cv2.imshow('Disparity image SGBM', disImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return;


# ************************************************************************************************
# **********                 FUNCTIONS: COMMON                                          **********
# ************************************************************************************************

def ResizeImage(img, scale):
    imgResized = cv2.resize(src=img, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return imgResized;


def RotateImage(srcImg, angleToRotate):
    height, width = srcImg.shape[:2]
    maxLen = max(height, width)
    rotationMat = cv2.getRotationMatrix2D(center=(maxLen/2,maxLen/2), angle=angleToRotate, scale=1.0)
    desImg = cv2.warpAffine(src=srcImg, M=rotationMat, dsize=(maxLen, maxLen))
    return desImg;


def RotateLeftAndRightImagesAccordingToOurSystem(leftImg, rightImg):
    height, width = leftImg.shape[:2]
    leftImg90 = RotateImage(leftImg, 90)     # 1. rotate left img 90-deg counter-clockwise (left direction)
    rightImg270 = RotateImage(rightImg, 270) # 2. rotate right img 90-deg clockwise (right direction)
                                             # cuz img is horizontal, but EXIF changes according to camera gravity
    leftImgRotated = leftImg90[0:width, 0:height]              # 3. crop left img
    rightImgRotated = rightImg270[0:width, width-height:width] # 4. crop right img
                                    # ref (extract ROI): http://docs.opencv.org/3.2.0/d3/df2/tutorial_py_basic_ops.html
    return leftImgRotated, rightImgRotated;


def ConvertLeftAndRightImagesToGray(leftImg, rightImg):
    leftImgGray = cv2.cvtColor(src=leftImg, code=cv2.COLOR_RGB2GRAY)
    rightImgGray = cv2.cvtColor(src=rightImg, code=cv2.COLOR_RGB2GRAY)
    return leftImgGray, rightImgGray;


# ************************************************************************************************
# **********                 FUNCTIONS: 3D RECONSTRUCTION                               **********
# ************************************************************************************************

def LoadCalibResult(calibFilename):
    # cv2.FileStorage: 这是OpenCV中用于读写文件的类。它可以处理包括XML和YAML在内的文件格式，通常用于读取或保存相机校准数据、配置参数等。
    fs = cv2.FileStorage(filename=calibFilename, flags=cv2.FILE_STORAGE_READ)
    intrinsic  = fs.getNode('intrinsic').mat() # ref: https://github.com/opencv/opencv_contrib/issues/834
    distortion = fs.getNode('distortion').mat()
    rotation   = fs.getNode('rotation').mat()
    projection = fs.getNode('projection').mat()
    Q          = fs.getNode('Q').mat()
    return intrinsic, distortion, rotation, projection, Q;


def RectifyLeftAndRightImagesUsingCalibMatrices(leftImg, rightImg, \
                                                leftIntrinsic, leftDistortion, leftRotation, leftProjection, \
                                                rightIntrinsic, rightDistortion, rightRotation, rightProjection):
    height, width = leftImg.shape[:2]

    # precompute maps for cvRemap()
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(cameraMatrix=leftIntrinsic, distCoeffs=leftDistortion, \
                            R=leftRotation, newCameraMatrix=leftProjection, size=(width,height), m1type=cv2.CV_16SC2)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(cameraMatrix=rightIntrinsic, distCoeffs=rightDistortion, \
                            R=rightRotation, newCameraMatrix=rightProjection, size=(width, height), m1type=cv2.CV_16SC2)
                    # check for detail: http://docs.opencv.org/2.4.8/modules/imgproc/doc/geometric_transformations.html
                    #                   http://docs.opencv.org/3.2.0/dc/dbb/tutorial_py_calibration.html
    # rectify 2 images using maps
    leftImgRectified = cv2.remap(src=leftImg, map1=leftMapX, map2=leftMapY, interpolation=cv2.INTER_LINEAR)
    rightImgRectified = cv2.remap(src=rightImg, map1=rightMapX, map2=rightMapY, interpolation=cv2.INTER_LINEAR)
    return leftImgRectified, rightImgRectified;


def StereoMatching(leftImg, rightImg):
    stereo = cv2.StereoSGBM_create(minDisparity     = STEREOMATCHING_SGBM_MIN_DISPARITY, \
                                   numDisparities   = STEREOMATCHING_SGBM_N_DISPARITIES, \
                                   blockSize        = STEREOMATCHING_SGBM_MATCHING_WIN_SIZE, \
                                   P1               = STEREOMATCHING_SGBM_DISPARITY_SMOOTH_P1, \
                                   P2               = STEREOMATCHING_SGBM_DISPARITY_SMOOTH_P2, \
                                   disp12MaxDiff    = STEREOMATCHING_SGBM_MAX_DISPARITY_CHECK, \
                                   preFilterCap     = STEREOMATCHING_SGBM_PREFILTER_CAP, \
                                   uniquenessRatio  = STEREOMATCHING_SGBM_UNIQUENESS_RATIO, \
                                   speckleWindowSize= STEREOMATCHING_SGBM_SPECKLE_WIN_SIZE, \
                                   speckleRange     = STEREOMATCHING_SGBM_SPECKLE_RANGE, \
                                   mode             = STEREOMATCHING_SGBM_MODE)
    disparityImg = stereo.compute(leftImg, rightImg)#.astype(np.float32) / 16.0  # is 16-bit signed single-channel
    return disparityImg;


def StereoMatchingWithCalibFiles(leftImg, rightImg, leftCalibFilename, rightCalibFilename):
    # load calibration files
    leftIntrinsic, leftDistortion, leftRotation, leftProjection, leftQ = LoadCalibResult(leftCalibFilename)
    rightIntrinsic, rightDistortion, rightRotation, rightProjection, rightQ = LoadCalibResult(rightCalibFilename)

    # rotate, rectify, convert-to-gray the left and right images
    # leftImgRotated, rightImgRotated = RotateLeftAndRightImagesAccordingToOurSystem(leftImg, rightImg)
    leftImgRectified, rightImgRectified = RectifyLeftAndRightImagesUsingCalibMatrices(leftImg, rightImg, \
                                                      leftIntrinsic, leftDistortion, leftRotation, leftProjection, \
                                                      rightIntrinsic, rightDistortion, rightRotation, rightProjection)
    leftImgRectifiedGray, rightImgRectifiedGray = ConvertLeftAndRightImagesToGray(leftImgRectified, rightImgRectified)

    # stereo matching
    disparityImg = StereoMatching(leftImgRectifiedGray, rightImgRectifiedGray)
    cv2.imwrite("leftImgRectified.bmp", leftImgRectified)
    cv2.imwrite("rightImgRectified.bmp", rightImgRectified)
    return disparityImg, leftImgRectified, leftQ;


def CheckPlyFileExportCondition(disparityX, disparityY, disparityZ):
    # position conditions
    condPosX = (-STEREOMATCHING_PLY_FILTER_RANGE_XY < disparityX and disparityX < STEREOMATCHING_PLY_FILTER_RANGE_XY)
    condPosY = (-STEREOMATCHING_PLY_FILTER_RANGE_XY < disparityY and disparityY < STEREOMATCHING_PLY_FILTER_RANGE_XY)
    condPosZ = (STEREOMATCHING_PLY_FILTER_RANGE_Z_FROM<disparityZ and disparityZ<STEREOMATCHING_PLY_FILTER_RANGE_Z_TO)

    return (condPosX and condPosY and condPosZ );


def SaveWorldImageToPLY(worldImg, plyFilename):
    worldImgHeight, worldImgWidth = worldImg.shape[:2]

    with open(plyFilename, 'w') as f: # write to a new file (NOT APPEND)
        f.write('ply\n')
        f.write('format ascii 1.0\n')

        # count possible display points
        count = 0
        for i in range(0, worldImgHeight):
            for j in range(0, worldImgWidth):
                disparityX = worldImg.item(i, j, 0) # to access all B,G,R values --> call .item() separately for all
                disparityY = worldImg.item(i, j, 1) # http://docs.opencv.org/3.2.0/d3/df2/tutorial_py_basic_ops.html
                disparityZ = worldImg.item(i, j, 2)

                if CheckPlyFileExportCondition(disparityX, disparityY, disparityZ):
                    count = count + 1
        f.write('element vertex ' + str(count) + '\n')

        # other headers
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')

        # write points
        for i in range(0, worldImgHeight):
            for j in range(0, worldImgWidth):
                disparityX = worldImg.item(i, j, 0)  # to access all B,G,R values --> call .item() separately for all
                disparityY = worldImg.item(i, j, 1)  # http://docs.opencv.org/3.2.0/d3/df2/tutorial_py_basic_ops.html
                disparityZ = worldImg.item(i, j, 2)

                # CheckPlyFileExportCondition函数通过结合位置和颜色的条件，为导出到PLY文件的点提供了一个基本的筛选机制。这对于3D重建和视觉呈现等应用来说是很有用的，因为它可以提高生成的3D模型的质量和可用性。
                if CheckPlyFileExportCondition(disparityX, disparityY, disparityZ):
                    f.write(format(disparityX,'.4f') + ' ' + format(disparityY,'.4f') + ' ' + format(disparityZ,'.4f')  + '\n')
    return;


def SaveDisparityImageToPLY(disparityImg, perspectiveMatrix, plyFilename):
    worldImg = cv2.reprojectImageTo3D(disparity=disparityImg, Q=perspectiveMatrix, handleMissingValues=True)
    SaveWorldImageToPLY(worldImg, plyFilename)
    return;




# ************************************************************************************************
# **********                                    MAIN                                    **********
# ************************************************************************************************

# leftImg = cv2.imread('testdata01_withCalibration/0002_A01.jpeg')
# rightImg = cv2.imread('testdata01_withCalibration/0002_A02.jpeg')
# leftImg = cv2.imread('img2.bmp')
# rightImg = cv2.imread('img1.bmp')
leftImg = cv2.imread('testHkvs/testleft.bmp')
rightImg = cv2.imread('testHkvs/testright.bmp')

start = time.time()
disparityImg, leftImgRectified, perspectiveMatrix = StereoMatchingWithCalibFiles(leftImg, rightImg, 'testHkvs/hkvs_l.txt', 'testHkvs/hkvs_r.txt')
print(time.time() - start)

if SHOW_STEREOMATCHING_DISPARITY_IMG:
    ShowDisparityImageSGBM(disparityImg)

start = time.time()
SaveDisparityImageToPLY(disparityImg,  perspectiveMatrix, 'test2.ply')
print("test2.ply 已生成")
print(time.time() - start)

