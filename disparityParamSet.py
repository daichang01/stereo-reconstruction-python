import numpy as np
import cv2

def compute_disparity(img_left, img_right, num_disparities, block_size):
    """
    使用StereoSGBM算法计算视差图。
    """
    # 确保block_size是奇数
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 5:
        block_size = 5

    # 初始化StereoSGBM对象
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16 * num_disparities, blockSize=block_size)
    # 计算视差
    disparity = stereo.compute(img_left, img_right)
    return disparity

def adjust_disparity_parameters(window_name, img_left, img_right):
    """
    创建滑动条用于调整视差参数，并显示视差图。
    """
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.createTrackbar('num', window_name, 2, 10, lambda x: None)
    cv2.createTrackbar('blockSize', window_name, 5, 25, lambda x: None)

    app = 0
    while True:
        # 读取滑动条位置
        num = cv2.getTrackbarPos('num', window_name)
        block_size = cv2.getTrackbarPos('blockSize', window_name)

        # 计算视差
        disparity = compute_disparity(img_left, img_right, num, block_size)

        if app == 0:
            print('视差图维度: ' + str(disparity.ndim))
            print(type(disparity))
            max_index = np.unravel_index(np.argmax(disparity, axis=None), disparity.shape)
            app = 1

        # 视差图归一化
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity_color = cv2.applyColorMap(disparity_normalized, 2)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, disparity_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    # 加载校正后的左右图像
    img_left = cv2.imread('leftImgRectified.png', 0)  
    img_right = cv2.imread('rightImgRectified.png', 0)  

    # 调整视差参数并计算视差图
    adjust_disparity_parameters('SGBM', img_left, img_right)

if __name__ == "__main__":
    main()
