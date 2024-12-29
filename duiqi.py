import cv2
import numpy as np

'''
对齐两图像，并返回对齐后的图像
'''

def select_points(image, num_points):
    points = []
    window_name = "Select Points"
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image)
            if len(points) >= num_points:
                cv2.destroyWindow(window_name)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    return np.array(points, dtype=np.float32)

def align_images_manual(image1, image2, num_points=4):
    # 手动选择对应点
    print("请在第一张图像中选择{}个点".format(num_points))
    points1 = select_points(image1.copy(), num_points)
    print("请在第二张图像中选择相同的{}个点".format(num_points))
    points2 = select_points(image2.copy(), num_points)

    # 计算单应性矩阵
    h_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    if h_matrix is not None:
        # 获取image2的尺寸
        height, width = image2.shape[:2]

        # 对image1应用单应性变换以对齐到image2
        aligned_image1 = cv2.warpPerspective(image1, h_matrix, (width, height))
    else:
        print("无法计算单应性矩阵")
        aligned_image1 = image1  # 如果找不到可靠的单应性矩阵，则不进行变换

    return aligned_image1, h_matrix

# 加载图像
img1 = cv2.imread('2.png')
img2 = cv2.imread('1.png')

# 确保两张图片已经正确加载
if img1 is None or img2 is None:
    print("未能加载图像，请检查文件路径。")
else:
    # 执行对齐操作
    aligned_img1, h_matrix = align_images_manual(img1, img2, num_points=4)

    if h_matrix is not None:
        # 显示对齐后的图像
        cv2.imshow("Aligned Image", aligned_img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存对齐后的图像
        cv2.imwrite("aligned_building_00.png", aligned_img1)
    else:
        print("未能成功对齐图像")