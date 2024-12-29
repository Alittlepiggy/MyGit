import cv2
import numpy as np

def apply_opening_filter(image, kernel_size=(20, 20)):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # 开运算：先腐蚀后膨胀，去除小的散点
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return opening

def apply_closing_filter(image, kernel_size=(25, 25)):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # 闭运算：先膨胀后腐蚀，保留较大的聚集点
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return closing

# 读取图像
image_path = '/data/yazhou/.some/open-cd/outputs/stitched_vis_result.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 二值化处理
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 应用开运算去除小的散点
closing = apply_closing_filter(binary_image)
opening = apply_opening_filter(binary_image)

# 应用闭运算保留较大的聚集点
closing = apply_closing_filter(opening)
closing = apply_opening_filter(closing,(10,10))
closing = apply_closing_filter(closing,(20,20))


# # 显示原始图像、二值化图像、开运算后的图像和闭运算后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Binary Image', binary_image)
# cv2.imshow('Opening Filtered Image', opening)
# cv2.imshow('Closing Filtered Image', closing)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存处理后的图像
closing = closing[270:4454-370,680:8192-470]
cv2.imwrite('final.png', closing)