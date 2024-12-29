import cv2
import numpy as np

def detect_and_draw_white_building(source_image_path, target_image_path):
    # 读取源图像和目标图像
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)
    
    if source_image is None or target_image is None:
        print("Error: Unable to open one of the image files.")
        return
    
    # 转换为灰度图像
    gray_source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    
    # 使用阈值处理提取白色区域
    _, white_mask = cv2.threshold(gray_source_image, 200, 255, cv2.THRESH_BINARY)
    
    # 形态学操作：膨胀和腐蚀
    kernel = np.ones((12, 12), np.uint8)
    dilated_mask = cv2.dilate(white_mask, kernel, iterations=2)
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 定义面积阈值
    area_threshold = 3000  # 根据需要调整
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 10)  # 红色矩形框
    
    # 显示结果图像
    # cv2.imshow('Source Image', source_image)
    # cv2.imshow('Target Image with Annotations', target_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite('find.png',target_image)

# 示例调用函数
detect_and_draw_white_building('/data/yazhou/.some/open-cd/outputs/stitched_vis_result.png', '/data/yazhou/.some/open-cd/mydata/original/1.png')



