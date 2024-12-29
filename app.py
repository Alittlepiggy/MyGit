from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import json

app = Flask(__name__)

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

def align_images_manual(image1, image2, points1, points2):
    h_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    if h_matrix is not None:
        height, width = image2.shape[:2]
        aligned_image1 = cv2.warpPerspective(image1, h_matrix, (width, height))
    else:
        print("无法计算单应性矩阵")
        aligned_image1 = image1

    return aligned_image1, h_matrix

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented_image = np.zeros_like(image)
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
    return segmented_image

def analyze_and_refine(aligned_image1, segmented_image1, segmented_image2, param1, param2):
    # 这里可以添加进一步的分析推理逻辑
    result_text = f"Analysis complete with parameters: {param1}, {param2}"

    # 示例输出润色
    refined_result = f"Refined result based on parameters: {param1} and {param2}"

    return refined_result

@app.route('/analyze', methods=['POST'])
def analyze():
    image1_file = request.files['image1']
    image2_file = request.files['image2']
    points1_str = request.form['points1']
    points2_str = request.form['points2']
    param1 = request.form['param1']
    param2 = request.form['param2']

    points1 = json.loads(points1_str)
    points2 = json.loads(points2_str)

    image1_nparray = np.frombuffer(image1_file.read(), np.uint8)
    image1 = cv2.imdecode(image1_nparray, cv2.IMREAD_COLOR)

    image2_nparray = np.frombuffer(image2_file.read(), np.uint8)
    image2 = cv2.imdecode(image2_nparray, cv2.IMREAD_COLOR)

    points1 = np.array([(int(p['x']), int(p['y'])) for p in points1], dtype=np.float32)
    points2 = np.array([(int(p['x']), int(p['y'])) for p in points2], dtype=np.float32)

    aligned_image1, h_matrix = align_images_manual(image1, image2, points1, points2)
    if h_matrix is None:
        return jsonify({"result": "Failed to align images"}), 400

    segmented_image1 = segment_image(aligned_image1)
    segmented_image2 = segment_image(image2)

    result = analyze_and_refine(aligned_image1, segmented_image1, segmented_image2, param1, param2)

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)