import cv2
import os

def split_image(image_path, output_folder, chunk_size=1024):
    
    image_name = os.path.basename(image_path).split('.')[0]
    image_output_folder = os.path.join(output_folder, image_name)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    img = cv2.imread(image_path)
    height, width, _ = img.shape
    x_chunks = (width + chunk_size - 1) // chunk_size
    y_chunks = (height + chunk_size - 1) // chunk_size

    for i in range(x_chunks):
        for j in range(y_chunks):
            left = i * chunk_size
            upper = j * chunk_size
            right = min(left + chunk_size, width)
            lower = min(upper + chunk_size, height)

            if  upper + chunk_size > height:
                upper = height - chunk_size
            if  left + chunk_size > width:
                left = width - chunk_size
            
            chunk = img[upper:lower, left:right]
            output_path = os.path.join(image_output_folder, f"{image_name}_{i}_{j}.png")
            cv2.imwrite(output_path, chunk)

if __name__ == "__main__":
    image_paths = ['/data/yazhou/.some/change_detection.pytorch-main/datika1.jpg', '/data/yazhou/.some/change_detection.pytorch-main/datika2.png']  
    output_folder = "splits2"

    for image_path in image_paths:
        split_image(image_path, output_folder)



