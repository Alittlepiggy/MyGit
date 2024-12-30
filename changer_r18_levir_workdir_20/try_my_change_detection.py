from opencd.apis import OpenCDInferencer
import os
import cv2
import numpy as np
# Load models into memory
inferencer = OpenCDInferencer(model='/data/yazhou/.some/open-cd/configs/changer/changer_ex_r18_512x512_40k_levircd.py', weights='/data/yazhou/.some/open-cd/changer_r18_levir_workdir_10/iter_1090.pth', classes=('unchanged', 'changed'), palette=[[0, 0, 0], [255, 255, 255]])
# # Inference

# #10---1010不错 1060很不错 1090很不错(目前最佳)
# #20---1080  800细节 880不错 1080不错 1020也还行
# #50---1000  1300不错 1150还不错
# #200---     1400还不错
# imageA1='/data/yazhou/.some/open-cd/data/LEVIR-CD/test/A/test_18.png'
# imageB1='/data/yazhou/.some/open-cd/data/LEVIR-CD/test/B/test_18.png'

# imageA1 = '/data/yazhou/.some/open-cd/splits/1/1_4_4.png'
# imageB1 = '/data/yazhou/.some/open-cd/splits/aligned_building_00/aligned_building_00_4_4.png'
# inferencer([[imageA1, imageB1]], show=False, out_dir='OUTPUT_PATH')


def get_image_pairs(folder_A, folder_B):
    image_pairs = []
    files_A = sorted(os.listdir(folder_A))
    files_B = sorted(os.listdir(folder_B))

    for file_a, file_b in zip(files_A, files_B):
        path_a = os.path.join(folder_A, file_a)
        path_b = os.path.join(folder_B, file_b)
        image_pairs.append((path_a, path_b))
    
    return image_pairs

folder_A = "/data/yazhou/.some/open-cd/splits/1"
folder_B = "/data/yazhou/.some/open-cd/splits/aligned_building_00"

# folder_A = '/data/yazhou/.some/open-cd/splits1/processed_image1'
# folder_B = '/data/yazhou/.some/open-cd/splits1/processed_image2'

# folder_A = '/data/yazhou/.some/open-cd/splits2/datika1'
# folder_B = '/data/yazhou/.some/open-cd/splits2/datika2'

image_pairs = get_image_pairs(folder_A, folder_B)


# # Input a list of images
# images = [['/data/yazhou/.some/open-cd/mydata/test/A/test_1.png', '/data/yazhou/.some/open-cd/mydata/test/A/test_2.png'], ['/data/yazhou/.some/open-cd/mydata/test/B/test_1.png', '/data/yazhou/.some/open-cd/mydata/test/B/test_2.png']] # image1_A can be a file path or a np.ndarray
# inferencer(images, show=True, wait_time=0.5) # wait_time is delay time, and 0 means forever

# # Save visualized rendering color maps and predicted results
# # out_dir is the directory to save the output results, img_out_dir and pred_out_dir are subdirectories of out_dir
# # to save visualized rendering color maps and predicted results
inferencer(image_pairs, out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')


def read_chunk_files(folder):
    chunk_files = {}
    for filename in sorted(os.listdir(folder)):
        parts = filename.split('_')
        i = int(parts[-2])
        j = int(parts[-1].split('.')[0])
        chunk_files[(i, j)] = os.path.join(folder, filename)
    return chunk_files


def stitch_images(chunk_files, original_width, original_height, chunk_size=1024):
    stitched_image = np.zeros((original_height, original_width, 3), dtype=np.float32)
    # weight_matrix = np.zeros((original_height, original_width), dtype=np.float32)

    x_chunks = (original_width + chunk_size - 1) // chunk_size
    y_chunks = (original_height + chunk_size - 1) // chunk_size

    for j in range(y_chunks):
        for i in range(x_chunks):
            if (i, j) in chunk_files:
                chunk_path = chunk_files[(i, j)]
                chunk = cv2.imread(chunk_path).astype(np.float32)
                
                left = i * chunk_size
                upper = j * chunk_size
                right = min(left + chunk_size, original_width)
                lower = min(upper + chunk_size, original_height)
                
                if  upper + chunk_size > original_height:
                    upper = original_height - chunk_size
                if  left + chunk_size > original_width:
                    left = original_width - chunk_size

                stitched_image[upper:lower, left:right] += chunk[:lower-upper, :right-left]
                # weight_matrix[upper:lower, left:right] += 1.0
    
    # Normalize the stitched image by the weight matrix
    # stitched_image /= np.maximum(weight_matrix[:, :, np.newaxis], 1e-6)
    
    return stitched_image.astype(np.uint8)

vis_output_folder = os.path.join('outputs', "vis")
# Read original images to get their dimensions

original_width = 8192
original_height = 4454

# Stitch the visualized results
vis_chunk_files = read_chunk_files(vis_output_folder)
stitched_vis_image = stitch_images(vis_chunk_files, original_width, original_height)

# Save the stitched visualized result
stitched_vis_output_path = os.path.join('outputs', "stitched_vis_result.png")
cv2.imwrite(stitched_vis_output_path, stitched_vis_image)

print(f"Stitched visualized result saved to {stitched_vis_output_path}")



