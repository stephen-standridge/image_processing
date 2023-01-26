import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import threading
import os
import argparse
import math

threads = []
def get_dominant_color(image_path):
    # read image file
    img = cv2.imread(image_path)

    # resize image to speed up processing
    img = cv2.resize(img, (200, 200))

    # convert image to the RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # cluster the pixels and assign labels
    clt = KMeans(n_clusters = 10)
    clt.fit(img)
    colors = clt.cluster_centers_
    labels = clt.labels_
    lumin_list = []

    # count labels to find most popular
    label_counts = Counter(labels)
    total_pixels = img.shape[0]
    dominant_colors = []
    color_count = 10
    ramp_colors = colors.astype(int)
    # calculate and add luminosity to the color
    for color in ramp_colors:
        R = color[0]
        G =	color[1]
        B = color[2]
        luminance = math.sqrt((0.299 * (R*R)) + (0.587 * (G*G)) + (0.114 * (B*B)))
        
        lumin_list.append(luminance)

    # add luminosity vals into numpy array
    color_and_lumin = np.insert(ramp_colors, 3, lumin_list, axis=1)
    # sort colors based on luminance
    dominant_colors = sorted(color_and_lumin, key=lambda x: x[3])
		
    # create output image showing dominant color
    output_image = np.zeros((20 * color_count, 20, 3), dtype = "uint8")
    for i, color in enumerate(dominant_colors):
        # print(color)
        output_image[20*i:20*(i+1)] = [color[2], color[1], color[0]]

    
    # create a folder if it doesn't exist
    if not os.path.exists("extracted_colors"):
        os.mkdir("extracted_colors")
    
    image_name = image_path.split('/')[-1]
    # save the output image
    cv2.imwrite(f"extracted_colors/{image_name}", output_image)
    
    # display the image
    cv2.imshow(f"Dominant Colors of {image_path}", output_image)
    print(f"Dominant Colors of {image_path}")
    for i, color in enumerate(dominant_colors):
        print(f"Color {i+1} - {color[:3]}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_paths", nargs='+', help="paths to images to be processed")
    args = parser.parse_args()
    print(args)
    for image_path in args.image_paths:
        thread = threading.Thread(target=get_dominant_color, args=(image_path,))
        thread.start()
        threads.append(thread)

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    # clear all threads
    threading._cleanup()