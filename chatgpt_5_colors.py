import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import threading
import os
import argparse

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
    labels = clt.fit_predict(img)

    # count labels to find most popular
    label_counts = Counter(labels)
    total_pixels = img.shape[0]
    dominant_colors = []
    color_count = 5
    for color, count in label_counts.most_common(color_count):
        dominant_colors.append({"color": tuple(map(int, clt.cluster_centers_[color])),
                                "percent": count / total_pixels * 100})
    dominant_colors = sorted(dominant_colors, key=lambda x: x["percent"])

    # create output image showing dominant color
    output_image = np.zeros((20 * color_count, 20, 3), dtype = "uint8")
    for i, color in enumerate(dominant_colors):
        output_image[20*i:20*(i+1)] = color["color"][::-1]

    
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
        print(f"Color {i+1} - {color['color'][::-1]} - {color['percent']:.2f}%")
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