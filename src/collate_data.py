import os
import cv2
import numpy as np
from PIL import Image

#Directories and classes
raw_dir = os.path.join("data", "raw_downloads")
output_dir = os.path.join("data", "wound_dataset")
target_classes = ["burn", "scrape_abrasion", "cut_incision", "bruise", "ulcer"]

#Thresholds for filtering quality
#Averages of the entire image
BLUR_THRESHOLD = 80.0
DARK_THRESHOLD = 40.0
BRIGHT_THRESHOLD = 215.0

for cls in target_classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)


def is_good_quality(image_path):
    #Checks if an image passes quality standards for blurriness and exposure.
    #Read image in grayscale for quality checks
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        return False, "Unreadable"

    #Exposure Check (Mean Intensity)
    mean_brightness = np.mean(img_gray)
    if mean_brightness < DARK_THRESHOLD:
        return False, f"Too Dark ({mean_brightness:.1f})"
    if mean_brightness > BRIGHT_THRESHOLD:
        return False, f"Too Bright ({mean_brightness:.1f})"

    #Blur Check (Variance of Laplacian)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False, f"Too Blurry (Score: {laplacian_var:.1f})"

    return True, "Passed"


def clean_and_extract(source_folder, target_class, dataset_prefix):
    if not os.path.exists(source_folder):
        print(f"Skipping: Folder not found -> {source_folder}")
        return

    saved_count = 0
    rejected_count = 0

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(source_folder, filename)
            
            #Quality control check
            passed, reason = is_good_quality(src_path)
            if not passed:
                rejected_count += 1
                continue

            #Process good images
            new_filename = f"{target_class}_{dataset_prefix}_{saved_count}.jpg"
            dest_path = os.path.join(output_dir, target_class, new_filename)
            
            try:
                with Image.open(src_path) as img:
                    rgb_img = img.convert('RGB')
                    resized_img = rgb_img.resize((224, 224))
                    resized_img.save(dest_path, 'JPEG')
                saved_count += 1
            except Exception as e:
                print(f"Skipping broken file {filename}: {e}")
                
    print(f"Saved {saved_count} images to '{target_class}' ({rejected_count} rejected for poor quality).")


# Execution
print("Starting dataset cleaning\n")

clean_and_extract(os.path.join(raw_dir, "kaggle_wound", "abrasion wound"), "abrasion", "kg1")
clean_and_extract(os.path.join(raw_dir, "kaggle_wound", "bruises wound"), "bruise", "kg1")
clean_and_extract(os.path.join(raw_dir, "kaggle_wound", "cut wound"), "cut", "kg1")

clean_and_extract(os.path.join(raw_dir, "kaggle_wound"), "burn", "kg2")

print("Completed")