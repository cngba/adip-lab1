import cv2
import numpy as np

# Convert to grayscale image
img = cv2.imread('Lenna.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)


cv2.imwrite('binary.jpg', img_binary)

# define kernel for using in binary dilation 
kernel_1 = np.ones((3,3), np.uint8) 
kernel_255 = np.full((3,3), 255, np.uint8, order='C')

# or (same as kernel_1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 

# # # #
# SELF-IMPLEMENT ---------------------------------
# # # #

import numpy as np

def is_binary(image):
    """Checks if an image is binary (contains only 0s and 255s)."""
    unique_values = np.unique(image)
    return np.array_equal(unique_values, [0, 255]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1])

def dilate(image, kernel):
    """Performs dilation on a binary image using the given kernel."""
    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Pad the image to handle edge cases
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    dilated = np.zeros_like(image)

    # Apply dilation: if any pixel under the kernel is 255, the center pixel becomes 255
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + kernel_h, j:j + kernel_w]
            dilated[i, j] = 255 if np.any(region[kernel == 1] == 255) else 0
    return dilated

def erode(image, kernel):
    """Performs erosion on a binary image using the given kernel."""
    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Pad the image to handle edge cases
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    eroded = np.zeros_like(image)

    # Apply erosion: only set center pixel to 255 if all corresponding pixels in the kernel are 255
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + kernel_h, j:j + kernel_w]
            eroded[i, j] = 255 if np.all(region[kernel == 1] == 255) else 0
    return eroded

def opening(image, kernel):
    """Performs morphological opening (erosion followed by dilation)."""
    return dilate(erode(image, kernel), kernel)

def closing(image, kernel):
    """Performs morphological closing (dilation followed by erosion)."""
    return erode(dilate(image, kernel), kernel)

def hit_or_miss(image, kernel, tolerance=0):
    """Performs the hit-or-miss transformation for binary images."""
    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Pad the image to handle edge cases
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    # Apply hit-or-miss transformation
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + kernel_h, j:j + kernel_w]
            match_hit = np.all(region[kernel == 1] == 255)  # Check if the hit region matches
            match_miss = np.all(region[kernel == 0] == 0)  # Check if the miss region matches
            result[i, j] = 255 if (match_hit and match_miss) else 0
    return result

def thinning(image):
    """
    Performs Zhang-Suen thinning algorithm on a binary image.
    Reduces white regions to single-pixel width while preserving connectivity.
    """
    # Convert 255 to 1 for processing
    image = (image // 255).astype(np.uint8)
    
    img_h, img_w = image.shape
    changed = True

    while changed:
        changed = False
        
        # First sub-iteration
        to_remove = np.zeros((img_h, img_w), dtype=np.uint8)
        for i in range(1, img_h - 1):
            for j in range(1, img_w - 1):
                if image[i, j] == 1:
                    neighbors = image[i-1:i+2, j-1:j+2].flatten()
                    P = [neighbors[1], neighbors[2], neighbors[5], neighbors[8],
                         neighbors[7], neighbors[6], neighbors[3], neighbors[0]]

                    A = sum((P[k] == 0 and P[(k + 1) % 8] == 1) for k in range(8))
                    B = sum(P)
                    
                    if (2 <= B <= 6 and A == 1 and P[0] * P[2] * P[4] == 0 and P[2] * P[4] * P[6] == 0):
                        to_remove[i, j] = 1

        image[to_remove == 1] = 0
        changed = np.any(to_remove)

        # Second sub-iteration
        to_remove = np.zeros((img_h, img_w), dtype=np.uint8)
        for i in range(1, img_h - 1):
            for j in range(1, img_w - 1):
                if image[i, j] == 1:
                    neighbors = image[i-1:i+2, j-1:j+2].flatten()
                    P = [neighbors[1], neighbors[2], neighbors[5], neighbors[8],
                         neighbors[7], neighbors[6], neighbors[3], neighbors[0]]

                    A = sum((P[k] == 0 and P[(k + 1) % 8] == 1) for k in range(8))
                    B = sum(P)
                    
                    if (2 <= B <= 6 and A == 1 and P[0] * P[2] * P[6] == 0 and P[0] * P[4] * P[6] == 0):
                        to_remove[i, j] = 1

        image[to_remove == 1] = 0
        changed = changed or np.any(to_remove)

    # Convert back to 255 format
    return image * 255

# # # #
# CALL SELF-IMPLEMENT ---------------------------------
# # # #

binary_dilate = dilate(img_binary, kernel_1)
cv2.imwrite('binary_dilate.jpg', binary_dilate)

binary_erode = erode(img_binary, kernel_1)
cv2.imwrite('binary_erode.jpg', binary_erode)

binary_opening = opening(img_binary, kernel_1)
cv2.imwrite('binary_opening.jpg', binary_opening)

binary_closing = closing(img_binary, kernel_1)
cv2.imwrite('binary_closing.jpg', binary_closing)

hit_or_miss_img = hit_or_miss(img_binary, kernel_1)
cv2.imwrite('hit_or_miss.jpg', hit_or_miss_img)

thinning_img = thinning(img_binary)
cv2.imwrite('thinning_img.jpg', thinning_img)

# # # #
# OPENCV -----------------------------------------
# # # #

cv_binary_erode = cv2.erode(img_binary, kernel, iterations=1)
cv2.imwrite('cv_binary_erode.jpg', cv_binary_erode)

cv_binary_dilate = cv2.dilate(img_binary, kernel, iterations=1)
cv2.imwrite('cv_binary_dilate.jpg', cv_binary_dilate)

cv_binary_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel) 
cv2.imwrite('cv_binary_opening.jpg', cv_binary_opening)

cv_binary_closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('cv_binary_closing.jpg', cv_binary_closing)

cv_hit_or_miss = cv2.morphologyEx(img_binary, cv2.MORPH_HITMISS, kernel)
cv2.imwrite('cv_hit_or_miss.jpg', cv_hit_or_miss)

cv_thinning = cv2.ximgproc.thinning(img_binary)
cv2.imwrite('cv_thinning.jpg', cv_thinning)

# https://web.archive.org/web/20160322113207/http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/