import cv2
import numpy as np
import random

img = cv2.imread('Lenna.jpg')
_, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('binary.jpg', img_binary)

# define kernel for using in binary dilation 
kernel_1 = np.ones((3,3), np.uint8) 
kernel_255 = np.full((3,3), 255, np.uint8, order='C')

# or 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# # # #
# SELF-IMPLEMENT ---------------------------------
# # # #

def is_binary(image):
    """Checks if an image is binary (contains only 0s and 1s)."""
    unique_values = np.unique(image)
    return np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1])

def dilate(image, kernel):
    """Performs dilation on binary or grayscale images."""
    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Pad image
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    dilated = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + kernel_h, j:j + kernel_w]
            if is_binary(image):  # Check if binary
                dilated[i, j] = 1 if np.any(region[kernel == 1] == 1) else 0
            else:  # Grayscale case (maximum filter)
                dilated[i, j] = np.max(region[kernel == 1])

    return dilated

def erode(image, kernel):
    """Performs erosion on binary or grayscale images."""
    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Pad image
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    eroded = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + kernel_h, j:j + kernel_w]
            if is_binary(image):
                eroded[i, j] = 1 if np.all(region[kernel == 1] == 1) else 0
            else:
                eroded[i, j] = np.min(region[kernel == 1])
    return eroded

def opening(image, kernel):
    return dilate(erode(image, kernel))

def closing(image, kernel):
    return erode(dilate(image, kernel))

def hit_or_miss(image, kernel, tolerance=0):
    kernel_h, kernel_w = kernel.shape
    img_h, img_w = image.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + kernel_h, j:j + kernel_w]
            if is_binary(image):
                match_hit = np.all(region[kernel == 1] == 1)
                match_miss = np.all(region[kernel == 0] == 0)
                result[i, j] = 1 if (match_hit and match_miss) else 0
            else:
                match_hit = np.all(np.abs(region[kernel == 1]) - 255 <= tolerance)
                match_miss = np.all(region[kernel == 0] <= tolerance)
                result[i, j] = 1 if (match_hit and match_miss) else 0
    return result

def thinning(image):
    """
    Performs Zhang-Suen thinning algorithm on a binary image.

    Args:
        img (numpy.ndarray): Binary image (0 and 255).
    
    Returns:
        numpy.ndarray: Thinned image.
    """
    if not is_binary(image):
        print("Thinning: Not binary image")
        return None
    
    img_h, img_w = image.shape
    changed = True

    while changed:
        changed = False
        to_remove = np.zeros((img_h, img_w), dtype=np.uint8)

        for i in range(1, img_h - 1):
            for j in range(1, img_w - 1):
                if (image[i, j] == 1):
                    neighbors = image[i-1:i+2, j-1:j+2].flatten()
                    '''
                    [8 1 2] 
                    [7   3]
                    [6 5 4]
                    '''
                    P = [neighbors[1], neighbors[2], neighbors[5],
                         neighbors[8], neighbors[7], neighbors[6],
                         neighbors[3], neighbors[0]]
                    A = sum((P[k] == 0 and P[k + 1] % 8 == 1) for k in range(8))
                    if 2 <= sum(P) and A == 1 and P[0] * P[2] * P[4] == 0 and P[0] * P[4] * P[6] == 0:
                        to_remove[i, j] = 1

        image[to_remove == 1] = 0
        changed = np.any(to_remove)

    return image * 255

def boundary_extraction(image, kernel):
    if not is_binary(image):
        print("Boundary Extraction: Not binary image")
        return None
    
    eroded = erode(image, kernel)
    boundary = image - eroded
    return boundary

def hole_filling(image, kernel, seed_point=None):
    """
    Fills holes in a binary image using morphological dilation.
    
    Args:
        image (numpy.ndarray): Binary image (0s and 1s).
        kernel (numpy.ndarray): Structuring element.
        seed_point (tuple): (row, col) coordinates of a point inside the hole.
    
    Returns:
        numpy.ndarray: Image with filled holes.
    """    

    def get_hole(image):
        """Finds all seed points inside holes (background pixels surrounded by foreground)."""
        holes = []
        img_h, img_w = image.shape
        for i in range(1, img_h - 1):
            for j in range(1, img_w - 1):
                if image[i, j] == 0:  # Background pixel
                    neighbors = image[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == 1):  # Must have at least one foreground neighbor
                        holes.append((i, j))
        
        if len(holes) == 0:
            return None
        return random.choice(holes)  # Return list of all hole seed points

    if not is_binary(image):
        print("Hole Filling: Not binary image")
        return None    

    if seed_point is None:
        seed_point = get_hole(image)
        if not seed_point:
            print("No holes detected!")
            return image

    # Get the complement of the image (invert 0s and 1s)
    image_complement = 1 - image

    filled = np.zeros_like(image)
    filled[seed_point] = 1

    while True:
        # Expand the region
        dilated = dilate(filled, kernel)
        # Constrain expansion
        new_filled = np.minimum(dilated, image_complement)
        # Check for convergence
        if np.array_equal(new_filled, filled):
            break
        filled = new_filled

    return image | filled

def extract_connected_components(image):
    """Extracts connected components from a binary image using DFS.
    
    Args:
        image (numpy.ndarray): Binary image (0s and 1s).
    
    Returns:
        numpy.ndarray: Labeled image where each component has a unique ID.
        int: Total number of connected components found.
    """
    if not is_binary(image):
        print("Extracting Connected Components: Not binary image")
        return None  

    img_h, img_w = image.shape
    # Output labeled image
    labeled_image = np.zeros_like(image, dtype=np.int32)
    # Track visited pixels
    visited = np.zeros_like(image, dtype=bool)
    # Component Label
    label = 0 
    # Directions 8-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Depth First Search
    def dfs(r, c, label):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if not (0 <= x <= img_h and 0 <= y <= img_w):
                continue
            if visited[x, y] or image[x, y] == 0:
                continue
            visited[x, y] = True
            labeled_image[x, y] = label
            for dr, dc in directions:
                stack.append((x + dr, y + dc))

    for i in range(img_h):
        for j in range(img_w):
            if image[i, j] == 1 and not visited[i, j]:
                label += 1
                dfs(i, j, label)

    return labeled_image, label

def convex_hull(image):
    def convex_hull_points(points):
        points = sorted(points)

        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] * o[1]) * (b[0] - o[0])

        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]
    labeled_image, num_components = extract_connected_components(image)
    output = np.zeros_like(image)

    for label in range(1, num_components + 1):
        components_pixels = np.argwhere(labeled_image == label)
        if len(components_pixels) < 3:
            continue

        hull = convex_hull_points([tuple(p) for p in components_pixels])

        for x, y in hull:
            output[x, y] = 1
    
    return output

def thickening(image, kernel, iterations=1):
    """Performs thickening operation on a binary image.
    
    Args:
        image (numpy.ndarray): Binary image (0 and 1).
        kernel (numpy.ndarray): Structuring element for dilation.
        iterations (int): Number of thickening iterations.
    
    Returns:
        numpy.ndarray: Thickened binary image.
    """
    if not is_binary(image):
        print("Thickening: Input must be a binary image.")
        return None

    thickened = image.copy()
    for _ in range(iterations):
        dilated = dilate(thickened, kernel)  # Standard dilation
        thickened = np.maximum(thickened, dilated)  # Keep new pixels from dilation

    return thickened

def pruning(image, max_length=5):
    """Removes small branches from a skeletonized image.
    
    Args:
        image (numpy.ndarray): Skeletonized binary image.
        max_length (int): Maximum branch length to remove.
    
    Returns:
        numpy.ndarray: Pruned skeleton image.
    """
    def count_neighbors(image, x, y):
        """Counts the number of 1-valued neighbors for a given pixel."""
        neighbors = image[x-1:x+2, y-1:y+2].flatten()
        return np.sum(neighbors) - image[x, y]  # Exclude itself

    def find_endpoints(image):
        """Finds all endpoints in a skeletonized binary image."""
        endpoints = []
        img_h, img_w = image.shape

        for i in range(1, img_h - 1):
            for j in range(1, img_w - 1):
                '''Pixels with exactly one neighbor.'''
                if image[i, j] == 1 and count_neighbors(image, i, j) == 1:
                    endpoints.append((i, j))

        return endpoints
        
    if not is_binary(image):
        print("Pruning: Input must be a binary skeleton image.")
        return None
    
    pruned = image.copy()

    for _ in range(max_length):
        endpoints = find_endpoints(pruned)
        if not endpoints:
            break  
        for x, y in endpoints:
            pruned[x, y] = 0  # Remove endpoint

    return pruned


# # # #
# CALL SELF-IMPLEMENT ---------------------------------
# # # #

binary_dilate = dilate(img_binary, kernel_1)
cv2.imwrite('binary_dilate.jpg', binary_dilate)

binary_erode = erode(img_binary, kernel_1)
cv2.imwrite('binary_erode.jpg', binary_erode)

cv_binary_erode = cv2.erode(img_binary, kernel_1, iterations=1)
cv2.imwrite('cv_binary_erode.jpg', cv_binary_erode)


# # # #
# OPENCV -----------------------------------------
# # # #


img_dilation = cv2.dilate(img, kernel, iterations=1)

# cv2.imwrite('img_dilation.jpg', img_dilation)

# https://web.archive.org/web/20160322113207/http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/