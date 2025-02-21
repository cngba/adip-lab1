import numpy as np

arr = np.array([[3, 3, 3],
                [3, 3, 3],
                [3, 3, 3]], dtype=np.uint8)

print(np.pad(arr, pad_width=((arr.shape[0]//2, arr.shape[0]//2),
                                            (arr.shape[1]//2, arr.shape[1]//2)), mode='constant', constant_values=0))