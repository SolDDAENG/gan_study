import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import rescale, resize

# 원본 이미지
img = rgb2gray(data.coffee())
img = resize(img, (64, 64))
print(img.shape)

plt.axis('off')
plt.imshow(img, cmap='gray')
plt.show()

# 수평 모서리 필터
filter1 = np.asarray([
    [ 1,  0,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

new_image = np.zeros(img.shape)

img_pad = np.pad(img, 1, 'constant')

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        try:
            new_image[i, j] = \
                img_pad[i-1, j-1] * filter1[0, 0] + \
                img_pad[i-1, j] * filter1[0, 1] + \
                img_pad[i-1, j+1] * filter1[0, 2] + \
                img_pad[i, j-1] * filter1[1, 0] + \
                img_pad[i, j] * filter1[1, 1] + \
                img_pad[i, j+1] * filter1[1, 2] +\
                img_pad[i+1, j-1] * filter1[2, 0] + \
                img_pad[i+1, j] * filter1[2, 1] + \
                img_pad[i+1, j+1] * filter1[2, 2]
        except:
            pass
plt.axis('off')
plt.imshow(new_image, cmap='Greys')
plt.show()

# 수직 모서리 필터
filter2 = np.array([
    [ -1,  0,  1],
    [ -1,  0,  1],
    [ -1,  0,  1]
])

new_image = np.zeros(img.shape)

img_pad = np.pad(img,1, 'constant')

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        try:
            new_image[i,j] = \
            img_pad[i-1,j-1] * filter2[0,0] + \
            img_pad[i-1,j] * filter2[0,1] + \
            img_pad[i-1,j+1] * filter2[0,2] + \
            img_pad[i,j-1] * filter2[1,0] + \
            img_pad[i,j] * filter2[1,1] + \
            img_pad[i,j+1] * filter2[1,2] +\
            img_pad[i+1,j-1] * filter2[2,0] + \
            img_pad[i+1,j] * filter2[2,1] + \
            img_pad[i+1,j+1] * filter2[2,2] 
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='Greys');
plt.show()

# 스프라이드 2인 수평 모서리 필터
filter1 = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

stride = 2

new_image = np.zeros((int(img.shape[0] / stride), int(img.shape[1] / stride)))

img_pad = np.pad(img,1, 'constant')

for i in range(0,img.shape[0],stride):
    for j in range(0,img.shape[1],stride):
        try:
            new_image[int(i/stride),int(j/stride)] = \
            img_pad[i-1,j-1] * filter1[0,0] + \
            img_pad[i-1,j] * filter1[0,1] + \
            img_pad[i-1,j+1] * filter1[0,2] + \
            img_pad[i,j-1] * filter1[1,0] + \
            img_pad[i,j] * filter1[1,1] + \
            img_pad[i,j+1] * filter1[1,2] +\
            img_pad[i+1,j-1] * filter1[2,0] + \
            img_pad[i+1,j] * filter1[2,1] + \
            img_pad[i+1,j+1] * filter1[2,2] 
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='Greys');
plt.show()