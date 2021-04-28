import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from sklearn.cluster import KMeans
from PIL import Image
from numpy import asarray

path=".\\K_Means\\"
pic_name='image.png'
test_pic=path+pic_name
#img = imread(test_pic)

image=Image.open(test_pic)
img=asarray(image) # use this way to load png file; use imread to load png file it will be float
img_size = img.shape
print(img_size)

X = img.reshape(img_size[0] * img_size[1], img_size[2]) #reshape image to flat
print(X)
km = KMeans(n_clusters=40) #
km.fit(X)
# Use the centroids to compress the image
print(type(km.cluster_centers_))
print(type(km.cluster_centers_[0]))
X_compressed = km.cluster_centers_[km.labels_]
X_compressed = np.clip(X_compressed.astype('uint8'), 0, 255)
print(X_compressed.shape)

'''numpy.clip(a, a_min, a_max, out=None, **kwargs)[source]
a = np.arange(10)
np.clip(a, 1, 8)x
array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
'''

# Reshape X_recovered to have the same dimension as the original image 128 * 128 * 3
X_compressed = X_compressed.reshape(img_size[0], img_size[1], img_size[2])
# print(km.cluster_centers_)
print(len(km.labels_))
print(X_compressed.shape)

fig, ax = plt.subplots(1, 2, figsize = (12, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(X_compressed)
ax[1].set_title('Compressed Image with 30 colors')
for ax in fig.axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

from PIL import Image
im = Image.fromarray(X_compressed)
im.save(path+"compressed"+pic_name)