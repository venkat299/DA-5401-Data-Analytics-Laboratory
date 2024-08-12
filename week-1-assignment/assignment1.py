#----------------------------------------------------------------------------
# Created By  : Venkatesh Duraiarasan, venkateshie@live.com (DA24C021)
# Created Date: 2024-08-11
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ================
# Data Acquisition [20 points] 
# ================
# sample image "psyduck.jpeg" is converted to "psyduck.csv" using extenal tools


# =========================
# Data Cleansing & Loading [10 points]
# =========================

# read from csv and convert to dataframe
_dir = os.path.dirname(__file__)
img = pd.read_csv(os.path.join(_dir, "./psyduck.csv"), names=("x","y"))

# offset negative coordinates(if present) in the image dataframe
x_min, y_min = (img['x'].min(), img['y'].min())
if x_min<0:
    img['x'] = img['x'].add(x_min+1)
    x_min = (round(x_min*-1) if x_min<0 else x_min)
if y_min<0:
    y_min = (round(y_min*-1) if y_min<0 else y_min)
    img['y'] = img['y'].add(y_min+1)

# discretization steps
# set the resolution
img = img.mul(1000)

# convert float to int
img = img.astype(int, errors='ignore')

# create a sparse matrix : img_matrix
_max = max(img['x'].max(),img['y'].max())
img_matrix = np.zeros((_max+1,_max+1))
img_matrix[img['x'], img['y']]=1 # update the cells
del img

img_original    = np.nonzero(img_matrix)

# ==============
# Transformation [10 points]
# ==============

# direct way : using np.rot
# img_90_left     = np.nonzero(np.rot90(img_matrix,k=1,axes=(1,0)))  # direct method
# img_flip_h      = np.nonzero(np.rot90(img_matrix,k=2,axes=(0,1)))  # direct method

# other way : using permutation matrix
permutation_mat_flip = np.identity(_max+1)[:,::-1]
img_90_left     = np.nonzero(np.dot( img_matrix.T, permutation_mat_flip)) 
img_flip_h     = np.nonzero(np.dot(img_matrix, permutation_mat_flip))

# ==============
# Visualization [10 points]
# ==============

# visulaize using subplots
fig, axs = plt.subplots(1,3)
fig.set_size_inches(12,4)
x1, y1 = 0, 0
x2, y2 = 1, 1
ax_1 = axs.flat[0]
ax_1.scatter(img_original[0], img_original[1])
ax_1.text(.05, 1.05, "original", transform=ax_1.transAxes, ha="left", va="top")
ax_1.xaxis.set_visible(False)
ax_1.yaxis.set_visible(False)


ax_2 = axs.flat[1]
ax_2.scatter(img_90_left[0], img_90_left[1])
ax_2.text(.05, 1.05, "rotate 90 right", transform=ax_2.transAxes, ha="left", va="top")
ax_2.xaxis.set_visible(False)
ax_2.yaxis.set_visible(False)

ax_3 = axs.flat[2]
ax_3.scatter(img_flip_h[0], img_flip_h[1])
ax_3.text(.05, 1.05, "flip horizontally", transform=ax_3.transAxes, ha="left", va="top")
ax_3.xaxis.set_visible(False)
ax_3.yaxis.set_visible(False)

plt.show()
# plt.savefig(os.path.join(_dir, "./result.png"))



