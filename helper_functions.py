import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label
from mpl_toolkits.mplot3d import Axes3D


# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict['n_cars'] = len(car_list)
    
    # Define a key "n_notcars" and store the number of notcar images
    data_dict['n_notcars'] = len(notcar_list)
    
    # Read in a test image, either car or notcar
    test_img = cv2.imread(car_list[0])
    
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict['image_shape'] = test_img.shape
    
    # Define a key "data_type" and store the data type of the test image.
    data_dict['data_type'] = test_img.dtype
    
    # Return data_dict
    return data_dict

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def convert_color(image, color_space='RGB'):
    # assume source color space is BGR
    # which is the default cv2.imread of jpeg/png
    if color_space == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif color_space == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space == 'RGB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return None

def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return bin_centers, hist_features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def get_hog_features(img, orient, pix_per_cell, cell_per_block, fv=True, vis=False):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, visualise=True, 
                                  feature_vector=fv)
        return features, hog_image
    else:
        features           = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, visualise=False, 
                                  feature_vector=fv)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='YCrCb', 
                     spatial_size=(16, 16),hist_bins=16, 
                     hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2):
    
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for f in imgs:
        # Read in each one by one
        # print(f)
        img = cv2.imread(f)
        
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(img, color_space=color_space)
        
        # Apply bin_spatial() to get spatial color features
        bin_fv = bin_spatial(feature_image, size = spatial_size)
        
        # Apply color_hist() to get color histogram features
        bincen, hist_fv = color_hist(feature_image, nbins=hist_bins)

        # Assuming 2nd channel because L channel of HLS is assumed
        hog_fv0 = get_hog_features(feature_image[:,:,0], 
                                  hog_orient, hog_pix_per_cell, 
                                  hog_cell_per_block, fv=True, vis=False)
        
        hog_fv1 = get_hog_features(feature_image[:,:,1], 
                                  hog_orient, hog_pix_per_cell, 
                                  hog_cell_per_block, fv=True, vis=False)
        
        hog_fv2 = get_hog_features(feature_image[:,:,2], 
                                  hog_orient, hog_pix_per_cell, 
                                  hog_cell_per_block, fv=True, vis=False)
        
        features.append(np.concatenate((bin_fv, hist_fv, hog_fv0, hog_fv1, hog_fv2)))
        
    # Return list of feature vectors
    return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_hog_features(imgs, color_space='YCrCb', 
			 hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2):

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for f in imgs:
        # Read in each one by one
        # print(f)
        img = cv2.imread(f)

        # apply color conversion if other than 'RGB'
        feature_image = convert_color(img, color_space=color_space)

        # Assuming 2nd channel because L channel of HLS is assumed
        hog_fv0 = get_hog_features(feature_image[:,:,0],
                                  hog_orient, hog_pix_per_cell,
                                  hog_cell_per_block, fv=True, vis=False)

        hog_fv1 = get_hog_features(feature_image[:,:,1],
                                  hog_orient, hog_pix_per_cell,
                                  hog_cell_per_block, fv=True, vis=False)

        hog_fv2 = get_hog_features(feature_image[:,:,2],
                                  hog_orient, hog_pix_per_cell,
                                  hog_cell_per_block, fv=True, vis=False)

        features.append(np.concatenate((hog_fv0, hog_fv1, hog_fv2)))

    # Return list of feature vectors
    return features

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def get_labeled_bboxes(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = [(np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))]
        bboxes.append(bbox)
    # Return the image
    return bboxes

