import numpy as np

from features import *

def extract_features(img, color_space=color_space, spatial_size=spatial_size,
                     hist_bins=hist_bins, orient=orient, 
                     pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                     hog_channel=hog_channel, spatial_feat=spatial_feat, 
                     hist_feat=hist_feat, hog_feat=hog_feat, precomputed_hog_features=None):
    """
    Extract features from a single image.
    """
    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if precomputed_hog_features is not None:
            hog_features = precomputed_hog_features
        elif hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
