color_space = 'YCrCb' # RGB, HSV, LUV, HLS, YUV or YCrCb

## HOG
hog_feat = True # HOG features on or off
orient = 9 # number of orientation bins
pix_per_cell = 8 # size of a cell in pixels
cell_per_block = 2 # number of cells per block
hog_channel = 'ALL' # image channel to include. Can be 0, 1, 2, or "ALL"

## Spatial
spatial_feat = True # Spatial features on or off
spatial_size = (32, 32) # Spatial binning dimensions

## Histogram
hist_feat = True # Histogram features on or off
hist_bins = 32    # Number of histogram bins

## Misc
window_size = (64, 64) # default sliding window size
y_start_stop = [390, 670] # min and max y coordinate to search in slide_window()
