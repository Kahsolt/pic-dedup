#!/usr/bin/env python3

## 0. User Settings

# sim_thresh for **simgrp** empirical references:
#  0.70 ~ 0.80: pictures that are alike in colormaps or histograms
#  0.80 ~ 0.95: similar in hues and color-blocks, typically a picture series
#  0.95 ~ 1.00: proximately the same picture but only differs in resolution or size
# suggested value: 0.70 ~ 0.95
#
# sim_thresh for **filter** empirical references:
#  0.80 ~ 0.95: similiar hue presented
#  0.95 ~ 1.00: proximately the same hue presented
# suggested value: 0.85 ~ 0.95
SIMILAR_THRESHOLD = 0.80

# hw-ratio tolerance
# suggested value: 10 ~ 30
HWR_TOLERANCE = 20

# only compare the inscribed circle area
ROUND_MASK = False

# fast search config set:
#  1. only compare images within HWR_TOLERANCE
#  2. only compare edge featvec
FAST_SEARCH = True

# image feature vector width/height
# suggested value: 8, 12, 16, 24, 32
FEATURE_VECTOR_HW = 16

# principle hues count of a image
# suggested value: 3 ~ 5
PRINCIPLE_HUES = 4

# main window size
# suggested value: (800, 600), (912, 684), (1024, 768)
WINDOW_SIZE = (800, 600)

# picture hwlimit in preview, please set according to WINDOW_SIZE
# suggested value: 250, 300, 350
PREVIEW_HW = 250

# split preview 
PREVIEW_SPLIT = True


## 1. System Settings

# bulk commit after every certain actions 
DB_COMMIT_TTL = 300

# index recursively into subfolders
RECURSIVE_INDEX = False

# image extionsions, indexer identify files by extension name
IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']

# do not use these names as folder name, rather use its parents' basename
FOLDER_NAME_BLACKLIST = ['image', 'images', 'manga', 'mangas', 'picture', 'pictures', 'data', 'webdata']

# global sleep interval in seconds
# suggested value: 0.5 ~ 1.5
SLEEP_INTERVAL = 1.0

# worker count factor (e.g. set to 2 means 2 * os.cpu_count())
# suggested value: 2.0 ~ 4.0
WORKER_FACTOR = 2.0

# cache size of item count
# suggested value: 100 ~ 300
CACHE_SIZE = 150

# index database file, set None for auto configuring
DB_FILE = None

# auto reindex after reset db (this should take a century long..)
AUTO_REINDEX = False


## 2. System Constants (hard coded with careful design, DO NOT TOUCH !!)

# padding value of non-square images
# FIXED VALUE: 255 (pure white color pixel)
NONE_PIXEL_PADDING = 255

# reduce hues in HSV mode
# FIXED VALUE: (16, 8, 8)
REDUCED_HUE_SCALES = (16, 8, 8)

# reduce hues in HSV mode
# FIXED VALUE: 95 (damn empirical value)
HUE_DISTINGUISH_DISTANCE = 95

# const max retval of rgb_distance()
# FIXED VALUE: 764.8339663572415
HUE_MAX_DISTANCE = 764.8339663572415