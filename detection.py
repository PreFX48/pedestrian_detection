import numpy
import random
from hog import *
from sklearn.svm import LinearSVC
from skimage.transform import rescale
import time

WINDOW_HEIGHT = 128
WINDOW_WIDTH = 64
WINDOW_STEP = 8

# region coordinates legend: (x_start, y_start, x_end, y_end)

def intersect_square(region1, region2):
    if max(region1[0], region2[0]) < min(region1[2], region2[2]):
        if max(region1[1], region2[1]) < min(region1[3], region2[3]):
            return min(region1[2] - region2[0], region2[2] - region1[0]) *\
                   min(region1[3] - region2[1], region2[3] - region1[1])
    return 0

def union_square(region1, region2):
    return (region1[2] - region1[0])*(region1[3] - region1[1]) +\
           (region2[2] - region2[0])*(region2[3] - region2[1]) - intersect_square(region1, region2)

def suppress_non_maxima(windows):
    windows_maximum_flags = [True]*len(windows)
    windows_x = sorted(windows, key=lambda w: w[0])
    for window in range(len(windows_x)):
        if not windows_maximum_flags[window]:
            continue
        right_window = window+1
        while right_window < len(windows_x) and windows_x[right_window][0] < windows_x[window][2]:
            if intersect_square(windows_x[right_window][:4],
                                windows_x[window][:4]) /   \
                union_square(windows_x[right_window][:4],
                                windows_x[window][:4]) <= 0.5:
                if windows_x[window][4] >= windows_x[right_window][4]:
                    windows_maximum_flags[right_window] = False
                else:
                    windows_maximum_flags[window] = False
                    break
            right_window += 1
    windows = list(zip(windows_x, windows_maximum_flags))
    windows.sort(key=lambda w: w[0][2])
    windows_maximum_flags = list(map(lambda w: w[1], windows))
    windows_y = list(map(lambda w: w[0], windows))
    for window in range(len(windows_y)):
        if not windows_maximum_flags[window]:
            continue
        lower_window = window+1
        while lower_window < len(windows_y) and windows_y[lower_window][1] < windows_y[window][3]:
            if intersect_square(windows_y[lower_window][:4],
                                windows_y[window][:4]) /   \
                union_square(windows_y[lower_window][:4],
                                windows_y[window][:4]) <= 0.5:
                if windows_y[window][4] >= windows_y[lower_window][4]:
                    windows_maximum_flags[lower_window] = False
                else:
                    windows_maximum_flags[window] = False
                    break
            lower_window += 1
    return [windows_y[i] for i in range(len(windows_y)) if windows_maximum_flags[i]]



def train_detector(imgs, gt):
    region_sizes = [None]*len(imgs)
    positive_samples = None

    # positive samples
    counter = 0
    import time
    start_time = time.time()
    for image_number in range(len(imgs)):
        for region in gt[image_number]:
            if region_sizes[image_number] is None:
                region_sizes[image_number] = (region[3]-region[1], region[2]-region[0])
            image = imgs[image_number]
            hog = extract_hog(image, region)
            if positive_samples is None:
                positive_samples = numpy.empty((0, hog.shape[0]))
            positive_samples = numpy.vstack((positive_samples, hog))
            hog = extract_hog(image, region, flip_left_to_right=True)
            positive_samples = numpy.vstack((positive_samples, hog))
            cur_time = (time.time() - start_time)*(len(imgs)-counter) / (counter+EPSILON)
        print('positive samples:', counter, 'out of', len(imgs), 'time remaining:', str(int(cur_time//60))+'m',
                                                                                    str(int(cur_time%60))+'s')
        counter += 1


    # negative samples
    negative_samples = numpy.empty((0, positive_samples.shape[1]))
    random.seed()
    counter = 0
    for image_number in range(len(imgs)):
        for region in gt[image_number]:
            image = imgs[image_number]
            for i in range(3):
                x_start_coord = random.randint(0, imgs[image_number].shape[1] - 1 - region_sizes[image_number][1])
                y_start_coord = random.randint(0, imgs[image_number].shape[0] - 1 - region_sizes[image_number][0])
                x_end_coord = x_start_coord + region_sizes[image_number][1]
                y_end_coord = y_start_coord + region_sizes[image_number][0]
                new_region_rect = (x_start_coord, y_start_coord, x_end_coord, y_end_coord)
                if intersect_square(region, new_region_rect)/union_square(region, new_region_rect) < 0.2:
                    hog = extract_hog(image, new_region_rect)
                    negative_samples = numpy.vstack((negative_samples, hog))
                cur_time = (time.time() - start_time)*(len(imgs)-counter) / (counter+EPSILON)
        print('negative samples:', counter, 'out of', len(imgs), 'time remaining:', str(int(cur_time//60))+'m',
                                                                                    str(int(cur_time%60))+'s')
        counter += 1

    positive_labels = numpy.ones((positive_samples.shape[0], 1), dtype='uint8')
    negative_labels = numpy.zeros((negative_samples.shape[0], 1), dtype='uint8')
    samples = numpy.vstack((positive_samples, negative_samples))
    labels = numpy.vstack((positive_labels, negative_labels))

    classificator = LinearSVC()
    classificator.fit(samples, labels)

    # bootstrapping
    hard_hogs = numpy.empty((0, samples.shape[1]))
    for image_number, image in enumerate(gt):
        windows = detect(classificator, imgs[image_number])
        for window in windows:
            successFlag = False
            for pedestrian in image:
                if intersect_square(pedestrian, window[:4]) / union_square(pedestrian, window[:4]) < 0.5:
                    successFlag = True
                    break
            if not successFlag:
                hard_hogs = numpy.vstack((hard_hogs, extract_hog(imgs[image_number], window[:4])))
    hard_labels = numpy.zeros((hard_hogs.shape[0], 1), dtype='uint8')
    samples = numpy.vstack((positive_samples, hard_hogs))
    labels = numpy.vstack((positive_labels, hard_labels))
    classificator.fit(samples, labels)


    return classificator


def get_window_hog(hog_blocks, x, y):
    return hog_blocks[y//HOG_CELL_ROWS:(y+WINDOW_HEIGHT)//HOG_CELL_ROWS - 1,
                      x//HOG_CELL_COLS:(x+WINDOW_WIDTH)//HOG_CELL_COLS - 1].flatten()


def detect(model, img):
    img = img.copy()
    upscaled_images = [img]
    downscaled_images = []
    upscale_factor = upscale_step = 1.1
    downscale_factor = downscale_step = 0.94
    upscale_number = 5
    downscale_number = 40
    for i in range(upscale_number):
        upscaled_images.append(rescale(img, upscale_factor))
        upscale_factor *= upscale_step
    for i in range(downscale_number):
        if int(downscale_factor * img.shape[0]) >= WINDOW_HEIGHT and\
           int(downscale_factor * img.shape[1]) >= WINDOW_WIDTH:
            downscaled_images.append(rescale(img, downscale_factor))
            downscale_factor *= downscale_step
        else:
            break
    scaled_images = list(reversed(downscaled_images)) + upscaled_images

    windows = []
    for i, image in enumerate(scaled_images):
        if i < downscale_number:
            scale_factor = downscale_step**(downscale_number - i)
        else:
            scale_factor = upscale_step**(i - downscale_number)
        scale_factor = image.shape[0] / img.shape[0]
        hog_blocks = get_hog_blocks(image)
        height = image.shape[0]
        width = image.shape[1]
        current_windows = []
        # current_window_scores = numpy.zeros(((height - WINDOW_HEIGHT)//WINDOW_STEP, (width - WINDOW_WIDTH)//WINDOW_STEP),
        #                             dtype='float64')
        cur_img = img.copy()
        for y in range(0, height - WINDOW_HEIGHT - (height - WINDOW_HEIGHT)%WINDOW_STEP, WINDOW_STEP):
            for x in range(0, width - WINDOW_WIDTH - (width - WINDOW_WIDTH)%WINDOW_STEP, WINDOW_STEP):
                hog = get_window_hog(hog_blocks, x, y)
                result = model.predict(hog)
                if result == 1:

                    score = model.decision_function(hog)[0]
                    # current_window_scores[y//WINDOW_STEP, x//WINDOW_STEP] = score
                    current_windows.append((x/scale_factor, y/scale_factor,
                                            (x + WINDOW_WIDTH)/scale_factor,
                                            (y + WINDOW_HEIGHT)/scale_factor, score))

        windows.extend(current_windows)
    windows = suppress_non_maxima(windows)
    # for window in windows:
    #     img[window[1], window[0]:window[2]] = (255, 0, 255)
    #     img[window[3]-1, window[0]:window[2]] = (255, 0, 255)
    #     img[window[1]:window[3], window[0]] = (255, 0, 255)
    #     img[window[1]:window[3], window[2]-1] = (255, 0, 255)
    # from skimage.viewer import ImageViewer
    # viewer = ImageViewer(img)
    # viewer.show()
    return windows
