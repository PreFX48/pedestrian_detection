import numpy
from skimage.io import imread
from scipy.ndimage import convolve
from skimage import img_as_float
import math
from skimage.transform import resize
from skimage.feature import hog


HOG_CELL_ROWS = 8
HOG_CELL_COLS = 8
HOG_BIN_COUNT = 8
HOG_BLOCK_ROW_CELLS = 2
HOG_BLOCK_COL_CELLS = 2
EPSILON = 0.0000000001


def normalize_angle(angle):
    while angle >= 2*math.pi:
        angle -= 2*math.pi
    while angle < 0:
        angle += 2*math.pi
    return angle

def extract_hog(img, roi, flip_left_to_right=False):
    img = img.copy()[roi[1]:roi[3], roi[0]:roi[2]]
    img = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    img = numpy.resize(img, (128, 64))
    if flip_left_to_right:
        img = numpy.fliplr(img)
    return hog(img, orientations=HOG_BIN_COUNT, cells_per_block=(HOG_BLOCK_ROW_CELLS, HOG_BLOCK_COL_CELLS))


# def extract_hog(img, roi, flip_left_to_right=False):
#     img = img.copy()[roi[1]:roi[3], roi[0]:roi[2]]
#     img = resize(img, (128, 64))
#     img = img_as_float(img)
#
#     if flip_left_to_right:
#         img = numpy.fliplr(img)
#
#     height = img.shape[0]
#     width = img.shape[1]
#     intensity_img = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
#     grad_x = numpy.empty(intensity_img.shape[:2], dtype='float64')
#     grad_y = numpy.empty_like(grad_x)
#     grad_power = numpy.empty_like(grad_x)
#     grad_direction = numpy.empty_like(grad_x)
#     x_kernel = numpy.zeros((3, 3))
#     y_kernel = numpy.zeros((3, 3))
#     x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#     y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#     grad_x = convolve(intensity_img, x_kernel)
#     grad_y = convolve(intensity_img, y_kernel)
#     grad_power = (grad_x**2 + grad_y**2)**0.5
#     grad_direction = numpy.vectorize(math.atan2)(grad_y, grad_x)
#     grad_direction = numpy.vectorize(normalize_angle)(grad_direction)
#     grad_direction /= 2*math.pi
#     grad_direction *= HOG_BIN_COUNT
#     grad_direction = numpy.vectorize(round)(grad_direction) % HOG_BIN_COUNT
#
#     # разделение на ячейки:
#     cell_histograms = numpy.zeros((height // HOG_CELL_ROWS, width // HOG_CELL_COLS, HOG_BIN_COUNT), dtype='float64')
#     for y in range(0, height - height % HOG_CELL_ROWS, HOG_CELL_ROWS):
#         for x in range(0, width - width % HOG_CELL_COLS, HOG_CELL_COLS):
#             for cell_y in range(HOG_CELL_ROWS):
#                 for cell_x in range(HOG_CELL_COLS):
#                     cell_histograms[y // HOG_CELL_ROWS, x // HOG_CELL_COLS, grad_direction[y + cell_y, x + cell_x]] += grad_power[y + cell_y, x + cell_x]
#     # разделение на блоки:
#     image_descriptor = numpy.empty(0)
#     for y in range(height//HOG_CELL_ROWS + 1 - HOG_BLOCK_ROW_CELLS):
#         for x in range(width//HOG_CELL_COLS + 1 - HOG_BLOCK_COL_CELLS):
#             block_descriptor = numpy.empty(0)
#             for block_y in range(HOG_BLOCK_ROW_CELLS):
#                 for block_x in range(HOG_BLOCK_COL_CELLS):
#                     block_descriptor = numpy.hstack((block_descriptor, cell_histograms[y + block_y, x + block_x]))
#             block_descriptor /= math.sqrt(sum(block_descriptor ** 2) + EPSILON)
#             image_descriptor = numpy.hstack((image_descriptor, block_descriptor))
#     return image_descriptor

def get_hog_blocks(img):
    img = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
    hog_vector = hog(img, orientations=HOG_BIN_COUNT, cells_per_block=(HOG_BLOCK_ROW_CELLS, HOG_BLOCK_COL_CELLS))
    hog_vector = hog_vector.reshape((img.shape[0]//HOG_CELL_ROWS + 1 - HOG_BLOCK_ROW_CELLS,
                                     img.shape[1]//HOG_CELL_COLS + 1 - HOG_BLOCK_COL_CELLS,
                                     HOG_BIN_COUNT * HOG_BLOCK_ROW_CELLS * HOG_BLOCK_COL_CELLS))
    return hog_vector

# def get_hog_blocks(img):
#     img = img.copy()
#     img = img_as_float(img)
#
#     height = img.shape[0]
#     width = img.shape[1]
#     intensity_img = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
#     grad_x = numpy.empty(intensity_img.shape[:2], dtype='float64')
#     grad_y = numpy.empty_like(grad_x)
#     grad_power = numpy.empty_like(grad_x)
#     grad_direction = numpy.empty_like(grad_x)
#     x_kernel = numpy.zeros((3, 3))
#     y_kernel = numpy.zeros((3, 3))
#     x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#     y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#     grad_x = convolve(intensity_img, x_kernel)
#     grad_y = convolve(intensity_img, y_kernel)
#     grad_power = (grad_x**2 + grad_y**2)**0.5
#     grad_direction = numpy.vectorize(math.atan2)(grad_y, grad_x)
#     grad_direction = numpy.vectorize(normalize_angle)(grad_direction)
#     grad_direction /= 2*math.pi
#     grad_direction *= HOG_BIN_COUNT
#     grad_direction = numpy.vectorize(round)(grad_direction) % HOG_BIN_COUNT
#
#     # разделение на ячейки:
#     cell_histograms = numpy.zeros((height // HOG_CELL_ROWS, width // HOG_CELL_COLS, HOG_BIN_COUNT), dtype='float64')
#     for y in range(0, height - height % HOG_CELL_ROWS, HOG_CELL_ROWS):
#         for x in range(0, width - width % HOG_CELL_COLS, HOG_CELL_COLS):
#             for cell_y in range(HOG_CELL_ROWS):
#                 for cell_x in range(HOG_CELL_COLS):
#                     cell_histograms[y // HOG_CELL_ROWS, x // HOG_CELL_COLS, grad_direction[y + cell_y, x + cell_x]] += grad_power[y + cell_y, x + cell_x]
#     # разделение на блоки:
#     block_feature_number = HOG_BIN_COUNT * HOG_BLOCK_ROW_CELLS * HOG_BLOCK_COL_CELLS
#     block_matrix = numpy.empty((height // HOG_CELL_ROWS + 1 - HOG_BLOCK_ROW_CELLS,
#                                 width // HOG_CELL_COLS + 1 - HOG_BLOCK_COL_CELLS,
#                                 block_feature_number), dtype='float64')
#     for y in range(height//HOG_CELL_ROWS + 1 - HOG_BLOCK_ROW_CELLS):
#         for x in range(width//HOG_CELL_COLS + 1 - HOG_BLOCK_COL_CELLS):
#             block_descriptor = numpy.empty(0)
#             for block_y in range(HOG_BLOCK_ROW_CELLS):
#                 for block_x in range(HOG_BLOCK_COL_CELLS):
#                     block_descriptor = numpy.hstack((block_descriptor, cell_histograms[y + block_y, x + block_x]))
#             block_descriptor /= math.sqrt(sum(block_descriptor ** 2) + EPSILON)
#             block_matrix[y, x] = block_descriptor
#     return block_matrix
