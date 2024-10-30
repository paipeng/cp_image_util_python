from PIL import Image, ImageFilter
import numpy as np
import cv2

def imageCrop(image: Image, crop_x: int, crop_y: int, crop_width: int, crop_height: int) -> Image:
    cropped_image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    return cropped_image

def convertImageToNumpyArray(image: Image) -> np.array:
    return np.array(image)

def convertNPArrayToImage(matrix: np.array) -> Image:
    return Image.fromarray(matrix)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def readImage(image_path: str) -> Image:
    return Image.open(image_path)

def print_matrix(matrix):
    height, width = matrix.shape
    print("matrix dimensions:", width, " x ", height)
    for row in range(0, height, 1):
        for col in range(0, width, 1):
            pixel = matrix[row, col]  # Access pixel values
            print(pixel, end=" ")
        print()


def print_matrix2(matrix):
    height, width = matrix.shape
    print("matrix dimensions:", width, " x ", height)
    print('[')
    for row in range(0, height, 1):
        print('[', end= '')
        for col in range(0, width, 1):
            pixel = matrix[row, col]  # Access pixel values
            if col < width - 1:
                print(f"{pixel:2d}", end=",")
            else:
                print(f"{pixel:2d}", end='')
        if row < height - 1:
            print(']', end= ',')
        else:
            print(']', end= '')
        print('')
    print(']')

def calculateSharpness(image: Image):
    laplacianFilter = ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), 1, 0)
    laplacianFilter = ImageFilter.Kernel((3, 3), (0, -1, 0, 0, 4, 0, 0, -1, 0), 1, 0)
    
    if image.mode == 'L':
        grayImage = image
    else:
        grayImage = image.convert('L')  # Convert to grayscale
    laplaceImage = grayImage.filter(laplacianFilter)

    # Calculate variance of the Laplacian image
    pixels = laplaceImage.getdata()
    variance = sum(p**2 for p in pixels) / len(pixels) - (sum(pixels) / len(pixels))**2

    return variance
