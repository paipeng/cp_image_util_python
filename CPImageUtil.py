from PIL import Image
import numpy as np
import cv2
import tifffile

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

def saveBmpImage(image:Image, filename: str, dpi: float = 600):
    image.info['dpi'] = (dpi, dpi)
    image.save(filename, format="bmp", dpi=(dpi, dpi))

def saveTiffImage(image: Image, filename: str, dpi: float = 600):
    image.info['dpi'] = (dpi, dpi)
    image.save(filename, format="tiff", dpi=(dpi, dpi))

def save1bitTiffImage(image: Image, filename: str, dpi: float = 600):
    image = image.convert("1", colors=2)
    image.info['dpi'] = (dpi, dpi)
    image.save(filename, format="tiff", dpi=(dpi, dpi))

def save1bitTiff(image_data, filename, dpi=600.0):
    print(image_data)
    image_data = np.where(image_data >= 125, 1, 0)

    image_data = image_data.astype(np.uint8)
    print(image_data)
    resolution_x = dpi# / 2.54
    resolution_y = dpi# / 2.54

    with tifffile.TiffWriter(filename) as tif:
        tif.save(image_data, dtype=np.uint8, photometric='miniswhite', resolution=(resolution_x, resolution_y))



def draw_text(image, text, font, font_scale, color, x, y):
    # Define text parameters
    #text = "Hello, OpenCV!"
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #font_scale = 1
    #color = (255, 255, 255)  # White color
    thickness = 2
    line_type = cv2.FILLED

    # Get text dimensions
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    if x == -1:
        x = text_x
    # Draw text on the image
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, line_type)
    return image



def convert_to_bmp(jpeg_path: str, bmp_path: str, dpi = (600, 600)):
    try:
        # Open the JPEG image using Pillow
        with PIL.Image.open(jpeg_path) as img:
            # Convert the image to BMP format
            img.save(bmp_path, format="BMP", dpi=dpi)
            print("Conversion successful!")
    except Exception as e:
        print(f"Error converting image: {e}")

def showImage(image: np.ndarray, title: str = ''):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mkdir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def saveImage(image: np.ndarray, filename: str):
    cv2.imwrite(filename, image)

def saveImage2(image:np.ndarray, output_path: str, dpi = (600, 600)):
    image.save(output_path, format="BMP", dpi=dpi)

def readImage(image_path: str) -> Image:
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error converting image: {e}")
        return None
    
def resizeImage(image: Image, width: int, height: int, mode: Image.Resampling = Image.Resampling.BILINEAR):
    resized_image = image.resize((width, height), mode)
    return resized_image

def downsampling2xBinary(image: Image) -> Image:
    resized_image = np.zeros((int(image.height/2), int(image.width/2)), dtype=np.uint8)  # 1 channel
    #resized_image = np.ones((int(image.height), int(image.width)), dtype=np.uint8)  # 1 channel
    resized_image = resized_image * 255
    for i in range(1, image.height, 2):
        for j in range(1, image.width, 2):
            #resized_image.putpixel((j/2, i/2),image.getpixel((j, i)))
            resized_image[int((j)/2), int((i)/2)] = image.getpixel((i, j))
            #resized_image[j, i] = image.getpixel((i, j))
            if resized_image[int((j)/2), int((i)/2)] > 125:
                resized_image[int((j)/2), int((i)/2)] = 255
            else:
                resized_image[int((j)/2), int((i)/2)] = 0
    return Image.fromarray(resized_image)

def downsampling2x(image: Image) -> Image:
    resized_image = np.zeros((int(image.height/2), int(image.width/2)), dtype=np.uint8)  # 1 channel
    #resized_image = np.ones((int(image.height), int(image.width)), dtype=np.uint8)  # 1 channel
    resized_image = resized_image * 255
    for i in range(1, image.height, 2):
        for j in range(1, image.width, 2):
            #resized_image.putpixel((j/2, i/2),image.getpixel((j, i)))
            resized_image[int((j)/2), int((i)/2)] = image.getpixel((i, j))
    return Image.fromarray(resized_image)



def binaryImage(image: Image) -> Image:
    resized_image = np.zeros((int(image.height), int(image.width)), dtype=np.uint8)  # 1 channel
    for i in range(1, image.height, 1):
        for j in range(1, image.width, 1):
            resized_image[j, i] = image.getpixel((i, j))
            if resized_image[j, i] > 125:
                resized_image[j, i] = 255
            else:
                resized_image[j, i] = 0
    return Image.fromarray(resized_image)

def mean(image: Image) -> {int, int}:
    np_img = np.array(image)
    mean = np.mean(np_img[:, :])
    median = np.median(np_img[:, :])
    return (int)(mean + 0.5), int(median + 0.5)

def crop_image(image: Image, left: int, top: int, right: int, bottom: int) -> Image:
    width, height = image.size
    if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
        print('image size: ', width, '-', height, ' top: ', top, ' bottom: ', bottom, ' left: ', left, ' right: ', right)
        raise ValueError("Invalid crop area.")
    cropped_img = image.crop((left, top, right, bottom))
    return cropped_img

def convertBinary(image: Image) -> Image:
    return image.convert("L")