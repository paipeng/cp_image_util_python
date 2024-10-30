import os
import argparse
from CPImageUtil import readImage, calculateSharpness, imageCrop

def parse_arg():
    parser = argparse.ArgumentParser(description="calculate image sharpness in a given folder")
    parser.add_argument("input_path", help="Input path")
    args = parser.parse_args()
    print(f"Input file: {args.input_path}")
    return args

if __name__ == "__main__":
    args = parse_arg()
    files = os.listdir(args.input_path)
    files.sort()
    for file_name in files:
        #print(file_name)
        image_path = os.path.join(args.input_path, file_name)
        if os.path.isfile(image_path) and file_name.endswith('.bmp'):
            
            image = readImage(image_path)
            # resize
            width, height = image.size

            #print("Width:", width)
            #print("Height:", height)

            crop_size = 256
            cropImage = imageCrop(image=image, crop_x=(width-crop_size)/2, crop_y=(height-crop_size)/2, crop_width=crop_size, crop_height=crop_size)

            sharpness = calculateSharpness(image=image)
            print(image_path.replace(args.input_path + '/', '') + ': ' + str(sharpness))