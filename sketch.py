import cv2
import os
import argparse
import shutil

# table of formats supports in OpenCv ver. 4.6.0
supported_formats=['.bmp', '.dib', '.jpg', '.jpeg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pfm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']


def main():
    parser = argparse.ArgumentParser(description='Sketch from photo', add_help=False)
    parser._optionals.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Print this help and exit.')
    parser._positionals.add_argument('-i', '--input-dir', help='Input path dir.', required=True, metavar='DIR')
    parser._optionals.add_argument('-o', '--output-dir', help='Output path dir', required=False, metavar='DIR')
    
    parser._positionals.title = 'Required'
    parser._optionals.title = 'Optional'

    args=parser.parse_args()
    print (args)
    print('Input dir: ', args.input_dir)
    print('Output dir: ', args.output_dir)

    path_in = args.input_dir

    if args.output_dir:
        path_out = args.output_dir
    else:
        output_name = 'edited'
        path_out = os.path.join(path_in, output_name)
        
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    
    os.makedirs(path_out) 
    
    for filename in os.listdir(path_in):
        file=os.path.join(path_in,filename)
        
        if os.path.isfile(file):
            file_extention = os.path.splitext(filename)[1]
            file_out = os.path.join(path_out, f'{os.path.splitext(filename)[0]}_sketch{file_extention}')

            for format in supported_formats:
                if file_extention == format:
                    print('Making sketch for', filename)
                    image_to_sketch(file, file_out, k_size=23)


def image_to_sketch(file_in, file_out, k_size):
    # Open Image File
    img = cv2.imread(file_in)

    # Change BGR to Lab Color    
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Spliting an image into l_channel, a, b 
    l_channel, a, b = cv2.split(lab)
  
    # Contrasts l_channel Image
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(70,70))
    cl = clahe.apply(l_channel)
   
    # Merge color chain elements
    merge_img = cv2.merge((cl,a,b))
  
    # Enhances image
    enhanced_img = cv2.cvtColor(merge_img, cv2.COLOR_LAB2BGR)
   
    # Convert to Grey Image
    grey_img=cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
  
    # Invert Image
    invert_img=cv2.bitwise_not(grey_img)
   
    # Blur image
    blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size), 0)
   
    # Invert Blurred Image
    invblur_img=cv2.bitwise_not(blur_img)
   
    # Sketch Image
    sketch_img=cv2.divide(grey_img,invblur_img, scale=260.0)

    # Save Sketch
    cv2.imwrite(file_out, sketch_img)

if __name__ == '__main__':
    main()


