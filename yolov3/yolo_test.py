import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import csv

def detect_img(yolo):
    print("in detect_image")
    with open('test_malaria.txt', 'r') as filehandler:
        lines = csv.reader(filehandler, delimiter=" ")
        total_rbc=0
        #total_infected=0
        #rbc = 0
        total_trophozoite = 0
        total_schizont=0
        total_difficult=0
        total_ring=0
        total_leukocyte=0
        total_gametocyte=0
        for img_name in lines:
            print(img_name[0])
            image = Image.open(img_name[0])
            #rbc, infected = predict_single_image(img_name[0], model_rpn, model_classifier_only, cfg, class_mapping)
            r_image, rbc, trophozoite, schizont, difficult,ring, leukocyte, gametocyte  = yolo.detect_image(image)
            r_image.save('result_malaria/'+(img_name[0])[47:])
            total_rbc+=rbc
            #rbc = 0
            total_trophozoite += trophozoite
            total_schizont+=schizont
            total_difficult+=difficult
            total_ring+=ring
            total_leukocyte+=leukocyte
            total_gametocyte+=gametocyte
            print("total red blood cell all image: "+ str(total_rbc))
            print("total trophozoite  all img: " + str(total_trophozoite))
            print("total schizont  all img: " + str(total_schizont))
            print("total diff  all img: " + str(total_difficult))
            print("total ring  all img: " + str(total_ring))            
            print("total leukocyte  all img: " + str(total_leukocyte))
            print("total gametocyte  all img: " + str(total_gametocyte))


    '''
    while True:
        print("inside while")
        img = input('chose your image:o ' )
        try:
            image = Image.open('../malaria/images/'+img)
            print('image opened')
        except:
            print('Open Error! Try again!')
            continue
        else:
            print("detecting")
            r_image = yolo.detect_image(image)
            r_image.save('result_malaria/res_'+img)
    '''
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
