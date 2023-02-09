import argparse

from utils import trans_2_coco_format

def get_args():
    parser = argparse.ArgumentParser(description='Convert my annotation to coco format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_annotation_path', '-i', default='input_annotation.json',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    parser.add_argument('--output_annotation_path', '-o', default='output_annotation.json',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    trans_2_coco_format(args.input_annotation_path, args.output_annotation_path)