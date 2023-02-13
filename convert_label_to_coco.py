import argparse

from utils import trans_2_coco_format, combine_to_1_dataset
from dataset import get_datasets

def get_args():
    parser = argparse.ArgumentParser(description='Convert my annotation to coco format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_path', '-i', default='input_annotation.json',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    parser.add_argument('--output_path', '-o', default='output_annotation.json',
                        metavar='FILE', required=True,
                        help="Specify the file in which the annotation is stored")
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--split_ratio', '-s', type=float, default=0.9)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    if args.combine:
        datasets = get_datasets(args.input_path)
        combine_to_1_dataset(datasets, args.split_ratio, args.output_path)
    else:
        trans_2_coco_format(args.input_path, args.output_path)