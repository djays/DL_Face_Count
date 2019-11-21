#!/usr/bin/env python3
"""
Main file for running the application
"""

import argparse
from pathlib import Path
import pandas as pd
import models
import config
import logging
import metrics


def existing_path(path):
    path = Path(path)

    if not path.exists():
        raise argparse.ArgumentTypeError('file/directory doesn\'t exist')

    if path.is_file() and not path.suffix.lower() in config.SUPPORTED_IMG_EXTN:
        print(path.suffix.lower())
        raise argparse.ArgumentTypeError('Only file types with extension jpg or png are accepted')

    return path

def get_logger():
    logger = logging.getLogger(config.APP_NAME)
    logger.setLevel(config.APP_LOG_LEVEL)
    hdlr = logging.FileHandler(config.APP_LOG_FILE)
    logger.addHandler(hdlr)
    logger.propagate = False
    return logger

def get_data(path):
    """Generate dataframe from the given paths """
    # If it's a directory, scan for all supported extension and flatten the final list
    if path.is_dir():
        img_list = [list(path.glob('*' + extn)) for extn in config.SUPPORTED_IMG_EXTN]
        img_list = [ f for a_list in img_list for f in a_list ]
    else:
        img_list = [path]

    face_counts = [int(p.stem[0]) for p in img_list]

    data = pd.DataFrame({'PATH': img_list, 'COUNTS': face_counts})
    return data

def print_output(paths, counts):
    for i, path in enumerate(paths):
        print(config.OUTPUT_FORMAT % (path.name, counts[i]))

if __name__ == '__main__':
    #Path.cwd().n
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train/Test Model')
    parser.add_argument('path', type=existing_path, help='image file/directory for testing')
    parser.add_argument('--model', type=str, default='MTCNN', choices=['MTCNN'], help='The model to use for face counts, default MTCNN')
    parser.add_argument('--print-metrics', action='store_true', help='Compute and Output any metrics')
    args = parser.parse_args()
    logger = get_logger()
    logger.info("App Log initiated")

    data = get_data(args.path)
    logger.info("Data built with size %d" %(len(data)))

    model = models.get_model(args.model)()
    logger.info("Model built")

    counts = model.count_faces(data['PATH'])
    print_output(data['PATH'], counts)
    if args.print_metrics:
        metrics.print_metrics(data, counts)

    logger.info("Output/metrics printed, app done!")
