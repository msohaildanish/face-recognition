from recognizer import Recognizer
import argparse


def get_args():
    
    parse = argparse.ArgumentParser(description="Operation to create database")
    parse.add_argument('--faces-dir', default='data/images/faces', type=str, help="Cropped faces dir")
    parse.add_argument('--database-dir', default='data/database', type=str, help="Database saving/loading dir")
    args = parse.parse_args()

    return args

def main(args, recognizer):
    print("old databae contains {} faces".format(recognizer.database[1]))
    recognizer.gen_batch_database(args.faces_dir)
    print("New database contains {} faaces".format(len(recognizer.database[1])))


if __name__ == '__main__':
    args = get_args()
    detector = Recognizer(args.database_dir)
    main(args, detector)