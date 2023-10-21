import os
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to GeoTiff image", required=True)
    args = parser.parse_args()

    run_pipeline(args.filepath)