import os
import sys
import argparse
from osgeo import gdal


def validate_geotiff(filepath):
    """
    Exit program if invalid filepath or if raster not North-aligned
    """

    if filepath is None or not os.path.exists(filepath):
        print("Filepath does not exist")
        sys.exit(1)
    
    ds = gdal.Open(filepath)

    if not ds:
        print("Invalid image")
        sys.exit(1)

    gt = ds.GetGeoTransform()
    xskew = gt[2]
    yskew = gt[4]
    # ensure geotiff is north-aligned
    if not xskew == yskew == 0:
        print("Raster is not North-up")
        sys.exit(1)


def convert_to_8bit(filepath):
    """
    If raster is 16bit, convert into 8bit and perform 2-98 scaling
    Then read into a numpy array and return the array
    """
    if 


def run_pipeline(filepath, output_dir):
    """
    Main pipeline
    """
    
    # ensure geotiff is valid file
    validate_geotiff(filepath)

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        print("Invalid output_dir")
        sys.exit(1)

    # ensure have write access in output directory
    try:
        f = open(os.path.join(output_dir, "test.txt"), "w")
    except Exception:
        print("No write access in output directory")
        sys.exit(1)
    
    split_tiles(filepath)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="Path to GeoTiff image", required=True)
    parser.add_argument("--output_dir", help="Path to output folder", required=True)
    args = parser.parse_args()

    run_pipeline(args.filepath, args.output_dir)