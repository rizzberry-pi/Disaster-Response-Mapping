import os
import re
import sys
import cv2
import argparse
import numpy as np
from osgeo import gdal, osr
from pyproj import Proj, CRS
from shapely.geometry import Polygon
from skimage.morphology import skeletonize


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


def process_tif_to_array(filepath):
    """
    Read array into 3-channel BGR array
    If single-band GeoTiff, convert into 3-channel grey, grey, grey array to use as BGR
    Then perform 2-98 normalization (disabled currently)
    """
    ds = gdal.Open(filepath)
    # if single-band image
    if ds.RasterCount == 1:
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        # stack into 3-channel greyscale array
        arr = np.stack((arr, arr, arr), axis=2)
    # if 3-band image, then R = band 1, G = band 2, B = band 3
    elif ds.RasterCount == 3:
        red_band = ds.GetRasterBand(1)
        red_arr = red_band.ReadAsArray()
        green_band = ds.GetRasterBand(2)
        green_arr = green_band.ReadAsArray()
        blue_band = ds.GetRasterBand(3)
        blue_arr = blue_band.ReadAsArray()
        arr = np.stack((blue_arr, green_arr, red_arr), axis=2)
    # if 4-band image, then R = band 3, G = band 2, B = band 1
    elif ds.RasterCount == 4:
        red_band = ds.GetRasterBand(3)
        red_arr = red_band.ReadAsArray()
        green_band = ds.GetRasterBand(2)
        green_arr = green_band.ReadAsArray()
        blue_band = ds.GetRasterBand(1)
        blue_arr = blue_band.ReadAsArray()
        arr = np.stack((blue_arr, green_arr, red_arr), axis=2)
    # if some other number of bands, reject input
    else:
        print("Unknown raster format")
        sys.exit(1)

    # # now, perform 2-98 normalization (currently disabled)
    # lower_bound = np.percentile(arr.flatten(), 2)
    # upper_bound = np.percentile(arr.flatten(), 98)
    # arr = ((arr - lower_bound) / (upper_bound - lower_bound)) * 255
    # arr[arr <= 0] = 0
    # arr[arr >= 255] = 255
    arr = arr.astype(np.uint8)
    return arr


def run_pipeline(filepath, output_dir, dtm_path):
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
    
    # obtain the extent of the dataset [ulx, uly, lrx, lry] as well as the epsg code of its crs
    ds = gdal.Open(filepath)
    gt = ds.GetGeoTransform()
    src_extent = [gt[0], gt[3], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize]
    src_epsg = Proj(ds.GetProjection()).crs.to_epsg()
    ds = None

    # convert the GeoTiff into a 3-channel BGR array
    bgr_arr = process_tif_to_array(filepath)

    # run edge-detect algorithm to output mask of detected roads
    roads = edge_detect(bgr_arr)

    highlighted_roads = extract_height(roads, src_extent, dtm_path, src_epsg, gt)

    # overlay roads onto base image
    highlighted_roads = np.stack((highlighted_roads, highlighted_roads, highlighted_roads), axis=2)
    alpha = 0.3
    output = cv2.addWeighted(bgr_arr, alpha, highlighted_roads, 1 - alpha, 0.0)

    # save into output file
    output_path = os.path.join(output_dir, "output.png")
    cv2.imwrite(output_path, output)


def transform_coordinate(x, y, in_epsg, out_epsg):
    """
    Given an x and y real world coordinate defined in in_epsg,
    reproject them to the corresponding x and y in out_epsg
    and return the reprojected x, y as a tuple (x, y)
    """
    # define both the input and output srs as osgeo.osr.SpatialReference objects
    in_srs = osr.SpatialReference()
    in_srs.ImportFromEPSG(in_epsg)
    # ensures that osr.CoordinateTransform() returns the coordinates as x, y and not y, x
    in_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(out_epsg)
    out_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transformer = osr.CoordinateTransformation(in_srs, out_srs)
    x, y, _ = transformer.TransformPoint(x, y)

    return (x, y)


def query_dtm(arr, arr_gt, dtm, dtm_gt):
    # get the pixel and line values where there is detected to be a road (edge)
    lines, pixels = np.where(arr == 255)
    # stack them vertically as a (2 x n) array where one column is one point
    arr_img_coords = np.vstack((pixels, lines)).astype(np.float64)
    # convert pixels into xs and lines into ys
    wld_coords = np.zeros_like(arr_img_coords)
    wld_coords[0] = arr_img_coords[0] * arr_gt[1] + arr_gt[0]
    wld_coords[1] = arr_img_coords[1] * arr_gt[5] + arr_gt[3]
    # array of shape (1 x n) which is basically True if that point has a dtm value and False otherwise
    arr_has_z = np.zeros((1, arr_img_coords.shape[1]))
    # then now for the wld_coords, find their corresponding pixel and line values within the dtm array
    dtm_img_coords = np.zeros_like(arr_img_coords)
    dtm_img_coords[0] = (wld_coords[0] - dtm_gt[0]) / dtm_gt[1]
    dtm_img_coords[1] = (wld_coords[1] - dtm_gt[3]) / dtm_gt[5]
    x_within_range = np.logical_and(dtm_img_coords[0] >= 0, dtm_img_coords[0] < dtm.shape[1])
    y_within_range = np.logical_and(dtm_img_coords[1] >= 0, dtm_img_coords[1] < dtm.shape[0])
    arr_has_z = np.logical_and(x_within_range, y_within_range)
    img_coords_without_z = (arr_img_coords[:, np.where(~arr_has_z)[0]]).astype(np.int32)
    lines_without_z = img_coords_without_z[1]
    pixels_without_z = img_coords_without_z[0]
    arr[lines_without_z, pixels_without_z] = 50

    dtm_img_coords_with_z = (dtm_img_coords[:, np.where(arr_has_z)[0]]).astype(np.int32)
    dtm_img_lines = dtm_img_coords_with_z[1]
    dtm_img_pixels = dtm_img_coords_with_z[0]
    z_values = dtm[dtm_img_lines, dtm_img_pixels]

    img_coords_with_z = (arr_img_coords[:, np.where(arr_has_z)[0]]).astype(np.int32)
    lines_with_z = img_coords_with_z[1]
    pixels_with_z = img_coords_with_z[0]

    min_z = np.amin(z_values)
    max_z = np.amax(z_values)
    z_values = ((z_values - min_z) / (max_z - min_z) * 155 + 100).astype(np.uint8)

    arr[lines_with_z, pixels_with_z] = z_values

    return arr.astype(np.uint8)


def extract_height(arr, src_extent, dtm_path, src_epsg, src_gt):
    """
    Given the input array which is the mask of roads,
    as well as the extent of the array in the form [ulx, uly, lrx, lry],
    query the height of each HIGH pixel above sea level using the dtm
    """
    dtm_ds = gdal.Open(dtm_path)
    if dtm_ds is None:
        print("Invalid dtm file")
        sys.exit(1)
    # get the crs of the dtm and the src
    dtm_crs = Proj(dtm_ds.GetProjection()).crs
    dtm_epsg = dtm_crs.to_epsg()
    src_crs = CRS.from_epsg(src_epsg)
    ulx, uly, lrx, lry = src_extent
    _, xres, _, _, _, yres = src_gt
    # if the dtm crs is projected but the geotransform is in degrees
    if dtm_crs.is_projected and not src_crs.is_projected:
        # then, we need to convert the geotransform into the dtm crs
        # NOTE: in future, optimise by combining these two together
        ulx, uly = transform_coordinate(ulx, uly, 4326, dtm_epsg)
        lrx, lry = transform_coordinate(lrx, lry, 4326, dtm_epsg)
        xres *= 111139
        yres *= 111139
    # else if the dtm crs is geographical but the geotransform is in metres
    elif dtm_crs.is_geographic and src_crs.is_projected:
        # then transform the geotransform into 4326
        ulx, uly = transform_coordinate(ulx, uly, src_epsg, 4326)
        lrx, lry = transform_coordinate(lrx, lry, src_epsg, 4326)
        xres /= 111139
        yres /= 111139
    
    # now that both the src image and the dtm are in the same crs, create extents for both of them
    # in the form of shapely Polygon objects so that can check for intersection
    src_extent = [[ulx, uly], [lrx, uly], [lrx, lry], [ulx, lry]]
    dtm_gt = dtm_ds.GetGeoTransform()
    dtm_ulx, dtm_xres, _, dtm_uly, _, dtm_yres = dtm_gt
    dtm_lrx = dtm_ulx + dtm_xres * dtm_ds.RasterXSize
    dtm_lry = dtm_uly + dtm_yres * dtm_ds.RasterYSize
    dtm_extent = [[dtm_ulx, dtm_uly], [dtm_lrx, dtm_uly], [dtm_lrx, dtm_lry], [dtm_ulx, dtm_lry]]
    src_extent = Polygon(src_extent)
    dtm_extent = Polygon(dtm_extent)

    # then crop the dtm to read into a smaller array (save time)
    cropped_dtm, cropped_dtm_gt = crop_dtm(dtm_ds, src_extent, dtm_extent, dtm_gt)

    # update src_gt to also be the updated gt after reprojecting CRS
    src_gt = (ulx, xres, 0.0, uly, 0.0, yres)
    # then finally pass into the query_dtm function
    arr = query_dtm(arr, src_gt, cropped_dtm, cropped_dtm_gt)

    return arr


def crop_dtm(dtm_ds, src_extent, dtm_extent, dtm_gt):
    """
    Given a src_extent, dtm_extent and a dtm_ds,
    crop the dtm_ds to just the intersecting area with a small buffer and ReadAsArray
    NOTE: src_extent and dtm_extent both follow the same crs as dtm_gt
    """
    intersection = src_extent.intersection(dtm_extent)
    intersection_wkt = intersection.wkt
    pieces = intersection_wkt.split(",")
    xs, ys = [], []
    for piece in pieces:
        x, y = map(float, re.findall("[-.0-9]+", piece))
        xs.append(x)
        ys.append(y)
    # get the min and max x and y coordinates of intersection (real-world coords)
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    dtm_ulx, dtm_xres, _, dtm_uly, _, dtm_yres = dtm_gt

    # from the geotransform of the dtm, convert the coordinates into pixel and line
    pixel_min = (xmin - dtm_ulx) / dtm_xres
    pixel_max = (xmax - dtm_ulx) / dtm_xres
    line_min = (ymax - dtm_uly) / dtm_yres
    line_max = (ymin - dtm_uly) / dtm_yres

    # but subtract by a 50 pixel buffer just to ensure that the area being cropped by the dtm
    # is larger than necessary. Clip to ensure within range.
    pixel_min = max(pixel_min - 50, 0)
    pixel_max = min(pixel_max + 50, dtm_ds.RasterXSize - 1)
    line_min = min(line_min - 50, 0)
    line_max = min(line_max + 50, dtm_ds.RasterYSize - 1)

    # now that it should be within range, get the new ulx and uly of the cropped dtm
    ulx = dtm_ulx + pixel_min * dtm_xres
    uly = dtm_uly + line_min * dtm_yres
    # hence create the geotransform of the cropped dtm
    cropped_dtm_gt = (ulx, dtm_xres, 0.0, uly, 0.0, dtm_yres)

    # finally, crop and return both the cropped array and its correpsonding geotransform
    band = dtm_ds.GetRasterBand(1)
    dtm = band.ReadAsArray(int(pixel_min), int(line_min), int(pixel_max - pixel_min + 1), int(line_max - line_min + 1))

    return (dtm, cropped_dtm_gt)


def edge_detect(bgr_arr):
    # convert to greyscale
    grey_arr = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2GRAY)
    # apply gaussian blur filter of kernel size 7x7
    grey_arr = cv2.GaussianBlur(grey_arr, (7, 7), 0)
    # run canny edge detection
    edges = cv2.Canny(grey_arr, 100, 150)
    _, thresh = cv2.threshold(edges, 100, 150, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skeleton = cv2.threshold(thresh, 0, 1, cv2.THRESH_BINARY)[1]
    skeleton = (255 * skeletonize(skeleton)).astype(np.uint8)
    return skeleton


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="Path to GeoTiff image", required=True)
    parser.add_argument("--output_dir", help="Path to output folder", required=True)
    parser.add_argument("--dtm_path", help="path to dtm file", required=True)
    args = parser.parse_args()

    run_pipeline(args.filepath, args.output_dir, args.dtm_path)