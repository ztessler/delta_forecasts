import os
import numpy as np
import pandas
import geopandas
import rasterio
from rasterio.features import rasterize
from affine import Affine
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.io.srtm import SRTM1Source, SRTM3Source, SRTMDownloader

def delta_srtm_composite(env, target, source):
    dname = env['delta']
    resolution=env['resolution']
    delta = geopandas.read_file(str(source[0]))
    minlon, minlat, maxlon, maxlat = delta.bounds.values[0]
    extent = (minlon, maxlon, minlat, maxlat)

    local_path_template = os.path.join(os.environ['HOME'], 'data', 'SRTM{resolution}', dname, '{y}{x}.hgt')
    downloader = SRTMDownloader(local_path_template)

    if resolution == 3:
        SRTMSource = SRTM3Source
    elif resolution == 1:
        SRTMSource = SRTM1Source
    else:
        raise ValueError('resolution must be 1 or 3')

    srtm = SRTMSource(downloader=downloader,
                       max_nx=(int(np.floor(maxlon))-int(np.floor(minlon))+1),
                       max_ny=(int(np.floor(maxlat))-int(np.floor(minlat))+1))
    raster = srtm.fetch_raster(ccrs.PlateCarree(), extent, resolution)
    image, extent = raster[0]
    pix = 1./60/60*resolution
    affine = Affine(pix, 0, extent[0],
                    0, -pix, extent[3])

    with rasterio.open(
            str(target[0]), 'w', driver='GTiff',
            width=image.shape[1], height=image.shape[0],
            crs={'init':'epsg:4326'}, transform=affine,
            count=1, dtype=image.dtype) as dst:
        dst.write(image, 1)

    return 0

def clip_srtm_to_delta(env, target, source):
    delta = geopandas.read_file(str(source[0]))

    nodata = -9999

    with rasterio.open(str(source[1]), 'r') as src:
        kwargs = src.meta.copy()
        del kwargs['transform']

        mask = rasterize(delta.loc[0, 'geometry'], default_value=1, fill=0, out_shape=src.shape, transform=src.affine, dtype=src.dtypes[0])
        window = rasterio.get_data_window(mask, 0)
        image = src.read(1, window=window)
        mask = mask[slice(*window[0]), slice(*window[1])]
        image[mask==0] = nodata

        kwargs.update({
            'height': window[0][1] - window[0][0],
            'width': window[1][1] - window[1][0],
            'affine': src.window_transform(window),
            'nodata': nodata})

        with rasterio.open(str(target[0]), 'w', **kwargs) as dst:
            dst.write(image, 1)

        return 0


def estimate_max_elev(env, source, target):
    with rasterio.open(str(source[0]), 'r') as rast:
        dem = rast.read(1, masked=True)
    percentile = env['percentile']
    elev = np.percentile(dem[~dem.mask], percentile)
    with open(str(target[0]), 'w') as fout:
        fout.write(str(elev)+'\n')
    return 0


def estimate_delta_length(env, source, target):
    delta = geopandas.read_file(str(source[0]))
    center = delta.centroid.squeeze()
    aed = ccrs.AzimuthalEquidistant(central_longitude=center.x, central_latitude=center.y)
    delta_aed = delta.to_crs(aed.proj4_params)
    # boundary = delta_aed.convex_hull.boundary
    circ_radius = np.sqrt(delta_aed.area.squeeze()/np.pi)
    with open(str(target[0]), 'w') as fout:
        fout.write(str(circ_radius)+'\n')
    return 0


def txt_nums_to_df(env, source, target):
    deltas = env['deltas']
    df = pandas.Series(index=deltas)
    for (delta, s) in zip(deltas, source):
        with open(str(s), 'r') as fin:
            df[delta] = float(fin.readline().strip())
    df.to_pickle(str(target[0]))
    return 0


def compute_gradient(env, source, target):
    elev = pandas.read_pickle(str(source[0]))
    length = pandas.read_pickle(str(source[1]))
    geo_scaling = float(env['geo_scaling'])

    gradient = elev / (geo_scaling * length)
    gradient.to_pickle(str(target[0]))
    return 0
