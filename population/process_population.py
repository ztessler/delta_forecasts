import os
import geopandas
import rasterio
import cartopy.crs as ccrs
from rasterstats import zonal_stats

def delta_population_stats(env, target, source):
    year = env['year']
    deltas = geopandas.GeoDataFrame.from_file(str(source[0]))
    deltas = deltas.set_index('Delta')
    popfile = rasterio.open(str(source[1:]))

    laea = ccrs.LambertAzimuthalEqualArea()
    deltas_laea = deltas.to_crs(laea.proj4_params)


    popfile.close()
    return 0



