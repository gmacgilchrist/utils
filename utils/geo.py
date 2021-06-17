import xarray as xr
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds
import numpy as np

def _degrees_to_meters(dlon, dlat, lon, lat):
        """Converts lat/lon differentials into distances in meters
        PARAMETERS
        ----------
        dlon : xarray.DataArray longitude differentials
        dlat : xarray.DataArray latitude differentials
        lon  : xarray.DataArray longitude values
        lat  : xarray.DataArray latitude values
        RETURNS
        -------
        dx  : xarray.DataArray distance inferred from dlon
        dy  : xarray.DataArray distance inferred from dlat
        """

        distance_1deg_equator = 111000.0
        dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(lat)) * distance_1deg_equator
        dy = ((lon * 0) + 1) * dlat * distance_1deg_equator
        return dx, dy

def get_xgcm_horizontal(ds,gridlon='lon',gridlat='lat',periodic=None):
    ''' Generate metrics and grid locations'''

    ds = generate_grid_ds(ds, {'X':gridlon,'Y':gridlat})
    xgrid = Grid(ds, periodic=periodic)

    # Get horizontal distances
    dlonG = xgrid.diff(ds[gridlon], 'X', boundary_discontinuity=360)
    dlonC = xgrid.diff(ds[gridlon+'_left'], 'X', boundary_discontinuity=360)

    dlatG = xgrid.diff(ds[gridlat], 'Y', boundary='fill', fill_value=np.nan)
    dlatC = -xgrid.diff(ds[gridlat+'_left'], 'Y', boundary='fill', fill_value=np.nan)

    ds['dxG'], ds['dyG'] = _degrees_to_meters(dlonG, dlatG, ds[gridlon], ds[gridlat])
    ds['dxC'], ds['dyC'] = _degrees_to_meters(dlonC, dlatC, ds[gridlon], ds[gridlat])

    # Regenerate grid
    coords = {
        'X':{'center':gridlon,'left':gridlon+'_left'},
        'Y':{'center':gridlat,'left':gridlat+'_left'},
    }
    metrics = {
        'X':['dxC','dxG'],
        'Y':['dyC','dyG'],
    }
    xgrid = Grid(ds,coords=coords,metrics=metrics,periodic=periodic)

    return ds,xgrid
