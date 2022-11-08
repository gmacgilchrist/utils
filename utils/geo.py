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
        dx = dlon * np.cos(np.deg2rad(lat)) * distance_1deg_equator
        dy = ((lon * 0) + 1) * dlat * distance_1deg_equator
        return dx, dy

def get_xgcm_horizontal(ds,axes_dims_dict,position=None,periodic=None,boundary_discontinuity=360):
    ''' Generate metrics and grid locations'''
    
    gridlon=axes_dims_dict['X']
    gridlat=axes_dims_dict['Y']
    
    ds = generate_grid_ds(ds, {'X':gridlon,'Y':gridlat},
                          position=position)
    xgrid = Grid(ds, periodic=periodic)

    if position is None:
        suffix = 'left'
    else:
        suffix = position[1]
        
    # Get horizontal distances
    dlonG = xgrid.diff(ds[gridlon], 'X', boundary_discontinuity=boundary_discontinuity)
    dlonC = xgrid.diff(ds[gridlon+'_'+suffix], 'X', boundary_discontinuity=boundary_discontinuity)

    dlatG = xgrid.diff(ds[gridlat], 'Y', boundary='fill', fill_value=np.nan)
    dlatC = xgrid.diff(ds[gridlat+'_'+suffix], 'Y', boundary='fill', fill_value=np.nan)

    ds['dxG'], ds['dyG'] = _degrees_to_meters(dlonG, dlatG, ds[gridlon], ds[gridlat])
    ds['dxC'], ds['dyC'] = _degrees_to_meters(dlonC, dlatC, ds[gridlon], ds[gridlat])
    
    ds['rC']=ds['dxC']*ds['dyC']

    # Regenerate grid
    coords = {
        'X':{'center':gridlon,suffix:gridlon+'_'+suffix},
        'Y':{'center':gridlat,suffix:gridlat+'_'+suffix},
    }
    metrics = {
        'X':['dxC','dxG'],
        'Y':['dyC','dyG'],
        ('X','Y'):['rC']
    }
    xgrid = Grid(ds,coords=coords,metrics=metrics,periodic=periodic)

    return ds,xgrid
