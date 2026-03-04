import xarray as xr
from xgcm import Grid
import numpy as np
try:
    from xgcm.autogenerate import generate_grid_ds
except ImportError:
    generate_grid_ds = None

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


def _axis_is_periodic(periodic, axis):
        if isinstance(periodic, bool):
            return periodic
        if periodic is None:
            return False
        if isinstance(periodic, str):
            return periodic == axis
        return axis in periodic


def _wrap_discontinuity(delta, boundary_discontinuity, axis=None):
        if boundary_discontinuity is None:
            return delta

        if isinstance(boundary_discontinuity, dict):
            if axis is None:
                if len(boundary_discontinuity) == 1:
                    boundary_discontinuity = next(iter(boundary_discontinuity.values()))
                else:
                    raise ValueError(
                        "boundary_discontinuity is a dict; axis must be provided to select a value."
                    )
            else:
                boundary_discontinuity = boundary_discontinuity.get(axis)
                if boundary_discontinuity is None:
                    return delta

        return ((delta + boundary_discontinuity / 2.0) % boundary_discontinuity) - (
            boundary_discontinuity / 2.0
        )


def _center_to_left(coord, periodic=False, boundary_discontinuity=None, axis=None):
        dim = coord.dims[0]
        prev = coord.shift({dim: 1})

        if periodic:
            prev = prev.where(prev.notnull(), coord.isel({dim: -1}))
            delta = _wrap_discontinuity(coord - prev, boundary_discontinuity, axis=axis)
            return coord - 0.5 * delta

        edge_delta = coord.isel({dim: 1}) - coord.isel({dim: 0})
        first_prev = coord.isel({dim: 0}) - edge_delta
        prev = prev.where(prev.notnull(), first_prev)
        return 0.5 * (coord + prev)


def _forward_diff(coord, periodic=False, boundary="extrapolate", boundary_discontinuity=None, axis=None):
        dim = coord.dims[0]
        diffs = coord.diff(dim)

        if periodic:
            tail = coord.isel({dim: 0}) - coord.isel({dim: -1})
            tail = _wrap_discontinuity(tail, boundary_discontinuity, axis=axis)
        elif boundary == "nan":
            tail = xr.full_like(coord.isel({dim: -1}), np.nan)
        else:
            tail = coord.isel({dim: -1}) - coord.isel({dim: -2})

        tail = tail.expand_dims({dim: [coord[dim].values[-1]]})
        diffs = xr.concat([diffs, tail], dim=dim)
        return diffs.assign_coords({dim: coord[dim]})

def get_xgcm_horizontal(ds,axes_dims_dict,position=None,periodic=None,boundary_discontinuity=360):
    ''' Generate metrics and grid locations'''

    if generate_grid_ds is None:
        raise ImportError(
            "xgcm.autogenerate.generate_grid_ds is unavailable in this xgcm version. "
            "Use get_xgcm_horizontal_regular for regular lat-lon grids."
        )
    
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
    # This is a hacky replacement because xgcm has removed boundary_discontinuity keyword from diff
    dlonG = xgrid.diff(ds[gridlon], 'X', boundary='fill', fill_value=ds[gridlon][-1]+ds[gridlon][-1]-ds[gridlon][-2])
    dlonC = xgrid.diff(ds[gridlon+'_'+suffix], 'X', boundary='fill', fill_value=ds[gridlon+'_'+suffix][-1]+ds[gridlon+'_'+suffix][-1]-ds[gridlon+'_'+suffix][-2])

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


def get_xgcm_horizontal_regular(ds,axes_dims_dict,periodic=None,boundary_discontinuity=360,suffix="left"):
    """Generate xgcm horizontal coords and metrics from regular 1D lat-lon center coordinates."""

    gridlon = axes_dims_dict["X"]
    gridlat = axes_dims_dict["Y"]
    lon = ds[gridlon]
    lat = ds[gridlat]

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("get_xgcm_horizontal_regular expects 1D center coordinates for X and Y.")
    if lon.sizes[lon.dims[0]] < 2 or lat.sizes[lat.dims[0]] < 2:
        raise ValueError("X and Y coordinates must each contain at least two points.")

    periodic_x = _axis_is_periodic(periodic, "X")
    periodic_y = _axis_is_periodic(periodic, "Y")

    lon_left_name = f"{gridlon}_{suffix}"
    lat_left_name = f"{gridlat}_{suffix}"
    lon_left = _center_to_left(
        lon, periodic=periodic_x, boundary_discontinuity=boundary_discontinuity, axis="X"
    )
    lat_left = _center_to_left(lat, periodic=periodic_y, boundary_discontinuity=None)
    ds = ds.assign_coords({lon_left_name: lon_left, lat_left_name: lat_left})

    dlonG = _forward_diff(
        ds[gridlon], periodic=periodic_x, boundary_discontinuity=boundary_discontinuity, axis="X"
    )
    dlonC = _forward_diff(
        ds[lon_left_name], periodic=periodic_x, boundary_discontinuity=boundary_discontinuity, axis="X"
    )

    y_boundary = "extrapolate" if periodic_y else "nan"
    dlatG = _forward_diff(ds[gridlat], periodic=periodic_y, boundary=y_boundary)
    dlatC = _forward_diff(ds[lat_left_name], periodic=periodic_y, boundary=y_boundary)

    ds["dxG"], ds["dyG"] = _degrees_to_meters(dlonG, dlatG, ds[gridlon], ds[gridlat])
    ds["dxC"], ds["dyC"] = _degrees_to_meters(dlonC, dlatC, ds[gridlon], ds[gridlat])
    ds["rC"] = ds["dxC"] * ds["dyC"]

    coords = {
        "X": {"center": gridlon, suffix: lon_left_name},
        "Y": {"center": gridlat, suffix: lat_left_name},
    }
    metrics = {
        "X": ["dxC", "dxG"],
        "Y": ["dyC", "dyG"],
        ("X", "Y"): ["rC"],
    }
    xgrid = Grid(ds, coords=coords, metrics=metrics, periodic=periodic)

    return ds, xgrid
