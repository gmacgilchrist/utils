import xarray as xr
import numpy as np
from scipy.signal import argrelextrema

# Simple operations on numpy and xarray arrays

def crossing(da,dim,edge='above'):
    '''
    Return the coordinate along dim where da changes sign.
    Specify whether you wish to return the coordinate above or
    below the reversal.
    '''
    if edge=='above':
        shift = 1
    else:
        shift=-1
    reversal = np.sign(da)+np.sign(da.shift({dim:shift}))
    return da[dim].where(reversal==0,drop=True)

def extrema(da,dim,order=1):
    '''Return the coordinate along dim where there are local
    extrema. order specifies how far out the condition needs
    to be satisfied.'''
    ind = argrelextrema(da.values,np.less,order=order)
    mn = da[dim].isel({dim:ind[0]})
    ind = argrelextrema(da.values,np.greater,order=order)
    mx = da[dim].isel({dim:ind[0]})
    return {'min':mn,'max':mx}

def find_nonzero(da,dim,first=True):
    """ Find the coordinate of a first or last non-zero entry in an array along specified dimension. 
    
    PARAMETERS
        da [xr.DataArray]
        dim [str]
        first [bool]
            If True, return the index/coordinate for the first occurence of a nonzero entry in that 
            dimension (ascending coordinate). Otherwise, return last.
            
    OUPUT
        index
            Index of first/last nonzzero entry
        coord
            Coordinate of first/last nonzero entry
    """
    boolean = (hsnow!=0)
    if first:
        index = boolean.argmax(dim)
    else:
        index = (len(hsnow['lat_bin'])-1)-boolean.sortby('lat_bin',ascending=False).argmax('lat_bin')
    
    coord = hsnow['lat_bin'].isel(lat_bin=index)
    
    return index, coord