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