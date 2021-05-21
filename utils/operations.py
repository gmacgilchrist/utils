import xarray as xr
import numpy as np

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
