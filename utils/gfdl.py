### Collection of function for working with GFDL machines
# e.g. retrieving data etc.

def open_frompp(pp,ppname,out,local,time,var,get_static=True):
    _,paths = get_pathspp(pp,ppname,out,local,time,var,get_static=get_static)
    return xr.open_mfdataset(paths,use_cftime=True)

def get_pathspp(pp,ppname,out,local,time,var,get_static=True):
    filename = ".".join([ppname,time,var,'nc'])
    path = "/".join([pp,ppname,out,local,filename])
    paths = glob.glob(path)
    if get_static:
        static = ".".join([ppname,'static','nc'])
        paths.append("/".join([pp,ppname,static]))
    return path,paths

def issue_dmget(paths,wait=True):
    cmd = "dmget "+' '.join(paths)
    if not wait:
        cmp.append(" &")
    os.system(cmd)