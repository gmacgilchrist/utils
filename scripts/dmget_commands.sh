#!/bin/bash

root=/archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_piControl_D/gfdl.ncrc4-intel16-prod-openmp/pp

ppname=ocean_annual_rho2
out=av
local=annual_5yr
time=0[2-5]*
var=*

paths=${root}/${ppname}/${out}/${local}/${ppname}.${time}.${var}.nc

dmget ${paths} &