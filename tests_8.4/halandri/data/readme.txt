This is a complete set of the required input data used for the simulation of the Halandri catchment (Athens, Greece), 
using the fully hydrodynamic method throught the in-house FLOW-R2D model (Bellos and Tsakiris, 2016):

Bellos, V. and Tsakiris, G., 2016. A hybrid method for flood simulation in small catchments 
combining hydrodynamic and hydrological techniques. Journal of Hydrology, 540, 331-339.  

The set consists of the following files:

1) dtm.csv

This is the Digital Terrain Model of the catchment. 
The first two columns are the X, Y coordinates according to the Hellenic Geodetic Reference System 1987).
The third column is the surface elevation.
The fourth column indicates the land use, according to the CORINE Land Cover 2000 maps:

Value '1': Coniferous forest
Value '2': Disconinuous urban fabric
Value '3': Transitional woodland-shrub
Value '4': Sclerophyllous vegetation
Value '5': Mineral extraction sites


2) measurements.csv

This is the file with the measurements. 
The first column is the time of the event.
The second coumn is the rainfall height (in mm) as recorded at the meteorological station of Palea Penteli (coordinates: 38.04699, 23.86513)
The third column is the hydrograph mesaured at the outlet of the catchment, as recorded at the hydrometric station of Nea Penteli (coordinates: 38.05333, 23.85125).

All the measurements are recorded in the context of the DEUCALION Project:

http://deucalionproject.itia.ntua.gr/


3) flowr2d.csv

This is the file with the results derived using the FLOW-R2D model.
The first column is the time of the event.
The second column is the results derived from the fully hydrodynamic method throught the FLOW-R2D model and the Kostiakov equation for the infiltration.
The third column is the results derived from the hybrid method, the 15-minute Unit Hydrograph and the Kostiakov equation for the infiltration.
The fourth column is the results derived from the hybrid method, the 15-minute Unit Hydrograph and the Green-Ampt model for the infiltration.

The computational domain has been classified in three friction zones:

Zone A: Coniferous forest
Zone B: Disconinuous urban fabric + Mineral extraction sites
Zone C: Transitional woodland-shrub + Sclerophyllous vegetation

For all the methods, the Manning coefficients are:

Zone A: n=0.070 s/m^(1/3)
Zone B: n=0.020 s/m^(1/3)
Zone C: n=0.025 s/m^(1/3)

The Kostiakov equation has the following form: 

f(t)=a*b*t^(b-1)

where

f is the infiltration rate (cm/h)
t is the time (h)
a parameter equal to 3.584 cm/h^0.584 in this simulation
b parameter equal to 0.584 in this simulation

The Green-Ampt model has the following form:

f(t)=K*(((ш*Ди)/F(t))+1)

where

f is the infiltration rate (cm/h)
t is the time (h)
K is the hydraulic conductivity equal to 1.09 cm/h
ш is the wetting front equal to 2.67 cm
F(t) is the cumulative infiltration (cm/h) calculated by:

F(t)=K*t+ш*Ди*ln(1+(F(t)/(ш*Ди)))

4) dtm_modified.csv

This is the Digital Terrain Model of the catchment. It is the same with the dtm.csv file but a rectangle encloses the real computational domain.
The coordinates of the four vertices of the rectancle are (according to the Hellenic Geodetic Reference System 1987):

1: (486821.5,4211453.0)
2: (486821.5,4214393.0)
3: (490641.5,4214393.0)
4: (490641.5,4211453.0)

The first two columns are the X, Y coordinates according to the Hellenic Geodetic Reference System 1987). 
The third column is the surface elevation. The cells which are not part of the real computational domain takes the value '0.0'.
The fourth column indicates the friction zones based on the simulations implemented in Bellons and Tsakiris (2016).

Value '0': The cell is not part of the real computational domain
Value '1': Zone A
Value '2': Zone B
Value '3': Zone C


