from grd_dat import grd_dat
from prp_dat import nth_dat

xmin, xmax, nx = 0, 40, 401
ymin, ymax, ny = 0, 10, 101
tmin, tmax, nt = 0, 10, 101

x, y, t, u, v, p = nth_dat(xmin, xmax, 
            ymin, ymax, 
            tmin, tmax, 
            N_nth = 10)

print(x)
print(y)
print(t)

