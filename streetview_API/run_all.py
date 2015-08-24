import collection
reload(collection)
import shutil,os
from get_streetview import getview

# SW = [37.785087, -122.423623]
# NE = [37.811130, -122.382081]
rawdatafile = 'uniform_latlng_200_20m1r.csv'
# rawdatafile = 'Main_st_Buisinesses.csv'
# rawdatafile = 'Registered_Business_Map.csv'
where2store = './photodb_MainST200_20m1r/'

dmin = 0.04
#smal area including few blocks around galvanize
# NE = [37.789679, -122.395617]
# SW = [37.781750, -122.398989]


# NE = [37.800,-122.400]
# SW = [37.775,-122.425]
#larger area including part of park
# NE = [37.796259, -122.395532]
# # NE = [37.761259, -122.459532]

# SW = [37.756640, -122.461450]

#1k*1k
# SW = [37.78, -122.41]
# NE = [37.79, -122.40]

#northeast of SF
# SW = [37.755, -122.45]
# NE = [37.81, -122.38]

#box including everithing
SW = [37.7, -122.5]
NE = [37.9, -122.3]


g = getview(rawdatafile, SW, NE, where2store, ready2serve = True)
g.creatdistinct(dmin,validate= False)


meshsize = [5,5]
picsize = [194,128]

angles = ['F', 'B']#, 'R', 'L']#10,130,250,3]#,130,190,250,310]#40, 130, 220, 310]#45, 90, 135, 180]
# angles2 = [130]#225, 180, 315, 360]

#errsize = [(200x200),(300x300)]
errsize = [1996]#,2966, 3367]
# pitch = 90
for angle in angles:
	if angle == 3:
		pitch = 90
		fov = 120
	else:
		pitch = 15
		fov = 120
	g.query(meshsize,picsize,angle,errsize,1, fov, pitch,full =False)



targetpath = '../codeDL/'+where2store[2:]

if os.path.exists(targetpath):
	shutil.rmtree(targetpath)

shutil.copytree(where2store, targetpath)

