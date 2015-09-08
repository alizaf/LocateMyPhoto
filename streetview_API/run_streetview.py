import get_streetview
# reload(get_streetview)
import shutil,os
from get_streetview import getview

if __name__ == '__main__':
	rawdatafile = 'uniform_latlng_200_20m1r.csv' 	# Preprocessed image map is stored in uniform_latlngxxxx.csv
	where2store = './photodb_MainST200_20m1r/'
	dmin = 0.04	# distance between neigbour images

	#small area (1 km x 1 km) including few blocks around galvanize
	# NE = [37.789679, -122.395617]
	# SW = [37.781750, -122.398989]

	#northeast of SF
	# SW = [37.755, -122.45]
	# NE = [37.81, -122.38]

	# San Francisco
	SW = [37.7, -122.5]
	NE = [37.9, -122.3]

	g = getview(rawdatafile, SW, NE, where2store, ready2serve = True)
	g.creatdistinct(dmin,validate= False)

	# meshsize is used to label images for classification
	meshsize = [5,5]	#size of requested images
	picsize = [194,128]
	angles = ['F', 'B']#, 'R', 'L'] 	# Defines direction of streetview images (North = 0) F: forward, B: Backward

	errsize = [1996]#

	for angle in angles:
		if angle == 3:
			pitch = 90
			fov = 120
		else:
			pitch = 15
			fov = 120
		g.query(meshsize,picsize,angle,errsize,1, fov, pitch,full =False)
	targetpath = '../codeDL/'+where2store[2:] 	#path to save the image files

	if os.path.exists(targetpath):
		shutil.rmtree(targetpath)

	shutil.copytree(where2store, targetpath)

