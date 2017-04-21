import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

import numpy

def compToEllipse(sigma, mu, alpha, color):
	# adapted from
	# http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
        v, w = numpy.linalg.eigh(sigma)
        u = w[0] / numpy.linalg.norm(w[0])
        angle = numpy.arctan2(u[1], u[0])
        angle = 180 * angle / numpy.pi  # convert to degrees
        v = 2. * numpy.sqrt(2.) * numpy.sqrt(v)

	return Ellipse(xy=mu, width=v[0], height=v[1], angle=angle, alpha=alpha, fc = color)

def makeColorMap(numBins):
	# Adapted from
	# http://stackoverflow.com/a/25628397/226484	
	color_norm = colors.Normalize(vmin=0, vmax=numBins-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)
	return map_index_to_rgb_color

def plotGmm(ax, fig, comps, cMap):
	index = 0
	for comp in comps:
		pi = comp["pi"]
		mu = numpy.array(comp["mu"])
		sigma = numpy.array(comp["sigma"]).reshape((2,2))

		compColor = cMap(index)
		ax.scatter( [mu[0]], [mu[1]], s = 8, marker='o', color=compColor)

		alpha = 0.2

		ax.add_artist(compToEllipse(sigma, mu, alpha, compColor))
		index += 1
