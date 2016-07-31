"""
Original Code from sklearn documentation site:
http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from scipy import misc
import sys
import json
from os import environ

n_colors = 8
DEBUG = False

if ('DEBUG' in environ) and (environ['DEBUG'] == 'True'):
    DEBUG = True

if len(sys.argv) < 2:
    print "Too few arguments"
    sys.exit(1)

if len(sys.argv) == 3:
    n_colors = int(sys.argv[2])

imageNpy = misc.imread(sys.argv[1])
imageNpy = np.array(imageNpy, dtype=np.float64) / 255   # Normalization

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(imageNpy.shape)
assert d == 3
image_array = np.reshape(imageNpy, (w * h, d))

if DEBUG:
    print("Fitting model")
    t0 = time()

image_array_sample = shuffle(image_array, random_state=0)[:10000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
# kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)

if DEBUG:
    print("done in %0.3fs." % (time() - t0))

# Get labels for all points
if DEBUG:
    print("Predicting color indices on the full image (k-means)")
    t0 = time()

labels = kmeans.predict(image_array)

if DEBUG:
    print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def compare_colors(centroids):
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie2000

    global n_colors
    colors = list()

    for i in range(len(centroids)):
        [r, g, b] = centroids[i]
        print int(r*255), int(g*255), int(b*255), convert_color(sRGBColor(r, g, b), LabColor)
        colors.append(convert_color(sRGBColor(r, g, b), LabColor))

    for i in range(len(colors)):
        for j in range(len(colors)):
            deltaE = delta_e_cie2000(colors[i], colors[j])
            output = '%.4f\t' % deltaE
            sys.stdout.write(output)
        sys.stdout.write('\n')

def output_json(centroids):
    output_list = list();
    for i in range(len(centroids)):
        [r,g,b] = centroids[i]
        output_list.append( (r, g, b) )        ## tuple
    obj = dict();
    obj["colors"] = output_list
    obj["pathname"] = sys.argv[1]

    json_file = open("output.json", "a")
    json.dump(obj, json_file)
    json_file.write(',\n')
    json_file.close()

def output_report(data):
    import operator
    print "quantized-%s" % sys.argv[1]
    print '"Color"\t"Pixels"\t"Percentage"'
    # s = sorted(color_dict.items(), key=operator.itemgetter(1), reverse=True)
    # for keys,values in s:
    for keys in color_dict:
        values = color_dict[keys]
        print '%s\t%d\t%.2f%%' % (keys, values, float(values) / len(labels) * 100)



# Count the number of pixels assigned to each centroid
freq = [0]*n_colors
for i in range(len(labels)):
    freq[labels[i]] += 1

# Create a dict of the color centroids and
# sort then according to their frequency
color_dict = dict()
for i in range(0, len(kmeans.cluster_centers_)):
    [r, g, b] = kmeans.cluster_centers_[i]
    # color_dict['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))] = freq[i]
    color_dict['#%02x%02x%02x\t%03d %03d %03d' % (int(r*255), int(g*255), int(b*255), int(r*255), int(g*255), int(b*255))] = freq[i]

# compare_colors(kmeans.cluster_centers_)

output_json(kmeans.cluster_centers_)
#output_report(color_dict)
# misc.imsave("quantized-%s" % sys.argv[1], recreate_image(kmeans.cluster_centers_, labels, w, h))



if DEBUG:
    # Display all results, alongside original image
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(imageNpy)

    plt.figure(2)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Quantized image (%d colors, K-Means)' % n_colors)
    plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

    plt.show()
