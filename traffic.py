import os

import skimage

from skimage import transform ,data

from PIL import Image

import cv2

import matplotlib.pyplot as plt 

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            #images.append(skimage.data.imread(f))
            images.append(skimage.data.imread(f))
            #images.append(Image.open(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "F:\TensorFlow"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\Testing")

images, labels = load_data(train_data_directory)

# Print the `images` format
print(images[0].format)

# Print the `images` mode
print(images[0].mode)

# Print the number of `images`'s elements
print(images[0].size)

# Print the first instance of `images`
print(images[0])

# Print the number of labels
print(len(set(labels)))

im=images[14]
im.show()
im.save("1.ppm")

#set 2 figure to show hist and images or you can only see the last figure
plt.figure(1) 
# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)



# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 3650, 4000]

plt.figure(2)
# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    #print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].size, 
     #                                             images[traffic_signs[i]].min(), 
     #                                             images[traffic_signs[i]].max()))
#plt.show()

plt.figure(3)
# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)

#images28 = [image.resize((28,28),Image.BILINEAR) for image in images]
images28 = [transform.resize(image, (28, 28)) for image in images]
images28[1].show()
# Show the plot in the end of program
plt.show()
