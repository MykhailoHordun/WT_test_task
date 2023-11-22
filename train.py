from model import *
from utils import parse_image, normalize, resize

# Dataset loading
data_path = "D:/TestData/images/*" # Replace with the path for the training dataset
dataset = tf.data.Dataset.list_files(data_path, seed=42) 

# Dataset preparing
dataset = dataset.map(parse_image)
dataset = dataset.map(normalize)
dataset = dataset.map(resize)

# Train/Val split
train,val = dataset.take(int(54208*0.95)), dataset.skip(int(54208*0.95)) 
train = train.batch(64)
val = val.batch(64)

model = unet_model() # creating a U-net model

model.fit(                      # model training
        train,
        epochs=1,
        validation_data=val)

model.save('unet.h5') # weights saving 