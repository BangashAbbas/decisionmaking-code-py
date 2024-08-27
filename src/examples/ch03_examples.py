
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from google.colab import files
uploaded = files.upload()

import io

df = pd.read_csv(io.StringIO(uploaded['telemedicine_use.csv'].decode('utf-8')))

df.head()

dataset, info = tfds.load('imdb_reviews', with_info = True,
                                  as_supervised = True)

train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset.element_spec

for review, label in train_dataset.take(1):
    print(review.numpy())
    print()
    print(label.numpy())

BUFFER_SIZE = 500
BATCH_SIZE = 32

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for example, label in train_dataset.take(1):
    print('texts: ', example.numpy()[:3])
    print()
    print('labels: ', label.numpy()[:3])

# prompt: generate histogram

import matplotlib.pyplot as plt
df.hist(figsize=(15,10))
plt.show()

# prompt: genarte heatmap

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()

# prompt: genrate line graphs

import matplotlib.pyplot as plt
df.plot(kind='line',figsize=(15,5))
plt.title('Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
