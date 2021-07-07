# 1. load data

import numpy as np
import tensorflow.keras.optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print(x_train[54, 12, 13, 1])  # 인덱스54의 이미지에서 (12, 13) 위치에 해당하는 픽셀의 초록색 채널 값.

# 2. architecture
# (1) Sequential model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# model = Sequential([
#                     Flatten(input_shape=(32, 32, 3)),
#                     Dense(200, activation='relu'),
#                     Dense(150, activation='relu'),
#                     Dense(10, activation='softmax')
# ])

# (2) functional api
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(32, 32, 3))

x = Flatten()(input_layer)  # Flatten 벡터의 길이 => 3,072 = 32 x 32 x 3

x = Dense(200, activation='relu')(x)
x = Dense(150, activation='relu')(x)

output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(input_layer, output_layer)

model.summary()

# Train
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

# Analysis
model.evaluate(x_test, y_test)
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

preds = model.predict(x_test)  # [10000, 10]
preds_single = CLASSES[np.argmax(preds, axis=-1)]  # axis = -1 : 마지막 차원으로 배열 압축 => [10000, 1]
actual_single = CLASSES[np.argmax(y_test, axis=-1)]

import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]

    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)
plt.show()