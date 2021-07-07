# WGAN-GP
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import matplotlib.pyplot as plt

from models.WGANGP import WGANGP
from utils.loaders import load_celeb
import pickle

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)

# run params
SECTION = 'gan'
RUN_ID = '0003'
DATA_NAME = 'celeb'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode = 'build' #'load' #

# Load Data
BATCH_SIZE = 64
IMAGE_SIZE = 64

x_train = load_celeb(DATA_NAME, IMAGE_SIZE, BATCH_SIZE)
# print(x_train[0][0][0])
plt.imshow((x_train[0][0][0] + 1) / 2)
plt.show()

# 모델 생성
gan = WGANGP(
    input_dim=(IMAGE_SIZE, IMAGE_SIZE, 3),
    critic_conv_filters=[64, 128, 256, 512],
    critic_conv_kernel_size=[5, 5, 5, 5],
    critic_conv_strides=[2, 2, 2, 2],
    critic_batch_norm_momentum=None,
    critic_activation='leaky_relu',
    critic_dropout_rate=None,
    critic_learning_rate=0.0002,
    generator_initial_dense_layer_size=(4, 4, 512),
    generator_upsample=[1, 1, 1, 1],
    generator_conv_filters=[256, 128, 64, 3],
    generator_conv_kernel_size=[5, 5, 5, 5],
    generator_conv_strides=[2, 2, 2, 2],
    generator_batch_norm_momentum=0.9,
    generator_activation='leaky_relu',
    generator_dropout_rate=None,
    generator_learning_rate=0.0002,
    optimiser='adam',
    grad_weight=10,
    z_dim=100,
    batch_size=BATCH_SIZE
)

if mode == 'build':
    gan.save(RUN_FOLDER)

else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

gan.critic.summary()
gan.generator.summary()

# 모델 훈련
EPOCHS = 6000
PRINT_EVERY_N_BATCHES = 5
N_CRITIC = 5
BATCH_SIZE = 64

gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    n_critic=N_CRITIC,
    using_generator=True
)

fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)  # 비평자(진짜와 가짜를 평균한 이미지)
plt.plot([x[1] for x in gan.d_losses], color='green', linewidth=0.25)  # 비평자(진짜 이미지)
plt.plot([x[2] for x in gan.d_losses], color='red', linewidth=0.25)  # 비평자(가짜 이미지)
plt.plot(gan.g_losses, color='orange', linewidth=0.25)  # 생성자

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.xlim(0, 2000)
plt.ylim(0, 2)

plt.show()