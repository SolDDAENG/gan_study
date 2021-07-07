# VAE 분석 - 얼굴 데이터셋
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.stats import norm
import pandas as pd
from tensorflow.python.autograph.pyct import transformer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.VAE import VariationalAutoencoder
from utils.loaders import load_model, ImageLabelLoader

SECTION = 'vae'
RUN_ID = '0001'
DATA_NAME = 'faces'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

DATA_FOLDER = 'data/celeb/'
IMGAE_FOLDER = 'data/celeb/img_align_celeba/img_align_celeba/'

# 데이터 적재
INPUT_DIM = (128, 128, 3)
att = pd.read_csv(os.path.join(DATA_FOLDER, 'list_attr_celeba.csv'))
imageLoader = ImageLabelLoader(IMGAE_FOLDER, INPUT_DIM[:2])
# print(att.head())

# 모델 만들기
vae = load_model(VariationalAutoencoder, RUN_FOLDER)

# 얼굴 이미지 재구성
n_to_show = 10

data_flow_generic = imageLoader.build(att, n_to_show)

example_batch = next(data_flow_generic)
example_images = example_batch[0]

z_points = vae.encoder.predict(example_images)

reconst_images = vae.decoder.predict(z_points)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(n_to_show):
    img = example_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i + 1)
    sub.axis('off')
    sub.imshow(img)

for i in range(n_to_show):
    img = reconst_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i + 1 + n_to_show)
    sub.axis('off')
    sub.imshow(img)

# 잠재 공간 분포
z_test = vae.encoder.predict_generator(data_flow_generic, steps=20, verbose=1)

x = np.linspace(-3, 3, 100)

fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(50):
    ax = fig.add_subplot(5, 10, i+1)
    ax.hist(z_test[:, i], density=True, bins=20)
    ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x, norm.pdf(x))  # pdf : 확률 밀도 함수
# plt.show()

# 새로 생성한 얼굴
n_to_show = 30

znew = np.random.normal(size=(n_to_show, vae.z_dim))  # 표준 정규 분포에서 200개의 차원을 가진 30개의 포인트를 샘플링

reconst = vae.decoder.predict(np.array(znew))  # 디코더에 전달

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(n_to_show):
    ax = fig.add_subplot(3, 10, i+1)
    ax.imshow(reconst[i, :, :, :])  # 128 x 128 x 3 크기의 결과 이미지 출력
    ax.axis('off')
# plt.show()

# 잠재 공간상의 계산
# 이미지를 저차원 공간으로 매핑하면 잠재 공산의 벡터에 대해 연산을 수행할 수 있는 장점이 있다.
# ex) 우는 얼굴 => 웃는 얼굴
def get_vector_from_label(label, batch_size):

    data_flow_label = imageLoader.build(att, batch_size, label = label)

    origin = np.zeros(shape = vae.z_dim, dtype = 'float32')
    current_sum_POS = np.zeros(shape = vae.z_dim, dtype = 'float32')
    current_n_POS = 0
    current_mean_POS = np.zeros(shape = vae.z_dim, dtype = 'float32')

    current_sum_NEG = np.zeros(shape = vae.z_dim, dtype = 'float32')
    current_n_NEG = 0
    current_mean_NEG = np.zeros(shape = vae.z_dim, dtype = 'float32')

    current_vector = np.zeros(shape = vae.z_dim, dtype = 'float32')
    current_dist = 0

    print('label: ' + label)
    print('images : POS move : NEG move :distance : 𝛥 distance')
    while(current_n_POS < 10000):

        batch = next(data_flow_label)  # next : 반복할 수 있을 때는 해당 값을 출력하고, 반복이 끝났을 때는 기본값을 출력
        im = batch[0]
        attribute = batch[1]

        z = vae.encoder.predict(np.array(im))

        z_POS = z[attribute==1]
        z_NEG = z[attribute==-1]

        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis = 0)
            current_n_POS += len(z_POS)
            new_mean_POS = current_sum_POS / current_n_POS
            movement_POS = np.linalg.norm(new_mean_POS-current_mean_POS)

        if len(z_NEG) > 0: 
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis = 0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_NEG
            movement_NEG = np.linalg.norm(new_mean_NEG-current_mean_NEG)

        current_vector = new_mean_POS-new_mean_NEG
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        print(str(current_n_POS)
              + '    : ' + str(np.round(movement_POS, 3))
              + '    : ' + str(np.round(movement_NEG, 3))
              + '    : ' + str(np.round(new_dist, 3))
              + '    : ' + str(np.round(dist_change, 3)))

        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        if np.sum([movement_POS, movement_NEG]) < 0.08:
            current_vector = current_vector / current_dist
            print('Found the ' + label + ' vector')
            break

    return current_vector   

def add_vector_to_images(feature_vec):

    n_to_show = 5
    factors = [-4,-3,-2,-1,0,1,2,3,4]

    example_batch = next(data_flow_generic)
    example_images = example_batch[0]
    example_labels = example_batch[1]

    z_points = vae.encoder.predict(example_images)

    fig = plt.figure(figsize=(18, 10))

    counter = 1

    for i in range(n_to_show):

        img = example_images[i].squeeze()
        sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
        sub.axis('off')        
        sub.imshow(img)

        counter += 1

        for factor in factors:
            # z_new = z + alpha * feature_vector : 잠개 공간에서 벡터연산 수행
            # ex) 인코딩 된 평균 위치에서 smiling 속성이 없는 이미지가 인코딩된 평균 위치를 빼면 웃음이 없는 곳에서 웃음이 있는 곳으로 향하는 벡터를 얻을 수 있다.
            changed_z_point = z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis('off')
            sub.imshow(img)

            counter += 1
    
    plt.show()


BATCH_SIZE = 500
attractive_vec = get_vector_from_label('Attractive', BATCH_SIZE)
mouth_open_vec = get_vector_from_label('Mouth_Slightly_Open', BATCH_SIZE)
smiling_vec = get_vector_from_label('Smiling', BATCH_SIZE)
lipstick_vec = get_vector_from_label('Wearing_Lipstick', BATCH_SIZE)
young_vec = get_vector_from_label('High_Cheekbones', BATCH_SIZE)
male_vec = get_vector_from_label('Male', BATCH_SIZE)

eyeglasses_vec = get_vector_from_label('Eyeglasses', BATCH_SIZE)
blond_vec = get_vector_from_label('Blond_Hair', BATCH_SIZE)

print('Attractive Vector')
add_vector_to_images(attractive_vec)

print('Mouth Open Vector')
add_vector_to_images(mouth_open_vec)

print('Smiling Vector')
add_vector_to_images(smiling_vec)

print('Lipstick Vector')
add_vector_to_images(lipstick_vec)

print('Young Vector')
add_vector_to_images(young_vec)

print('Male Vector')
add_vector_to_images(male_vec)

print('Eyeglasses Vector')
add_vector_to_images(eyeglasses_vec)

print('Blond Vector')
add_vector_to_images(blond_vec)

def morph_faces(start_image_file, end_image_file):

    factors = np.arange(0,1,0.1)

    att_specific = att[att['image_id'].isin([start_image_file, end_image_file])]
    att_specific = att_specific.reset_index()
    data_flow_label = imageLoader.build(att_specific, 2)

    example_batch = next(data_flow_label)
    example_images = example_batch[0]
    example_labels = example_batch[1]

    z_points = vae.encoder.predict(example_images)

    fig = plt.figure(figsize=(18, 8))

    counter = 1

    img = example_images[0].squeeze()
    sub = fig.add_subplot(1, len(factors)+2, counter)
    sub.axis('off')        
    sub.imshow(img)

    counter+=1

    for factor in factors:

        changed_z_point = z_points[0] * (1-factor) + z_points[1]  * factor
        changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

        img = changed_image.squeeze()
        sub = fig.add_subplot(1, len(factors)+2, counter)
        sub.axis('off')
        sub.imshow(img)

        counter += 1

    img = example_images[1].squeeze()
    sub = fig.add_subplot(1, len(factors)+2, counter)
    sub.axis('off')        
    sub.imshow(img)

    plt.show()


start_image_file = '000238.jpg' 
end_image_file = '000193.jpg' #glasses
morph_faces(start_image_file, end_image_file)

start_image_file = '000112.jpg'
end_image_file = '000258.jpg'
morph_faces(start_image_file, end_image_file)

start_image_file = '000230.jpg'
end_image_file = '000712.jpg'
morph_faces(start_image_file, end_image_file)