# 변이형 오토인코더 훈련 - 얼굴 데이터셋
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.VAE import VariationalAutoencoder
import numpy as np
from glob import glob

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


# run params
SECTION = 'vae'
RUN_ID = '0001'
DATA_NAME = 'faces'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode = 'build'
# mode = 'load'

DATA_FOLDER = 'data/celeb/img_align_celeba/'

# 데이터 적재
INPUT_DIM = (128, 128, 3)
BATCH_SIZE = 32

filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
NUM_IMAGES = len(filenames)
# print(filenames, '\n', NUM_IMAGES)

# ImageDataGenerator : 실시간 데이터 증가로 텐서 이미지 데이터의 배치를 생성. rescale : 배율 조정
data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(
    DATA_FOLDER, target_size=INPUT_DIM[:2],  # RGB이기 때문에 채널의 수를 바꾼다.
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='input',
    subset='training')

# 모델 만들기
vae = VariationalAutoencoder(
    input_dim=INPUT_DIM,
    encoder_conv_filters=[32, 64, 64, 64],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[2, 2, 2, 2],
    decoder_conv_t_filters=[64, 64, 32, 3],
    decoder_conv_t_kernel_size=[3, 3, 3, 3],
    decoder_conv_t_strides=[2, 2, 2, 2],
    z_dim=200,  # 사용할 잠재 공간수는 200개 => 얼굴은 숫자 이미지보다 훨씬 복잡하기 때문에 상세 정보를 충분히 인코딩 하기 위해...
    use_batch_norm=True,
    use_dropout=True)

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

vae.encoder.summary()
vae.decoder.summary()

# Train Model
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 10000  # 재구성 손실 가중치 파라미터.
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

vae.train_with_generator(  # 폴더에 있는 모든 이미지를 미리 메모리에 로딩하지 않고 python_generator를 사용해 VAE애 주입. => 이 VAE는 배치로 훈련하기 때문에 사전에 이미지를 메모리에 모두 로드할 필요가 없다.
    data_flow,
    epochs=EPOCHS,
    steps_per_epoch=NUM_IMAGES / BATCH_SIZE,
    run_folder=RUN_FOLDER,
    print_every_n_batches=PRINT_EVERY_N_BATCHES,
    initial_epoch=INITIAL_EPOCH
)
