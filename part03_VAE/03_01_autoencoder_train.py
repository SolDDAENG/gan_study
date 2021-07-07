# Auto Encoder
# 데이터에 인코딩 된 표현을 학습한 다음, 학습 된 인코딩 표현에서 입력 데이터를 (가능한한 가깝게) 생성하는 것이 목표
# 오토 인코더의 출력 => 입력에 대한 예측
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.loaders import load_mnist
from models.AE import Autoencoder

# 매개변수 설정
# 실행 매개변수
SECTION = 'vae'
RUN_ID = '0001'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

MODE = 'build'  #'load' #

# 데이터 적재
(x_train, y_train), (x_test, y_test) = load_mnist()

# 신경망 구조 정의
AE = Autoencoder(
    input_dim=(28, 28, 1),
    encoder_conv_filters=[32, 64, 64, 64],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[1, 2, 2, 1],
    decoder_conv_t_filters=[64, 64, 32, 1],
    decoder_conv_t_kernel_size=[3, 3, 3, 3],
    decoder_conv_t_strides=[1, 2, 2, 1],
    z_dim=2
)

if MODE == 'build':
    AE.save(RUN_FOLDER)
else:
    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

AE.encoder.summary()
AE.decoder.summary()

# 디코더는 합성곱 층을 제외하면 인코더의 반대.
# 그렇다고 디코더가 인코더와 완전한 반대 구조를 가질 필요는 없다.
# 디코더에 있는 마지막 층의 출력이 인코더의 입력과 크기가 같다면 어떤 구조도 가능하다. (손실 함수가 픽셀 단위로 비교하기 때문)

# 오토인코더 훈련
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
INITIAL_EPOCH = 0

AE.compile(LEARNING_RATE)

AE.train(
    x_train[:1000],
    batch_size=BATCH_SIZE,
    epochs=200,
    run_folder=RUN_FOLDER,
    initial_epoch=INITIAL_EPOCH
)