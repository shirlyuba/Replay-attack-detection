# Replay-attack-detection
My MIPT Master's work presents method of detection Replay attack on facial biometrics systems. It works with videos and video-streams.
The main idea of the method is described in the following articles (RU):
- Л. Р. Широкова, В. Н. Логинов. Нейросетевой метод детекции видеоизображения лица в видеопотоке системы лицевой биометрии. ТРУДЫ МФТИ. 2020. Том 12, № 4
- Л. Р. Широкова, В. Н. Логинов. Анализ эффективности архитектур нейронных сетей для детекции Replay Attack в системах лицевой биометрии. ТРУДЫ МФТИ. 2021. Том 13, № 1
- Л. Р. Широкова, В. Н. Логинов. Построение системы лицевой биометрии с защитой от Replay attack. ПРОБЛЕМЫ ИНФОРМАЦИОННОЙ БЕЗОПАСНОСТИ СОЦИАЛЬНО-ЭКОНОМИЧЕСКИХ СИСТЕМ, VII Всероссийская с международным участием научно-практическая конференция. Симферополь, 2021




<details open>
<summary>Install</summary>

You should install all dependecies [requirements.txt](https://github.com/shirlyuba/Replay-attack-detection/blob/main/requirements.txt):
```bash
$ git clone https://github.com/shirlyuba/Replay-attack-detection
$ cd Replay-attack-detection
$ pip install -r requirements.txt
```
</details>


<details open>
<summary>Create video</summary>

You can add your video or use script to make video from camera:
```bash
$ python make_video.py --help
usage: make_video.py [-h] -o OUTPUT -n NAME

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        path to output directory of videos
  -n NAME, --name NAME  name of video
```
</details>


<details open>
<summary>Create dataset</summary>

You should create dataset for training:
```bash
$ python create_dataset.py --help
usage: create_dataset.py [-h] [-i INPUT] -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input video
  -o OUTPUT, --output OUTPUT
                        path to output directory
```
</details>


<details open>
<summary>Train model</summary>

Code for training SqueezeNet model on your dataset:
```bash
$ python train.py --help
usage: train.py [-h] -d DATA [-i INIT_WEIGHTS] -o OUT_WEIGHTS -n EPOCHS

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  path to dataset
  -i INIT_WEIGHTS, --init_weights INIT_WEIGHTS
                        path to input weights
  -o OUT_WEIGHTS, --out_weights OUT_WEIGHTS
                        path to output weights
  -n EPOCHS, --epochs EPOCHS
                        number of epochs for training
```
(if INIT_WEIGHTS is not specified, will be used ImageNet weights)
</details>


<details open>
<summary>Demo</summary>

Visualization of model's work:
```bash
$ python liveness_demo.py --help
usage: liveness_demo.py [-h] [-i INPUT] -m MODEL [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input video
  -m MODEL, --model MODEL
                        path to trained model
  -o OUTPUT, --output OUTPUT
                        path to output video
(if INPUT is not specified, will be used camera(0))
(if OUTPUT is specified, visualization will be saved in OUTPUT path)
```
</details>
