# Project TDLOG : MAPAGRAM

Project undertaken during the software development techniques course at Ecole des Ponts, dispensed by Xavier CLERC.

This project aims to offer a website where the user can generate a heightmap with chosen conditions.
This elevation map is generated thanks to a Generative Adversarial Network (GAN) using Convolutional Neural Network (CNN) architecture.

## Built With
*  HTML, CSS, Javascript
* Python > 3.6
* [FLASK > 1.0.2](https://www.palletsprojects.com/p/flask/) - web application framework
* [Spyder](https://www.anaconda.com/distribution/) - Python development environment opened with Anaconda platform
* [PyCharm](https://www.jetbrains.com/fr-fr/pycharm/) - Python IDE
* [ATOM](https://atom.io) - text editor used for web application
* [Pylint](https://www.pylint.org/#install) - tool for checking Python code using PEP 8 recommendations


## Getting Started

* Download the project folder.

* Install [Python](https://www.python.org/downloads/) 3.6 or above. 

* Install dependencies with:
```
$ pip install -r requirements.txt
```
* Go to TDLOG project folder
```
$ cd path_project
```
* Run the project on terminal :
```
$ python app.py
```
* Generate your own elevation maps by visiting  http://localhost:5000/home on a Chrome navigator

## Deployment

If you want to explore the code and modify it, we recommand using PyCharm or Spyder development environments. See "Built with" section for links.

## Training your networks
From now on, the top level will be src.
```
$ cd src/
```

### Training the Discriminator alone
#### Create real images data_set
Run the script real_images.py, specifying NB_TIF_TAKEN and WANTED_DIM_IMAGE if you want to.
```
$ python real_images.py
```

#### Create fake images data_set
Run the script fake_images_fractal.py, specifying WANTED_DIM_IMAGE if you want to (it should be equal to the previous one),
to generate images with the perlin noise algorithm.
```
$ python fake_images_fractal.py
```

Run the script fake_images_generator.py, specifying NB_IMAGE_GENERATED and NB_CORE if you want to,
and GENERATOR_ID to use a specific generator training, to generate images with the generator.
```
$ python fake_images_fractal.py
```

Finally, for the actual training, run train_discriminator.py, specifying TRAINING ID,
SIZE_BATCH and NB_EPOCH. It will load the previously generated data_sets, so you can
train on them multiple times without having to recreate them.
```
$ python train_discriminator
```
 This will create a new training state at networks_data/discriminator/<TRAINING_ID>.pth.


### Training the Generator alone
Run the script train_generator.py specifying BATCH_SIZE, NUM_EPOCHS if you want to.
Specify GENERATOR_ID, it is the number of the previous training which you have added one.
Specify DISCRIMINATOR_ID, the last version of the discriminator.
```
$ python train_generator.py
```
This will create a new training state at networks_data/generator/<GENERATOR_ID>.pth.


The previous scripts allow you to train both networks on their own (typically useful
if you have an initial training with fractal noise for the discriminator) but are in general
slower than the following method and are thus deprecated.

### Training the Discriminator and the Generator at the same time
Run the script train.py, specifying BATCH_SIZE, NB_BATCH and IMAGE_SIZE if you want to.
BATCH_SIZE should not be to high (lower than 50).
Specify GENERATOR_ID and DISCRIMINATOR_ID. 
It is the number of the previous training to which you have added one.
```
$ python train.py
```
This will create new training states at networks_data/discriminator/<GENERATOR_ID>.pth
and networks/generator/<GENERATOR_ID>.pth.

## Version

This project is finished. The project lasted for 3 months, from 4th November 2019 to 27th January 2020.

## Authors

* **Alex FAUDUET** - major contribution: *GAN*
* **Auriane RIOU** - major contribution: *Javascript*
* **Candice VAN DEN BERGH** - major contribution: *Flask*
* **Théo VINCENT** - major contribution: *GAN*

## Tutor

* **Clémentine FOURRIER** - [INRIA](https://team.inria.fr/almanach/fr/team-members/)
