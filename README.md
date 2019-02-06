# Triplet Loss Network for Unsupervised Domain Adaptation
Pytorch Implementation of TripLet Loss for Unsupervised Domain Adaptation
## Authors 

* Imad Eddine Ibrahim Bekkouch i.bekkouch@innopolis.university
* Youssef Youssry y.ibrahim@innopolis.university
* Rustam Gafarov r.gafarov@innopolis.ru
* Adil Khan a.khan@innopolis.ru

Institute of Data Science & AI 

Innopolis University

Innopolis

## Getting Started
Please follow the instructions to get an up and running version of our code running on your local machine.
### Prerequisites
Please make sure you have the following installed.

1. Python 3.6
2. Pytorch 1.0
3. termcolor 1.1.0
4. Numpy 1.15+

### Running the code
First step is to unzip the office31 data
```
cd ./data
tar xvzf office31.tar.gz --keep-newer-filesls
```

Run the main file with using any of the following arguments:
```
usage: main.py [-h] [--image_size IMAGE_SIZE] [--num_classes NUM_CLASSES]
               [--train_iters TRAIN_ITERS] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS] [--pre_lr PRE_LR] [--lr1 LR1]
               [--lr2 LR2] [--random_seed RANDOM_SEED]
               [--labeled_target_ratio LABELED_TARGET_RATIO]
               [--validation_source_ratio VALIDATION_SOURCE_RATIO]
               [--mode {train,test}]
               [--source {usps,svhn_extra,mnist,svhn,amazon,webcam,dslr}]
               [--target {usps,svhn_extra,mnist,svhn,amazon,webcam,dslr}]
               [--model_path MODEL_PATH] [--graph_path GRAPH_PATH]
               [--sample_path SAMPLE_PATH] [--mnist_path MNIST_PATH]
               [--usps_path USPS_PATH] [--svhn_path SVHN_PATH]
               [--off_path OFF_PATH] [--log_step LOG_STEP]
               [--iter_dom_adap ITER_DOM_ADAP] [--log_pre LOG_PRE]
               [--verbose VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  --image_size IMAGE_SIZE
  --num_classes NUM_CLASSES
  --train_iters TRAIN_ITERS
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --pre_lr PRE_LR       learning rate for the source domain pre-training
  --lr1 LR1             encoder's and classifier's learning rate for domain
                        adaptation
  --lr2 LR2             discriminator's learning rate
  --random_seed RANDOM_SEED
  --labeled_target_ratio LABELED_TARGET_RATIO
                        ratio of target domain used for labeling
  --validation_source_ratio VALIDATION_SOURCE_RATIO
  --mode {train,test}
  --source {usps,svhn_extra,mnist,svhn,amazon,webcam,dslr}
                        source domain
  --target {usps,svhn_extra,mnist,svhn,amazon,webcam,dslr}
                        target domain
  --model_path MODEL_PATH
  --graph_path GRAPH_PATH
  --sample_path SAMPLE_PATH
  --mnist_path MNIST_PATH
  --usps_path USPS_PATH
  --svhn_path SVHN_PATH
  --off_path OFF_PATH
  --log_step LOG_STEP
  --iter_dom_adap ITER_DOM_ADAP
  --log_pre LOG_PRE     number of iterations to print the losses and accuracy
                        for pre training
  --verbose VERBOSE
```

## Code structure

Quick explanation of the code structure:

    .
    ├── data                      # Data Folder
    │   ├── office31              # Office31 dataset
    │   │   ├──amazon             # Amazon pictures
    │   │   ├──webcam             # Webcam pictures
    │   │   └──dslr               # Digital SLR pictures
    │   ├── mnist                 # MNIST digits dataset
    │   ├── usps                  # USPS digits dataset
    │   └── svhn                  # SVHN digits dataset
    ├── model                     # Code, pictures and saved models
    │   ├── code                  # Source Code
    │   │   ├──data_loader.py     # Data loaders for Office31, mnist, usps and svhn datasets
    │   │   ├──main.py            # Main
    │   │   ├──model.py           # Model definition (Classier, Encoder and Discriminator)
    │   │   └──solver.py          # Domain Adaptation Algorithm definition
    │   ├── graphs                # Accuracy and loss graphs
    │   └── models                # Saved Models
    ├── LICENSE                   # MIT LICENCE
    └── README.md                 # README
    
    
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details