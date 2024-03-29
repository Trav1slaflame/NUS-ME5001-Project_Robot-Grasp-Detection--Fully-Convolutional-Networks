# README

# NUS-ME5001-Project_Robot-Grasp-Detection--Fully-Convolutional-Networks

(Please bear in mind that you need to read and adapt to your needs some parts of the code. Feel free to open an issue if you need help. I will try to update README and comment on the code.)

This implementation is modified mainly based on the algorithm from Douglas Morrison, Peter Corke, and Jürgen Leitner described in [https://arxiv.org/abs/1412.3128](https://arxiv.org/abs/1804.05172).

In this repo, we explore more about the application of fully convolutional networks in robotic grasp detection. This kind of method tries to solve the robotic grasp problem utilizing some ideas from segmentation tasks and can predict pixel-level robotic grasp candidates for objects. By using this method, we can predict multiple grasp candidates for a single object and can make grasp predictions for multiple objects simultaneously.

The images used to train the network model are from **[Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)**.

# **Installation**

This code was developed with Python 3.6 on Ubuntu 18.04. Python requirements can be installed by:

```bash
pip install -r requirements.txt
```

# **Datasets**

Currently, only the **[Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)** is supported.

## **Cornell Grasping Dataset Preparation**

1. Download and extract **[Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)**.
2. Place the data to make the data folder like:

```bash
${NUS-ME5001-Project_Robot-Grasp-Detection--Fully-Convolutional-Networks}
|-- data
`-- |-- cornell
    `-- |-- 01
        |   |-- pcd0100.txt
        |   |-- pcd0100cneg.txt
        |   |-- pcd0100cpos.txt
        |   |-- pcd0100d.tiff
        |   |-- pcd0100r.png
	|   ......
	|-- 02
	|-- 03
	|-- 04
	|-- 05
	|-- 06
	|-- 07
	|-- 08
	|-- 09
	|-- 10
        `-- backgrounds
            |-- pcdb0002r.png
            ......
```

# **Training**

Training is done by the `train_ggcnn.py` script. Some of the parameters that need to be adjusted during the training process have been sorted into script `train_ggcnn.py`. You can directly change the value of each corresponding parameter in script `train_ggcnn.py`. 

In script `cornell_data.py` (starting from line 31) and script `image.py` (starting from line 206), keep the code for training and comment the code for testing.

Some basic examples:

```bash
# Train network model on Cornell Dataset
python train.py --description training_example --dataset cornell --dataset-path <Path To Dataset> --use-rgb 1 --use-depth 0
```

Trained models are saved in `output/models` by default, with the validation score appended.

# **Evaluation/Visualisation**

Evaluation or visualisation of the trained networks is done using the `eval_ggcnn.py` script. Some of the parameters that need to be adjusted during the evaluation process have been sorted into script `eval_ggcnn.py`. You can directly change the value of each corresponding parameter in script `eval_ggcnn.py`. 

In script `cornell_data.py` (starting from line 35 and picking a set of pics for testing) and script `image.py` (starting from line 209 and using Test Method 1), keep the code for testing and comment the code for training.

Important flags are:

- `--iou-eval` to evaluate using the IoU between grasping rectangles metric.
- `--eval-vis` to plot the network output (predicted grasping rectangles).

For example:

```bash
python eval_resnet50.py --dataset cornell --dataset-path <Path To Dataset> --iou-eval --trained-network <Path to Trained Network> --eval-vis
```