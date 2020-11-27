# coral_reef

Repository for the coral reef image segmentation challenge of ImageCLEF 2019

link to the challenge: https://www.crowdai.org/challenges/imageclef-2019-coral-pixel-wise-parsing


# Installation:

Clone this repository and `cd imageclef-2019-code`

Clone git repository for deeplabv3+:

```bash	
mkdir src
cd src
git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
cd ..
```

Create python environment (e.g. with conda) and activate it

```bash
virtualenv -p python3.7 env/
. env/bin/activate
```

```bash
pip install torch torchvision cudatoolkit==9.0 #For gpu
```

Install requirements
```bash
pip install -r requirements.txt
```

# Usage

First move all the CLEF images to `data/images` and the csv with clef annotations in `data/annotations.csv`. Then create masks with

```bash
python coral_reef/data/create_masks.py
``` 

And subsequently split into train and validation

```bash
python coral_reef/data/data_split.py
```

This will generate `data_train.json` and `data_valid.json` which will be used to train.

	
	~~~~
	conda create --name coral_reef python==3.6.7
	source activate coral_reef
	~~~~
	
## install pytorch
	---- Cuda available, Cuda version 9.0: ----
	~~~~	
	conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
	~~~~
    
    OR
    
	---- Cuda not available: ---
	~~~~	
	conda install pytorch-cpu torchvision-cpu -c pytorch
	~~~~

# install requirements
```bash
	pip install --upgrade pip
	pip install -r requirements.txt
```
