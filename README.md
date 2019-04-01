# coral_reef

Repository for the coral reef image segmentation challenge of ImageCLEF 2019

link to the challenge: https://www.crowdai.org/challenges/imageclef-2019-coral-pixel-wise-parsing


installation:

#clone this repo 
	```
	git clone git@tungsten.filament.uk.com:RD/coral_reef.git
	cd coral_reef
	```

#clone git repository for deeplabv3+:
	```	
	mkdir src
	cd src
	git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
	cd ..
	```

# create python environment (e.g. with conda) and activate it
	
	```
	conda create --name coral_reef python==3.6.7
	source activate coral_reef
	```

# install requirements
	```
	pip install --upgrade pip
	pip install -r requirements.txt
	```
