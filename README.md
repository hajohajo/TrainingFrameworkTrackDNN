# TrainingFrameworkTrackDNN
Framework for training deep neural networks to classify reconstructed tracks.

## Generate ntuples for training:

Create a suitable CMSSW environment in lxplus and edit trackingNtuple_cff.py

```
cmsrel CMSSW_10_6_8
cd CMSSW_10_6_8/src
cmsenv
git cms-addpkg Validation/RecoTrack
nano Validation/RecoTrack/python/trackingNtuple_cff.py
```

Set  _includeSeeds=False and _includeTrackingParticles=False in the config file.

```
scram b -j 10
```

Get a skeleton for the reconstruction step from some working flow that suits you. trackingOnly -versions are enough, and save time.

```
runTheMatrix.py -l 10824.1 -j0
```

Customise the process by removing all the qualityCuts made by track classifiers of different iterations. DO NOT MISS THIS STEP! This is slightly dependent on the which version of tracking is used but for phase1, it is enough to add the following lines to the process customization of “step3” in the reconstruction:

```
process.initialStep.qualityCuts = [-100.0, -100.0, -100.0]
process.lowPtQuadStep.qualityCuts = [-100.0, -100.0, -100.0]
process.highPtTripletStep.qualityCuts = [-100.0, -100.0, -100.0]
process.lowPtTripletStep.qualityCuts = [-100.0, -100.0, -100.0]
process.detachedQuadStep.qualityCuts = [-100.0, -100.0, -100.0]
process.detachedTripletStep.qualityCuts = [-100.0, -100.0, -100.0]
process.pixelPairStep.qualityCuts = [-100.0, -100.0, -100.0]
process.mixedTripletStep.qualityCuts = [-100.0, -100.0, -100.0]
process.pixelLessStep.qualityCuts = [-100.0, -100.0, -100.0]
process.tobTecStep.qualityCuts = [-100.0, -100.0, -100.0]
process.jetCoreRegionalStep.qualityCuts = [-100.0, -100.0, -100.0]
```

Also to produce the trackingNtuple, you need to add the lines 

```
from Validation.RecoTrack.customiseTrackingNtuple import customiseTrackingNtuple
process = customiseTrackingNtuple(process)
```

For the files add whichever file you want to produce the candidates from. These files can be found using the CMS Data Aggregation System (https://cmsweb.cern.ch/das/)

Good samples to use QCD_Flat_15to3000 with PU, TTbar with PU, High pT QCD samples and DisplacedSUSY (stoptobottom).

## Running the training
As long as the DatasetLoader.load_into_dataframe() methods point to valid paths for the trackingNtuple files, the should run just by givin the main file to the python interpreter. If you want to control how GPUs are assigned on the machine (TF takes them all by default), gpusetter.py can be edited for your preference. Training hyperparameters and network architecture are collected into config.py in order to have all tunable parameters in the same place. model_freeze.py takes care that the model can be deployed to tensorflow, producing the frozen_graph.pb that needs to be copied to the RecoTracker/FinalTrackSelectors/data folder to be accessible.


### About the machine
A GPU is a must for running these trainings, as it just takes too long on a CPU. I've been using a machine with up to four 1080GTX GPUs (4-5 years old tech) and that gives plenty of computing for this type of application since we're not using large images or such as inputs. Even one is enough, but splitting the training over multiple GPUs gives a non-negligible speedup. This level of hardware gives a turnaround time of roughly 1 hour to make a dense model with 30k-300k converge which is convenient for testing different things. Additionally its good to have some surplus memory available, a training set of O(100 million) tracks ends up taking around 60GB of memory.

### Loading the data
The DatasetLoader (dataset_loader.py) uses uproot (https://uproot.readthedocs.io) to read the .root formatted data into Pandas DataFrames (https://pandas.pydata.org) that are conveniently handled in python code. To speedup the loading it splits the operation into multiple processes based on the max_workers keyword argument so if you have more than one CPU core available, this gives a nice boost to loading the data to memory as the I/O operations are often slow. Loading the dataset of O(100 million) tracks ends up chugging around 60GB of memory even though the accuracy is reduced to float32.

### Balancing the data
The dataset can be unbalanced with respect to how many tracks are available for the different iterations. This matters since the network has to have good performance on all iterations of track reconstruction, and large imbalance in the 'sub-concepts' may lead to some iterations being categorically assigned to either fake or true based on what was seen during the training. helper_functions.py contains a convenience script balance_iterations() to subsample a DataFrame so that each iteration has an equal representation in the training set. Since this will end up dropping a ton of data its better to have plenty in the initial training set. Also it is better to enrich the training sample with processes that produce large numbers of tracks in the less populated iterations (mostly the displaced and jetcore).

### Training configuration
Parameters that are most often adjusted are collected into config.py for convenience. This includes the network architecture and hyperparameters.

### Custom layers
The network implements preprocessing of the input variables in its first layers InputSanitizerLayer and OneHotLayer in network/custom folder. InputSanitizerLayer clips the inputs, takes the absolute values and logarithms, and scales them to the range [-1.0, 1.0] based on the smallest and largest encountered variable. OneHotLayer takes the input integer trk_originalAlgo and embeds it into a one-hot vector. This is the recommended way to treat categorical variables where no explicit metric between the values is established.

### Sample weights
Importance weights can be provided to the different samples, however recent evidence points at subsampling being the way to go instead. Additionally sample weights can easily cause instabilities in the training process.

[1] https://arxiv.org/abs/1812.03372

### ModelManager
The ModelManager is a bit of a catch all class, taking care of initializing the custom layers correctly based on the values encountered in the training set, producing plots, saving the model, (re)initializing it and producing the necessary folders for storing the training outputs such as model weights and plots. Unfortunately that makes it a bit messy as well since it has way more responsibilities than good programming habits would allow.
