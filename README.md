# TrainingFrameworkTrackDNN
Framework for training deep neural networks to classify reconstructed tracks.

## Generate ntuples:

Create a suitable CMSSW environment in lxplus and edit trackingNtuple_cff.py

'''
cmsrel CMSSW_10_6_8
cd CMSSW_10_6_8/src
cmsenv
git cms-addpkg Validation/RecoTrack
nano Validation/RecoTrack/python/trackingNtuple_cff.py
'''

Set  _includeSeeds=False and _includeTrackingParticles=False in the config file.

'''
scram b -j 10
'''

Get a skeleton for the reconstruction step from some working flow that suits you. trackingOnly -versions are enough, and save time.

'''
runTheMatrix.py -l 10824.1 -j0
'''

Customise the process by removing all the qualityCuts made by track classifiers of different iterations. DO NOT MISS THIS STEP! This is slightly dependent on the which version of tracking is used but for phase1, it is enough to add the following lines to the process customization of “step3” in the reconstruction:

'''
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
'''

Also to produce the trackingNtuple, you need to add the lines 

'''
from Validation.RecoTrack.customiseTrackingNtuple import customiseTrackingNtuple
process = customiseTrackingNtuple(process)
'''

For the files add whichever file you want to produce the candidates from. These files can be found using the CMS Data Aggregation System (https://cmsweb.cern.ch/das/)
