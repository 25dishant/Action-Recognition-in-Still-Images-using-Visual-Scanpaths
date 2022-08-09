# Action-Recognition-in-Still-Images-using-Visual-Scanpaths
Action Recognition in still images has been a challenging task because unlike videos,
spatio-temporal features cannot be used here. But Human gaze behaviour can be
harnessed to incorporate the temporal information in the task of automated action classification in still images. In this thesis, we have proposed an LSTM based
context module that can learn the sequence of object proposals in which they are
being observed in a particular scene and based on this learned sequence an still image can be classified into one of the action classes. A sequencer algorithm reorders
the object proposals in the sequence provided by human gaze behaviour and before feeding to the LSTM, positional encoding is concatenated with each instance
appearance feature. LSTM is capable of learning the sequence in which each instance is being observed and it can also learn the relative position of the current
object with respect to the last object in the sequence and thereby remembering the
geometrical distribution of the object proposals in the scene. Proposed model is
a concatenation of the residual network and and an LSTM based context module
and has shown a mean average precision of 92.00% on action classification of still
images from PASCAL VOC-2012 action dataset. Residual networks with different
depths are experimented and it is found that as the depth of the feature extractor
network is increased, the mean average precision is also increased.
