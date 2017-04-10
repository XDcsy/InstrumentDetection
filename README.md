# InstrumentDetection
Based on: [SciPy](https://www.scipy.org/scipylib/index.html), [NumPy](http://www.numpy.org/), [scikit-learn](http://scikit-learn.org/stable/index.html)<br>
Codes in `mfcc.py` partly originates from [scikits.talkbox](https://github.com/cournape/talkbox).<br>
<br>
Follow these steps to run it:<br>
1. Convert all the training and testing audios to 16bit/32bit/floating-point `.wav` files. [pydub](http://stackoverflow.com/a/12391451/7708392) may help you convert MP3 to WAV.
2. Arrange the training audios in this structure: <br>+ Store the audios played by a same instrument in a same folder. <br>+ Name the folders the instruments' names. <br>+ Put all the folders in a same path. <br>+ Make sure there aren't any audios that are not training audios contained in the path. <br>
3. Put the testing audios together in one folder. The structure should look like this:
```
    training audios/
     |
     |-piano/
     |  |-*.wav
     |  |-*.wav
     |  |-...
     |
     |-guitar/
     |  |-*.wav
     |  |-*.wav
     |  |-...
     |
     |-violin/
     |  |-*.wav
     |  |-*.wav
     |  |-...
     |
     |-...
     
    testing audios/
     |-*.wav
     |-*.wav
     |-...
```
4. Run `generateMFCC.py`. Follow the program's instruction and enter the path of the training audios. You'll get mfcc, △MFCC and △△MFCC saved in `insrument_name.npy` files.
5. Run `trainmodel_SVM.py`. You'll get the SVM model named `model_svm` and a file named `names` which stores the names of the instruments.
4. Run `test.py` and the detection results will be shown.<br>
