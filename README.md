# InstrumentDetection
Based on: [SciPy](https://www.scipy.org/scipylib/index.html), [NumPy](http://www.numpy.org/), [scikit-learn](http://scikit-learn.org/stable/index.html)<br>
Codes in `mfcc.py` partly originates from [scikits.talkbox](https://github.com/cournape/talkbox).<br>
<br>
Follow these steps:<br>
1. Transfer all the training and testing music to 16bit/32bit/floating-point `.wav` files.
2. Save the training music played by a same instrument together in one same folder. It's recommended to name the folder the instrument's name. Copy `generateMFCC.py` to each folder and run them. You'll get mfcc, △MFCC and △△MFCC saved in `insrument_name.npy` files.
3. Copy the `.npy` files you just got and `trainmodel_SVM.py` to one same folder. Run the py file and you'll get the SVM model named `model_svm`.
4. Copy `model_svm` and `test.py` to the folder where only testing music is saved(no training music in the same folder). Run the py file and the detection results will be shown.
On going
