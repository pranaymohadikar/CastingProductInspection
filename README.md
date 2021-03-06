# Casting Product Inspection

## Context
This dataset is of casting manufacturing product.
**Casting** is a manufacturing process in which a liquid material is usually poured into a mould, which contains a hollow cavity of the desired shape, and then allowed to solidify.

### Reason for collection of this data is _casting defects_!!
Casting defect is an undesired irregularity in a metal casting process.
There are many types of defect in casting like blow holes, pinholes, burr, shrinkage defects, mould material defects, pouring metal defects, metallurgical defects, etc.
Defects are an unwanted thing in casting industry. For removing this defective product all industry have their quality inspection department. But the main problem is this inspection process is carried out manually. It is a very time-consuming process and due to human accuracy, this is not 100% accurate. This can because of the rejection of the whole order. So it creates a big loss in the company.
So to make the inspection process automatic and for this, we need to make deep learning classification model for this problem.

### Dataset
These all photos are top view of submersible pump impeller.
The dataset contains total _7348_ image data. These all are the size of _(300*300) pixels_ grey-scaled images. In all images, augmentation already applied.


There are mainly two categories:-
* Defective


  ![Defective](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/defective.jpeg)
* Ok


  ![Ok](https://github.com/pranaymohadikar/CastingProductInspection/blob/62492d2e3ee2f6a25a9d8426c289d521b46a379d/ok.jpeg)


### Exploratory Data Analysis
This dataset is converted into dataframe for the training purposes with labels and image location


![DF](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/dataframe.png)

After converting into training and testing folders this dataset is used for countplots and plotted the graph for the training dataset which is further used for training testing and validation.

![count](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/counts.png)

Countplot showing it has 3758 defective fronts and 2875 ok fronts

![cntplt](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/countplot.png)

Dataset is divided into 3 parts :
* Training
* Testing
* Validation

![shapes](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/shapes.png)
![datas](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/train_test_val.png)

### Convolutional Neural Network(CNN) model building
Created a NN for the dataset to train and predict the defective and ok fronts for industrial purpose.

This model has 6 layers and here is summary of model

![summary](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/model%20summary.png)

Optimizer used= 'Adam'
Also used model checkpoints for checking the overfitting so that when things go wrong model will be saved before that point.


### Results
accuracies and losses has been predicted for the given dataset.
__Accuracy__ : ~98%

![result](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/details%20about%20loss%20and%20acc.png)

With this we can also plot confusion matrix and provide classification report

![cm](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/confusion_matrix.png)

Here 0 = defective front and 1 = ok front
* x-axis = actual
* y-axis = predicted

Here precison and recall will be equally important so we will be checking Fscore.
* __FScore__ : ~99%

![rep](https://github.com/pranaymohadikar/CastingProductInspection/blob/main/classification_report.png)

