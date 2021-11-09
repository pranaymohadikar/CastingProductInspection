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
* ![Defective](CastingProductInspection/defective.jpeg)
* ![Ok]
  (https://github.com/pranaymohadikar/CastingProductInspection/blob/62492d2e3ee2f6a25a9d8426c289d521b46a379d/ok.jpeg)
