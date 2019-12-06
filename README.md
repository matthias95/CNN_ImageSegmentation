# CNN_ImageSegmentation  
Fully Convolutional Image Segmentation based on ResNet50.  

<figure class="image">
  <img src="videos/20191206_00-06-02/movie.gif" width="80%">
  <img src="videos/20191206_00-04-54/movie.gif" width="80%">
  <figcaption>Model trained on detecting dice</figcaption>
</figure>

## Content  
[Training.ipynb](Training.ipynb)  [Jupyter](https://jupyter.org/) Notebook for training  
[cnn_image_segmentation/resnet_segmentation_model.py](cnn_image_segmentation/resnet_segmentation_model.py) Module containe the code to create the model

## Dependencies  
Tensorflow 2.0  
Python 3


## Setup  
`
cd CNN_ImageSegmentation
python setup.py develop 
`

## References  
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)  
