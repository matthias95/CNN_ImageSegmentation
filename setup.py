from setuptools import setup

setup(name='cnn_image_segmentation',
      version='1.0',
      description='CNN Imagesegmentation-Model based on ResNet50',
      author='Matthias Bernars',
      packages=['cnn_image_segmentation'],
      install_requires=[
          'opencv-python',
          'numpy',
          'matplotlib',
          'psutil',
        ]
     )
