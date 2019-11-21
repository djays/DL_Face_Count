# Face Counter

This is a learning exercise for counting faces using the Multi-task Cascaded Convolutional Networks (MTCNN implementation from https://github.com/TreB1eN/InsightFace_Pytorch)
Implemented using Python 3.5 and PyTorch 0.41.  

## Installation

1. Install Python3.5+ 

2. Unzip archive and enter the root of the extracted code

3. Depending on your platform install Pytorch 0.41 and torchvision following instructions from https://pytorch.org.
For me this was
```
pip install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl
```

4. Install remaining dependencies using
```
pip install -r requirements.txt
```

5. Clone the InsightFace implementation as following
 ```
  git clone https://github.com/TreB1eN/InsightFace_Pytorch.git insight_face
  ```
6. Create symlink 
```
  ln -s insight_face/mtcnn_pytorch mtcnn_pytorch
```

## Running the code
```
python script.py [path-to-file-or-dir]
```

For full list of arguments
```
python script.py -h
```

## Debug
By default execution logs are stored in the current directory in debug.log

## Unit Tests

Tests are located in the test directory, while not complete, do allow for building upon 
and adding new test cases
e.g.
```
 python -m unittest test.test_metrics
```

## Code Structure
  script.py -- Main execution file 
 
  config.py -- Basic configuration constants such as maxmimum number of faces, minimum face size. 
 
  models : base.py -- House extensible wrapper around the MTCNN code which can be replaced by other models by extending the FaceCounterBase class.
 
  metrics.py -- Responsible for computing metrics. Currently only supports Mean Absolute error.
 
  test -- Package housing test cases. 

## Algorithm

   Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks use a three stage architecture and leverages the combination of detection and alignment to boost performance.
   An image is fed to generate an image pyramid (img resized to different sizes) which is fed into the first stage with NMS and bounding box regression to generate candidates. 
   
   The 2nd stage refines these bounding boxes, rejecting a large number of the candidates.
   The 3rd stage generates the facial keypoints for the final bounding box candidates.
   
   This cascade of networks allow for comparatively lighter networks yet high performance. 
       
## Choices
 1. Metric: As the property number of faces is in the continous domain, Mean Absolute Error seems to be suitable metric for inference.
 2. To cleanly integrate the InsightFace code and allow future extensions I use the factory method pattern for model selection. 


## Challenges 
 1. For few images MTCNN code seems to fail (stacking of bounding boxes), I didn't get sufficient time to fix this.
 2. By reducing minimum face size, performance improves however a large number of faces still tend to get missed. 
 This is in part due to different nuisance factors which hinder visibility of the faces such as occlusion and variational lighting.  

 
