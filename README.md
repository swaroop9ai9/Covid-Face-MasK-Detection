# Covid-Face-MasK-Detection.
Face mask detection using local camera frame by frame and classifies the frame with mask or without mask with probability and boundary box.

* Detect faces and determine whether they are wearing mask. *
* The code is implemented for face mask detection models with five mainstream deep learning frameworks （PyTorch、TensorFlow、Keras、MXNet和caffe） open sourced, and the corresponding inference codes.*

# Dataset.
* We trained the face mask detection models with five mainstream deep learning frameworks （PyTorch、TensorFlow、Keras、MXNet和caffe） open sourced, and the corresponding inference codes.
* We published 7959 images to train the models. The dataset is composed of WIDER Face (http://shuoyang1213.me/WIDERFACE/) and MAFA(http://www.escience.cn/people/geshiming/mafa.html).


# Implementation.

We used the structure of SSD. However, in order to make it run quickly in the browser, the backbone network is lite. The total model only has 1.01M parametes.

Input size of the model is 260x260, the backbone network only has 8 conv layers. The total model has only 24 layers with the location and classification layers counted.
*Implementation in MacOs*
Step 1: Open terminal. $ git clone https://github.com/swaroop9ai9/Covid-Face-MasK-Detection
Step 2: $ cd desktop/Covid-Face-MasK-Detection 
Step 3: For individual Image: $ python pytorch_infer.py  --img-path /path/to/your/img
Step 4: For Live Feed (Frame by Frame): python pytorch_infer.py --img-mode 0 --video-path 0 
Step 5: For Video Path: python pytorch_infer.py --img-mode 0 --video-path /path/to/video  

# Implementing it in [TensorFlow/Keras/MXNet/Caffe]
*The other four frameworks running method is similar to pytorch, just replace pytorchwith tensorflow, keras,caffe，mxnet, if you want to use tensorflow, just run:*
 $ python tensorflow_infer.py  --img-path /path/to/your/img
 
