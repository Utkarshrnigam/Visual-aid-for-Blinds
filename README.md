This project  makes the blind aware about the surroundings using video captioning and object detection. It is real time video captioning , which captures the frame from video and generates the caption for it and converts the generated caption into a voice using text to voice API.
                       
 
 
                                                                                                                                                                       
 
                                                                                                                                                                       4                                                                                                                                                                      
                      
        Chapter 2: Literature Survey
 
2.1 Image Captioning
2.1.1 ResNet50 model
* ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. This model was the winner of ImageNet challenge in 2015. The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks was difficult due to the problem of vanishing gradients.
* ResNet-50 that is a smaller version of ResNet 152 and frequently used as a starting point for transfer learning.
* Skip Connection — The Strength of ResNet - ResNet first introduced the concept of skip connection. The diagram below illustrates skip connection. The figure on the left is stacking convolution layers together one after the other. On the right we still stack convolution layers as before but we now also add the original input to the output of the convolution block. This is called skip connection.

                                                                                                                                                                              5
 
2.1.2 Word embeddings
Word embedding is one of the most popular representations of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.They are vector representations of a particular word
Our objective is to have words with similar context occupy close spatial positions. Mathematically, the cosine of the angle between such vectors should be close to 1, i.e. angle close to 0.Here comes the idea of generating distributed representations. Intuitively, we introduce some dependence of one word on the other words. The words in context of this word would get a greater share of this dependence. In one hot encoding representations, all the words are independent of each other, as mentioned earlier.

                           2.1.3 Functional API of keras
The Keras functional API provides a more flexible way for defining models.It specifically allows you to define multiple input or output models as well as models that share layers. More than that, it allows you to define ad hoc acyclic network graphs.Models are defined by creating instances of layers and connecting them directly to each other in pairs, then defining a Model that specifies the layers to act as the input and output to the model.

				2.1.4 LSTM
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.This is a behavior required in complex problem domains like machine translation, speech recognition, and more.
Hence standard RNNs fail to learn in the presence of time lags greater than 5 – 10 discrete time steps between relevant input events and target signals. The vanishing error problem casts doubt on whether standard RNNs can indeed exhibit significant practical advantages over time window-based feedforward networks. A recent model, “Long Short-Term Memory” (LSTM), is not affected by this problem. LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing constant error flow through “constant error carrousels” (CECs) within special units called cells
                                                                                                               				  2.2 Object Detection
YOLO is an extremely fast real time multi object detection algorithm. YOLO stands for “You Only Look Once”.The algorithm applies a neural network to an entire image. The network divides the image into an S x S grid and comes up with bounding boxes, which are boxes drawn around images and predicted probabilities for each of these regions.
The method used to come up with these probabilities is logistic regression. The bounding boxes are weighted by the associated probabilities. For class prediction, independent logistic classifiers are used.
Classes in yolo are - 
Person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donuts, cake, chair, sofa, potted plant, bed, dining table, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.
 
 
 
 
 
                                             
 
 
 
 
 Chapter 3: Detailed Work

3.1 Python Libraries :
* Tensorflow GPU : TensorFlow is an open source software library for high performance numerical computation.
*  Numpy : NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
*  Pandas : Pandas is a software library written for the Python programming language for data manipulation and analysis.
* Json : Json is a library for the Python programming language for dealing with json files.
* Re : The Python module re provides full support for Perl-like regular expressions in python.
* Collections : Collections in Python are containers that are used to store collections of data, for example, list, dictionary, set, tuple etc.
* Keras : Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible.
* Pickle : Python pickle module is used for serializing and de-serializing a Python object structure.
* Time : Python has defined a module, “time” which allows us to handle various operations regarding time, its conversions and representations, which finds its use in various applications in life.
* String : Python String module contains some constants, utility function, and classes for string manipulation.
* PIL : Python Imaging Library is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
* Pyttsx3 : pyttsx3 is a text-to-speech conversion library in Python. It works offline.
 
3.2 Proposed  deep learning model structure.
Our model basically consists of 2 different inputs, one for images and other for captions. Hence there will be two outputs for the above two inputs.
3.2.1 Image model
This model will accept the input of shape (2048, ) which is vector consisting of features extracted from the image using resnet50 model.
After that we add a dropout layer to reduce the overfitting.
Then we map this to 256 neurons. And this will be the output for the same.

 3.2.2 Caption model
This model will accept the input shape of (maxlen,) where maxlen in the longest possible caption in the dataset which is 49, the next layer will be embedding layer which will map each world in caption to vector of size  (50, ) . After that a dropout layer is added to reduce the chances of  overfitting. 
Then we add a LSTM(Long Short Term Memory) layer to predict the next possible world. This LSTM layer cells will generate activation vector of size (256, )     
And this will be the output for the same.                                                                                                          

3.2.3 Decoder model
We use decoder to combine the two outputs from the above two models and then map it to neurons of size (256, ) 
Finally we add output layer of size of vocab_size which is the total number of distinct words in dataset whose value is 10103.
3.2.4 Combined model
Then we create a model using keras functional api to create a model that gives input to image and caption model and then receive the output from decoder model and in addition we set the weights of embedding layer as of glove vector. And after that set the weights of embedding layer as non trainable.                                                                                                                                                                     
3.2.5 Model summary

3.3 Training of model:
Training starts after creating some important dictionaries needed for training. These include encoding_train(maps images to feature vectors)  idx_2_word(maps indexes to words), word_2_idx(maps words to idx), id_img(maps image ids to their name), description(maps images to their 5 captions).
Feeding data to the model created above - 
1. We don't send the image and whole caption at once, but instead we send image and the first world of the caption, and predict the next world.
2. Then calculate the loss and backpropagate the loss.
3. Then in the next step we send image again and the first word with the actual next word(not the predicted one) then repeat step 2.
4. We do step 3 for whole caption. And then For all 5 captions before sending the next image.
5. This is the training structure.
6. We have to convert these worlds to number so that our deep learning model can understand them. So, we use word_2_idx dictionary to map all words to indexes(0-10103).
7. Also padded all word array with 0’s because LSTM takes input for all cells at once and input cant be null

Following Above Steps above model was able to minimize the loss till 3.2(approx).


3.4 Use of generator function : 
Total size of data to be trained : 
We send each image for maximum of 
number of captions * maximum number words in caption
= 5*49 = 245
Total no of images = 118000(approx)
Total data points = 118000 * 245
		       = 28,910,000
Then we calculate length of partial caption which are mapped with glove vector of length 50
Let x = 49 * 50
         = 2450
And also each image feature vector is of size 2048
X = 2450 + 2048 = 4498
 
Finally size of data matrix = 28,910,000 * 4498
                                               = 130,037,180,000
Now even if we assume that one block takes 2 byte, then, to store this data matrix we will require more than 243 GB of main memory.
Hence we send mini batches consisting of 30 images each.
Used generator function to feed data matrix of these 30 images to our Deep learning model.



					3.5 Prediction 
For captioning we use open cv to capture image frames from the webcam. Then preprocess each frame and extract its feature vector from resnet50 model.
Then we send first send the image vector and a predefined starting sequence(<s>) of caption to the model. The model will predict the next word of the caption and we will stop the prediction until we receive a predefined ending sequence(<e>).
For Object detection after getting frame from webcam we preprocess and normalize it using openCV function cv2.dnn.blobFromImage and send it to the YOLO network, which returns the bounding box and the confidence of all possible classes in the image. Then we use the openCV to plot the bounding box to the image and store the classes to an array.
 
3.6 Working of Project
For Working of project your laptop and mobile phone should be on same server. Our project will be more effective if will work on our android devices.
For this we use two third party apps
1. IPWebcam : This app capture image from your phone and send it to a local server. From there we collect the response(which is the frame of video)
2. SoundWire Server : This simply sends the audio of your laptop to your phone a server and plays it on your phone
 
So, we capture the video from our phone and send it to our laptop in real time using IPWebcam. Then we send these image frames of video to our prediction model of captioning and object detection. Hence after receiving the output from both captioning and object we use Pyttsx3 to convert the captions and objects detected to audio and uses SoundWire Server application to send and play this audio to your mobile phone
 
   
Chapter 4 Sample Output


Scenario: Blind person in park
Input: Image of bench captured by software
Output: Caption generated “Bench in park”
Outcome: Helps blind to locate a sitting place in park.Scenario: Blind person in food market.
Input: Image of a fruit seller.
Output: Caption generated “Men is standing in grocery with oranges.”
Outcome:Helps blind to find fruit vendor in the market.Scenario: Blind person in house.
Input: Image of a bathroom
Output: Captions generated “Bathroom with toilet”  
Objects detected : Sink
Outcome: Helps blind to locate bathroom and sink in house.Scenario: Blind person near a crossing.
Input: Image of a crossing with stop sign.
Output: Captions generated “Man standing next to bike”.
Objects detected : Motor bike, person, stop sign.
Outcome: Awares the person about the nearby crossing.Scenario: Blind person in home.
Input: Image of a refrigerator.
Output: Objects detected : Refrigerator
Outcome: Helps blind to locate refrigerator in room.Scenario: Blind person walking on the street.
Input: Image of a cow on the road.
Output: Captions generated “
Horse in front of building.”
Objects detected : Cow
Outcome: Aware the blind about the stray animals on the street. Scenario: Blind person in street.
Input: Image of a street food vendor.
Output: Group of people sitting on table with food.
Outcome: Helps blind to search for street food vendor on local street.Scenario: Blind person in room.
Input: Image of a window.
Output: Captions generated “Window with window”
Outcome: Helps blind to find window in a room.Scenario: Children playing in the ground.
Input: Image of a boy playing cricket
Output: Captions generated “Two people playing soccer in the air”.
Objects detected : Person, baseball bat 
Outcome: Helps blind to get awareness about what is going in the field.


Chapter 5 Conclusion

This project will be useful for blind people in their day to day lives for getting awareness of the surroundings. Currently there are no as such aids for blinds which this project can be. In this project, using our collected dataset of 118k images blind can get aware of many types of situations in day to day life activities. We achieved satisfactory results and the project proved to be quite stable and suitable for the collected images in the dataset.

In future we plan to deploy this project on web or android. We also plan to include crosswalk recognition for blinds to cross zebra crossing on roads.


     References

[1] Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
Show and Tell: A Neural Image Caption Generator

[2] Marc Tanti, Albert Gatt, Kenneth P. Camilleri
What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?

[3] Joseph Redmon?, Santosh Divvala?†, Ross Girshick¶, Ali Farhadi?
YouOnlyLookOnce: Uni?ed,Real-TimeObjectDetection
[4]Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.                                        Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
[5]Joseph Redmon, Ali Farhadi.                                                                                  YOLOv3: An Incremental Improvement

