# ERT_Face_Alignment
One Millisecond Face Alignment with an Ensemble of Regression Trees

This is an implementation of the face alignment method(ERT) by Jia Jinrang. And it has been first implemented by FeiLee1992 in 2017. 
If it is useful to you, please star to support my work. Thanks.

##**About the model:**
Because the Github limits the size of the file(can not be larger than 100M), we can not updata our trained-well model. If needed, please contact me through the following E-mail: jjr5401@163.com

##**Configuration Environment:**

ubuntu + cv2 + boost

##**train data:**

We used the lfpw dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) with 68 landmarks in a face to train our code. The parameters in our experiment followed the instruction in the origin paper. The error on validation data is about 0.0600433 which is worse than that of the origin paper(0.049). This may because the diffierent dataset(lfpw in ours and helen in origin paper).

##**Installation:**

Clone the repository
Complie
run

---

we used the cmake and make tool to compile the code. You can follow the CMakeLists.txt in ERT_Train to write a new one which is suitable for your own environment.

If you have trained an xml.model, you can use it with the ERT_Test code. What is important, it is just a simple example to show the process of using a trained-well xml model. You can diy the code and try a more effective way to use it. Through multithreading, the speed can be as fast as the origin paper.

##**file instruction**

/ERT: the root directory of our project
  /ERT/ERT_Train: the code and result of the training process.
    /ERT/ERT_Train/code: the code of the ERT training process.
      /ERT/ERT_Train/code/src: all the .cpp files are here.
      /ERT/ERT_Train/code/inc: all the .h files are here.
    /ERT/ERT_Train/build: the build directory.
    /ERT/ERT_Train/model: the generated .xml files is here. we use .xml to save our model.
    /ERT/ERT_Train/train_cascade_1 to /train_cascade_X: these 1 to X directories are the results of train data of the X cascades. 
    /ERT/ERT_Train/validation_cascade_1 to /validation_cascade_X: these 1 to X directories are the results of validation data of the X cascades. 
    /ERT/ERT_Train/train_origin_landmark: origin_landmark of trian data is here.
    /ERT/ERT_Train/validation_origin_landmark: origin_landmark of validation data is here.
    /ERT/ERT_Train/haarcascade_frontalface_alt2.xml: opencv face detector.
    /ERT/ERT_Train/CMakeLists.txt: cmake file.
  /ERT/ERT_Test: use model to test image.

Enjoy it ~~
Please Star it. Thank you.
