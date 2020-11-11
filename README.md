# ERT_Face_Alignment
One Millisecond Face Alignment with an Ensemble of Regression Trees

This is an implementation of the face alignment method(ERT) by Jia Jinrang. And it has been first implemented by FeiLee1992 in 2017. 
If it is useful to you, please star to support my work. Thanks.

### About the model:
Because the Github limits the size of the file(can not be larger than 100M), we can not updata our trained-well model. If needed, please contact me through the following E-mail: jjr5401@163.com

### Configuration Environment:

ubuntu + cv2 + boost

### train data:

We used the lfpw dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) with 68 landmarks in a face to train our code. The parameters in our experiment followed the instruction in the origin paper. The error on validation data is about 0.0600433 which is worse than that of the origin paper(0.049). This may because the diffierent dataset(lfpw in ours and helen in origin paper).

### Installation:

Clone the repository
Complie
run

---

we used the cmake and make tool to compile the code. You can follow the CMakeLists.txt in ERT_Train to write a new one which is suitable for your own environment.

If you have trained an xml.model, you can use it with the ERT_Test code. What is important, it is just a simple example to show the process of using a trained-well xml model. You can diy the code and try a more effective way to use it. Through multithreading, the speed can be as fast as the origin paper.

### file instruction

- /ERT: the root directory of our project
  - /ERT/ERT_Train: the code and result of the training process.
    - /ERT/ERT_Train/code: the code of the ERT training process.
      - /ERT/ERT_Train/code/src: all the .cpp files are here.
      - /ERT/ERT_Train/code/inc: all the .h files are here.
    - /ERT/ERT_Train/build: the build directory.
    - /ERT/ERT_Train/model: the generated .xml files is here. we use .xml to save our model.
    - /ERT/ERT_Train/train_cascade_1 to /train_cascade_X: these 1 to X directories are the results of train data of the X cascades. 
    - /ERT/ERT_Train/validation_cascade_1 to /validation_cascade_X: these 1 to X directories are the results of validation data of the X cascades. 
    - /ERT/ERT_Train/train_origin_landmark: origin_landmark of trian data is here.
    - /ERT/ERT_Train/validation_origin_landmark: origin_landmark of validation data is here.
    - /ERT/ERT_Train/haarcascade_frontalface_alt2.xml: opencv face detector.
    - /ERT/ERT_Train/CMakeLists.txt: cmake file.
  - /ERT/ERT_Test: use model to test image.

Enjoy it ~~
Please Star it. Thank you.

### additional change note by Rinthel Kwon

- Add a root `CMakeLists.txt` as a workspace
  - You can build the whole project at once
- Make the project platform-independency
  - Use the [gulrak's c++ file system library](https://github.com/gulrak/filesystem)
    instead of the linux file system library, in order to make it platform-independent
  - Use the [nlohmann's modern json library](https://github.com/nlohmann/json)
    instead of the boost library for saving and loading a trained model,
    in order to remove a big 'boost' dependency
  - Now the project can be built not only on Windows, but also on macOS machine
    - Note that I didn't test on Linux, since I don't have it, but it should be without many modification.
- Use the cmake's `ExternalProject_Add()` command to automatically build the dependencies above
- Optimize code by using Eigen3, which can be accelerated by SIMD and use compile-time sized array,
  instead of cv::Mat, which only supports dynamic-sized array
  - This helps unnecessary memory allocation and deallocation processes during the learning
  - In my experience, the learning speed seems to be about x20 times faster than the original code
    when using my macOS machine
- Load and save the compact binary model
- Refactor code in order to reuse the same code for both of training and testing

##### how to build

```bash
$ cmake -H. -Bbuild
$ cmake --build build
```

##### how to execute

- Download face alignment training data from ibug (please refer the link above) and unzip files to `dataset/lfpw` or somewhere
- Run the training executable. The trained model files would be generated in `./result/model` by default.
```bash
$ ERT_Train -i ./dataset/lfpw
```
- Run the testing executable for webcam input
```bash
$ ERT_Test
```
- If you don't have a webcam, you can set and test an image file as an input
```bash
$ ERT_Test -i ./dataset/lfpw/testset/image_0001.png
```

##### to-do list

- [x] generate image and landmark file list without using subdirectory
- [x] utilize command line arguments to receive learning parameter
- [x] optimize the learning code for fast learning 
  - [x] use Eigen3 instead of cv::Mat for acceleration
  - [ ] use multthreads to learn tree
    - idea 1: use multithreads for evaluating each feature candidate
    - idea 2: use multithreads for evaluating weak regressor function for whole data
- [x] test a learnt model
- [x] load a learnt model saved by a json format
- [x] optimize test code
  - [x] use Eigen3, just like learning code
- [x] create a simple binary model, make it fast for loading and saving
  - a binary model size would be about x10 times smaller than a json model.
- [x] refactor
  - [x] renaming and following naming convension
  - [x] reuse the ERT model code for training / testing
- [ ] use [another face detection model](https://github.com/ShiqiYu/libfacedetection.git)
- [ ] find tracking method that does not have to reset normalized landmark shape with mean shape
  - currently normalized shape is reset by global mean shape for every frame
  - but when we use previous frame's shape, the shape become drifted rapidly. why?
- [ ] remove opencv dependency (is it possible?)