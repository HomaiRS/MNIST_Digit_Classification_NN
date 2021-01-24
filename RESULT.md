I implemented the multicategory PTA for digit classification in MATLAB. Data was downloaded from <http://yann.lecun.com/exdb/mnist/>. Four files listed on top of the webpage are required to be downloaded including training set images, training set labels, test set images, and test set labels. In the Matlab codes “MNIST_digit_classification.m” is the main function that calls “ReadBinaryImages.m” for reading binary MNIST dataset images. Below, I showed the first 100 images in the train set having their actual labels in the title to validate data reader function works properly, and labels are match with the images.

![TestImages_100](https://user-images.githubusercontent.com/43753085/105637029-d3377400-5e30-11eb-91d5-289f4fc45294.png)

“VectorizeMx” function is used to vectorize each 28*28 image. “ActivationStepfunction” is the activation function which in this question, step function is used as the activation function. “DesireOutput” function outputs d(x_i). To efficiently computate weights in the code, instead of programming the multicategory Perceptron Algorithm (PTA) with loope while calculating misclassification errors based on the values of induced local fields on the the vectorized version of the
28 × 28 image from the training set images, I used matrix multiplications. Further, in “myPTA_OnTestSet” function, I do not compute the weights agian, and instead I used the weights that are already learned on the first n≤60000 training set images, and applied them on the 10,000 test set images.

Since the patterns are not linearly separable, the misclassification errors may not converge to 0; therefore, I stopped the iterations (epochs) when the ratio of misclassified input patterns falls below some threshold ε. I observed this step terminates with 0 errors eventually in small number of epochs (e.g., 14, 16, etc.) using n = 50, η=1,ϵ~0 (10^(-6)). Thus, the training error is 0%. The error curve indicates some oscillations, and it is not strictly descending, as indicated below. As epoch values increases the error decreases, and eventually it reaches 0, as shown below.

![Part_f_ErrorPlot](https://user-images.githubusercontent.com/43753085/105637383-e6e3da00-5e32-11eb-91ce-0a037faeecd8.png)

Using only 50 training images to learn the weights, the percentage of the misclassification error over all 10,000 test samples will be 37.28%. The misclassification error of the training set is lower than this error on test set. Probably because we tune/compute weights on the chosen training images, so we have small error on this set, but these weights are not specifically learned on the test data, so the test data have higher misclassification error. In the next step we show that if we learn the weights using a larger number of train images, the classification results on the test set will be more accurate. Also, another reason for the discrepancy of errors obtained through the test and train sets might be the over-fitting problem while the train error is 0 and it contains small number of samples. 

Using n = 1000, η=1,ϵ~0 (10^(-6)), the misclassification error versus epochs plot is shown below. The error plot is more oscillatory than before, but eventually the error reaches 0 again (after ~350 epochs). 

![Part_g_ErrorPlot](https://user-images.githubusercontent.com/43753085/105637503-7be6d300-5e33-11eb-9843-1260d2581d27.png)

By running this on n = 60,000 training images using ϵ~0 (10^(-6)), the misclassification error increases by increasing the number of epochs. One reason that misclassification error does not converge to 0, or it does not go below the threshold/tolerance (ϵ) is that the patterns are not linearly separable. The error does not reach the stopping criteria even after 10,000 iterations. In the following, the error plot shows that the algorithm does not reach the solution even after running ~850 iterations. 

![Part_h_ErrorPlotV2](https://user-images.githubusercontent.com/43753085/105637694-4db5c300-5e34-11eb-9360-fe847f777f1b.png)

I used “rand” function in Matlab to initialize the 10*784 weights matrix. Thus, each time, that the categorical PTA code is ran, the weight matrix has different values. Also, ϵ is set to 10^(-6), and η equals to 1. The weights are trained on 60,000 train images and tested on 10,000 test images. The lowest train error (misclassification on train images) in second-run converged to 8.33%, and the corresponding test error for the learned weights was 40.5%. In first-run the training error was 83% and the test error was 73%. Using lower tolerances, I obtained a better test error while the misclassification error on the train set does not converge to 0. 


---

Given that multi-category PTA did not result in accurate digit classification since the patterns are not linearly separable, as asecond approach, I used backpropagation algorithm for the same classification task on the same MNIST dataset. 

I tried different network topologies to test what network architecture results in a better classification. In my first attempt, I constructed a flexible network that could have as many hidden layers and hidden neurons as I wanted, and the output layer was a single neuron indicated in Figure 1. I tried 1) two hidden layers with 10 hidden neurons in each layer 2) three hidden layers with 10 neurons in each layer 3) two hidden layers with 50 neurons in the first layer and 10 layers in the rest two layers. However, none of these network topologies led to a promising classification accuracy and low mean square error value. One of the biggest challenges I faced was what would be the best activation function for the last hidden layer that can map the (e.g., n=10) neurons of the last layer to a single value (digit) as each output digit was represented as discrete integers from 0 to 9. 

![Topology1_fail](https://user-images.githubusercontent.com/43753085/105638018-1e07ba80-5e36-11eb-85c2-fa0dedf09dcd.jpg)

I tried four different activation functions including tangh, ReLue function, signum, and sigmoid. The output of each network using either one of these activation functions looked odd; mainly, because some of these functions such as sigmoid ranges from 0 to 1 and contains continuous values, and interpreting the output was challenging since digits are expected to be discrete numbers. Also, sign function has always zero derivative, so it was not a proper activation function to use in backpropagation algorithm. After, tunning hyperparameters and using different activation functions, but not getting a satisfactory output from the network, I changed the topology of the network. In the new implementation of the network, I have 10 neurons in the output layer, and as many as I want in the hidden layers as indicated below. 

![Topology2_success](https://user-images.githubusercontent.com/43753085/105638058-50b1b300-5e36-11eb-8a81-1db374faefe9.jpg)

In the new architecture of the network, I represented digits 0, 1, …9, in the output layer as a one-zero vector that all its elements are 0 except the corresponding index to the actual digit that has a value of 1. The way I represented digits is shown in Figure 3.

![digits](https://user-images.githubusercontent.com/43753085/105638072-658e4680-5e36-11eb-9366-b75efb45d064.png)

I changed hyperparameters values including the learning rate, number of hidden layers, number of hidden neurons, activation function, variance/distribution used for initializing the weights, and etc. to see how the misclassification error/classification accuracy changes. Even though I changed the number of neurons and layers as well as other hyper parameters, I utilized the same activation function throughout the network. I used sigmoid function that is infinitely many times differentiable. 

I utilized online learning. I used 50,000 images as the training data and 10,000 as the test data. The test data and training data do not have any overlap. In order to find the optimal weights that minimizes the energy function in backpropagation algorithm, the model has been trained only on the training data and then tested on the test data. 

I visualized about 5 cases that have different hyperparameters in Figure 4, 5, 6, 7, 8. In each of these figures, I showed how the training and test accuracy changes per epoch until convergence, as well as showing how the mean squared error (MSE) on train and test data changes per epoch. In the top figure, MSE of the train is lower than the MSE of the test set as we are training the data on that set, so the residual should be smaller values. Also, the accuracy plot shows that the training maximum accuracy on the training data is 43% and maximum accuracy on the test set is ~30%. The parameters used in this run of the algorithm is eta=10 with maximum number of epoch = 200. The network has 2 hidden layers and 10 neurons per layer.
 
All the used hyperparameters are shown in the figures. In Figure 4, 5, 6, 7, and 8, I show how gradually I increased the accuracy on the test data by changing the hyperparameters.

![MSE_mean2](https://user-images.githubusercontent.com/43753085/105638135-b43be080-5e36-11eb-9a29-fe110de037ad.png)

![AccuracyPlot2](https://user-images.githubusercontent.com/43753085/105638164-d9305380-5e36-11eb-9627-e17e18ce220e.png)

Mean square error plot for training and test data. (Top). Classification accuracy plot for the training and test data. (bottom)

---

As it is shown, the classification accuracy is pretty low on both test and training set. The reason behind it is that the eta is very high and the algorithm does not converge to the optimal weight. Thus, I changed the parameters and reran the code.

![MSE_mean5](https://user-images.githubusercontent.com/43753085/105638205-0977f200-5e37-11eb-96bc-276592a86151.png)

![AccuracyPlot5](https://user-images.githubusercontent.com/43753085/105638219-18f73b00-5e37-11eb-811f-74d4abe07b26.png)

Mean square error plot for training and test data. (Top). Classification accuracy plot for the training and test data. (bottom)

---

![MSE_mean3](https://user-images.githubusercontent.com/43753085/105638261-58258c00-5e37-11eb-86ee-0af1ab602aea.png)

![AccuracyPlot3](https://user-images.githubusercontent.com/43753085/105638285-75f2f100-5e37-11eb-9518-aaab4bc3a216.png)

Mean square error plot for training and test data. (Top). Classification accuracy plot for the training and test data. (bottom)

---
I obtained the best classification accuracy using a three layers network with 20 neurons in the first layer and 10 neurons in the rest two layers. 

![bestAccuracy1](https://user-images.githubusercontent.com/43753085/105638321-a89ce980-5e37-11eb-8614-a42a479df156.png)

![bestAccuracy2](https://user-images.githubusercontent.com/43753085/105638324-ae92ca80-5e37-11eb-87e1-ee96865859dc.png)

I got my best classification using a three layers network with 20 neurons in the first layer and 10 neurons in the rest two layers. 
