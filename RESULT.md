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

I tried four different activation functions including tangh, ReLue function, signum, and sigmoid. The output of each network using either one of these activation functions looked odd; mainly, because some of these functions such as sigmoid ranges from 0 to 1 and contains continuous values, and interpreting the output was challenging since digits are expected to be discrete numbers. Also, sign function has always zero derivative, so it was not a proper activation function to use in backpropagation algorithm. After, tunning hyperparameters and using different activation functions, but not getting a satisfactory output from the network, I changed the topology of the network. 



