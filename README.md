# Iris Classifiers in TensorFlow
This project consists of a jupyter notebook (Iris_Classifiers.ipynb) and the iris dataset (Iris.csv). The notebook 
consists of 4 cells were the first one is responsiple for loading and preparing the data and 3 different ways of 
implementing neural networks in TensorFlow using high-level APIs.

# Notebook in depth
To run the notebook:
1. Create a python 3 virtual environment.
2. Install the dependencies using pip:
    * jupyter
    * tensorflow
    * seaborn
    
3. Run jupyter notebook (command: jupyter notebook).
4. A browser windows should open on localhost:8888/tree. In the browser open "Iris_Classifiers.ipynb" by double clicking on the file.

**Notice: Some of the functions described can take other, or more, arguents than stated.**

## Cell 1 - Loading and preparing the data
In this cell all the data is handled to the degree that it can be used as input for all 3 different methods. The first 
couple of lines are purely imports of the necessary libraries.

### Loading the data
We load in the data as DataFrames, using pandas "read_csv" function which in its simplest form takes in just the path 
of the datafile.

Display the data using the seaborn library.
- **set(style="ticks")**: Set aesthetic parameters. In this example it sets the axes style.
- **set_palette("husl")**: Set color palette.
- **pairplot(csv_data.iloc[:,1:6],hue="Species")**: Makes a plot of pairs in the data set. First argument is the data, 
    the second (hue) is the data variable that gets plot. 

### Preparing the data
First the data gets shuffled since the data is organized by category:
- **csv_data.sample(frac=1).reset_index(drop=True)**
    * *sample(frac=1)*: Takes a fraction (0-1) of the data. In this case it takes all data (since frac = 1).
    * *reset_index(drop=True)*: Generates a new Dataframe with new indexes. The drop argument in this example gets rid of 
    the old index.
       
Secondly, we separate the data and their labels into variables using the iloc function:
- **csv_data.iloc[:,1:5].values**: In this example iloc takes the entire first 5 (1:5) columns(first :) of the dataset.
- **csv_data.iloc[:,5].values**: Take the entire 6th column.
 
Then we prepare the content of the data by encoding the labels. First into numerical notation, meaning every class gets 
its own id:
-  **preprocessing.LabelEncoder().fit_transform(labels)**
    * *LabelEncoder()*: Encodes the input to values between 0 and 1.
    * *fit_transform()*: Fits label encoder and return encoded labels.

After encoding the classes into numerical notation we turn that into one-hot encoding, meaning 0 = [0, 0, 0], 
1 = [0, 1, 0], 2 = [1, 0, 0]. 
- **pd.get_dummies(encoded_labels).values**: Converts the categorical input into a table of dummy/indicator variables 
and takes the only the values of that table.

After all this the data will be split into a training and test set (75~25).
- **train_test_split(data,Y,test_size=0.2,random_state=0)**: Splits the input data into a training and test set.
    * *test_size*: Fraction (0-1) of the data indicating the size of the test set. In this example our test size is 20%.
    * *random_state*: Seed used by the random number generator. In this case it uses np.random.

## Cell 2 - Neural Networks with Estimators
This cell implements a simple neural network using estimators. Estimators is a high-level TensorFlow API that wraps up 
the training loop.

### Input function
We first define an input function. This returns the features and labels of the data in a specific format.
- **input_fn()**: Returns the features and labels for the estimator. feature are the correlation between the raw data 
and the estimators.
- **feature_name**: String containing the name of our feature_columns
- **feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]**: Defines the types of data of the models 
input. Here we use a 4 feature column to represent our features.

Defining the feature columns for a model can be quite tricky. For more info see: 
https://www.tensorflow.org/guide/feature_columns
### Defining the model

- **classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3, model_dir="./models/linear1")**
    * *n_classes*: The number of outputs.
    * *model_dir*: The directory where the model is saved. 

### Training the model
- **train(input_fn=input_fn(x_train, y_train), steps=1000)**: Trains the model using the training data (x_train, y_train).
Steps is the amount of training steps the model takes. Each step the model uses a certain amount of data (normally 
defined in batch_size) .

### Evaluating the model
- **evaluate()[""]**: Tests the trained model on the test data. It takes in the input function(input_fn(x_test, y_test)) 
and the amount of steps. The bracket notation defines which parameters you want to see of the evaluation.  

## Cell 3 - Deep Neural Networks with Estimators
The third cell implements a deep neural network using estimators. The only differences with cell 2 are in the 
classifier. First of all, the classifier we use instead of a LinearClassifier we use a DNNClassifier 
(DNN = Deep Neural Network) and secondly an extra input for defining the classifier, namely hidden_units. The 
hidden_units argument should be an array defining the nodes of the network, meaning every index defines the amount of 
nodes in a single layer of the network.

## Cell 4 - Deep Neural Networks with Keras
The final cell shows an example of implementing deep neural networks using Keras. Keras is a high-level TensorFlow API, 
but also exists as a standialone library. 

### Defining the model
We start by defining what kind of model we want to define, after which we can manually add all the layers we want.
- **keras.Sequential()**: Use a sequential model as base of our model.
- **model.add()**: Adds a layer to the model.
    * *layers.Dense(int, activation)*: Creates a "regular" layer of neurons. For the inputs: the integer represent the 
    amount of nodes and the activation is the activation function. Optionally, the input_shape argument can be given 
    declaring that layer as an input layer.

### Compiling the model
After you're done with defining your model you have to compile it.
- **compile(tf.keras.optimizers.Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])**
    * *optimizers*: The method used by the model to optimize parameters.
    * *loss*: Method of measuring loss.
    * *metrics*: The metrics we are interested in, in the output.

### Train the model
To train the model all you have to supply are: the training features, training labels and the amount of epochs. Epochs 
dare the amount of times the model should use your entire training set to train.
- **fit(x_train,y_train,epochs=100)**: Trains the model on the training data. Epochs are the amount of times the model goes 
over the training data. 

### Evaluating the model.
- **evaluate(x_test, y_test)**: Similar to estimators evaluation, but this time there is no need for an input function.