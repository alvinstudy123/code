
import matplotlib.pyplot as plt 
import os
import numpy as np


def load_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    x = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_GRAYSCALE)

            x.append(image)
            y.append(label)

    return np.array(x), np.array(y).astype('uint8')


def create_dataset(path):
    x, y = load_dataset("train", path)
    x_test, y_test = load_dataset("test", path)

    return x, y, x_test, y_test



class Layer_Dense:

    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T , dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_RELU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0

      
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        return data_loss


    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss_CategoricalCrossentropy(Loss):

    def forward(self,y_pred, y_true):
        #number of samples in a batch
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)


    #only if categorical labels:
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

    #for one-hot encoded labels:
        elif len(y_true.shape) == 1:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods
    
  

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1+np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1-self.output)*self.output


class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1-y_pred_clipped))
        return sample_losses
    
    def backward(self, output_values, y_true):
        no_samples = len(output_values)
        no_outputs = len(output_values[0])
        clipped_output_values = np.clip(output_values, 1e-7, 1-1e-7)
        
        self.dinputs = -(y_true/clipped_output_values - (1-y_true) / (1 - clipped_output_values) / no_outputs)

        #normalise gradients
        self.dinputs = self.dinputs / no_samples

class  optimizer_SGD:
    def __init__(self, learning_rate=1, decay=0, momentum = 0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.current_learning_rate = learning_rate

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate* (1. / (1. + self.decay*self.iterations))


        #update params
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Accuracy:
    def calculate(self,predictions,y):
        comparisons = (predictions == y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    
    def calculate_accumulated(self):
        accumulated_accuracy = self.accumulated_sum / self.accumulated_count
        return accumulated_accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    


x, y, x_test, y_test = create_dataset("D:\IB\EE\dataset EDITTED")

x = (x.reshape(x.shape[0], x.shape[1]*x.shape[2]).astype(np.float32) - 127.5)/127.5
x_test = (x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]).astype(np.float32) - 127.5)/127.5

#shuffling: 
keys = np.array(range(x.shape[0]))
np.random.shuffle(keys)
x = x[keys]
y = y[keys]

y = y.reshape(-1,1)
y[y == 2] = 1
y[y == 7] = 0
y_test = y_test.reshape(-1,1)
y_test[y_test == 2] = 1
y_test[y_test == 7] = 0




dense1 = Layer_Dense(x.shape[1],64)

dense2 = Layer_Dense(64,1)

activation1 = Activation_RELU()

activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()

accuracy = Accuracy()

optimizer = optimizer_SGD(learning_rate=1, decay = 0, momentum=0)


for epoch in range(100):
    
    if not (epoch % 10):
        print(f'epoch: {epoch}')


    train_steps = 1
    batch_size = 128

    if batch_size is not None:
        train_steps = len(x) //batch_size
        if train_steps * batch_size < len(x):
            train_steps += 1

    loss_function.new_pass()
    accuracy.new_pass()

    for step in range(train_steps):
        if batch_size is None:
            batch_x = x
            batch_y = y
        else:
            batch_x = x[step*batch_size:(step+1)*batch_size]
            batch_y = y[step*batch_size:(step+1)*batch_size]

        dense1.forward(batch_x)

        activation1.forward(dense1.output)

        dense2.forward(activation1.output)

        activation2.forward(dense2.output)
        

        loss = loss_function.calculate(activation2.output, batch_y)
        

        #calculating accuracy
        predictions = (activation2.output > 0.5) * 1
        accuracy_ = accuracy.calculate(predictions,batch_y)

   


        #backwards
        loss_function.backward(activation2.output,batch_y)
        
        activation2.backward(loss_function.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        
        

        #print
        if not (epoch % 10) and (not step % 50):
            (f'step: {step}, '+
             f'acc: {accuracy_}, '+
             f'data_loss: {loss:.3f}, '+
             f'lr: {optimizer.current_learning_rate}')
            
    epoch_data_loss = loss_function.calculate_accumulated()
    epoch_accuracy = accuracy.calculate_accumulated()

    if not epoch % 10:
        print(f'epoch acc: {epoch_accuracy:3f} '+
              f'epoch data loss: {epoch_data_loss:3f} '+
              f'lr: {optimizer.current_learning_rate} ')
         


    '''if loss < lowest_loss:
        print("new set of weights found, iteraction: ", iteration, "loss: ", loss, "acc: ", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    
    else: 
        dense1_weights = best_dense1_weights.copy()
        dense2_weights = best_dense2_weights.copy()
        dense1_biases = best_dense1_biases.copy()
        dense2_biases = best_dense2_biases.copy()'''


    #validation

    loss_function.new_pass()
    accuracy.new_pass()

    validation_steps = len(x_test) // batch_size
    if validation_steps * batch_size < len(x_test):
        validation_steps += 1

    for step in range(validation_steps):
        if batch_size is None:
            batch_x = x_test
            batch_y = y_test
        else:
            batch_x = x_test[step*batch_size:(step+1)*batch_size]
            batch_y = y_test[step*batch_size:(step+1)*batch_size]

        
        dense1.forward(batch_x)

        activation1.forward(dense1.output)

        dense2.forward(activation1.output)

        activation2.forward(dense2.output)
        

        loss = loss_function.calculate(activation2.output, batch_y)
        

        #calculating accuracy
        predictions = (activation2.output > 0.5) * 1
        accuracy_ = accuracy.calculate(predictions,batch_y)
    validation_loss = loss_function.calculate_accumulated()
    validation_accuracy = accuracy.calculate_accumulated()

    if not epoch % 10:
        print(f'validation, acc: {validation_accuracy:3f}, '+
              f'loss: {validation_loss:3f}')







