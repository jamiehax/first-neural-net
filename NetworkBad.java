import java.util.List;

public class Network {

     /*
      * DATA FIELDS OF NETWORK: numLayers: number of layers in network sizes: array
      * of number of neurons in each layer (layer=index). biases: list of layers
      * (starting at second layer), and for each layer a list is kept of the biases
      * of that layers neurons. weights: list of connections between layers, and for
      * each connection, a list is kept for the weight between each neuron in that
      * layer and every neuron in the previous layer.
      */

     int numLayers;
     int[] sizes;
     ArrayList<ArrayList<Double>> biases;
     ArrayList<ArrayList<ArrayList<Double>>> weights;

     // network constructor
     public Network(int[] sizes) {
          this.sizes = sizes;
          this.numLayers = sizes.length;
          this.biases = new ArrayList<ArrayList<Double>>();
          this.weights = new ArrayList<ArrayList<ArrayList<Double>>>();

          // add random biases
          for (int i = 1; i < sizes.length; i++)
               this.biases.add(getGaussianList(sizes[i]));

          // add random weights
          for (int layer = 1; layer < sizes.length; layer++) {
               ArrayList<ArrayList<Double>> layerList = new ArrayList<ArrayList<Double>>();

               for (int neuron = 0; neuron < sizes[layer]; neuron++)
                    layerList.add(getGaussianList(sizes[layer - 1]));

               this.weights.add(layerList);
          }

     }

     // returns output of network given a list of inputs
     public ArrayList<Double> feedForward(ArrayList<Double> inputs) {
          for (int connection = 0; connection < weights.size(); connection++) {

               ArrayList<Double> newInputs = new ArrayList<Double>();
               for (int neuron = 0; neuron < weights.get(connection).size(); neuron++) {

                    double dotProductSum = 0;
                    // calculate product between each input value and its weight, and sum
                    for (int input = 0; input < inputs.size(); input++) {
                         double product = inputs.get(input) * weights.get(connection).get(neuron).get(input);
                         dotProductSum = dotProductSum + product;
                    }
                    // add sum and bias, sigmoid it, and add it to new inputs for next layer
                    double activation = dotProductSum + biases.get(connection).get(neuron);
                    double a = sigmoid(activation);
                    newInputs.add(a);
               }
               inputs = newInputs;
          }
          return inputs;
     }

     // train the network without any test data to check progress against
     public void gradientDescent(ArrayList<ArrayList<ArrayList<Double>>> trainingData, int epochs, int miniBatchSize,
               double learningRate) {

          for (int i = 0; i < epochs; i++) {
               // shuffle data and create miniBatches list
               Collections.shuffle(trainingData);
               ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> miniBatches = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();

               // fill miniBatches list with the mini batches
               int k = 0;
               int n = 0;
               ArrayList<ArrayList<ArrayList<Double>>> miniBatch = new ArrayList<ArrayList<ArrayList<Double>>>();
               while (k < trainingData.size()) {
                    miniBatch.add(trainingData.get(k));
                    k++;
                    n++;
                    if (n >= miniBatchSize || k == trainingData.size()) {
                         miniBatches.add(new ArrayList<ArrayList<ArrayList<Double>>>(miniBatch));
                         n = 0;
                         miniBatch.clear();
                    }
               }

               for (int batch = 0; batch < miniBatches.size(); batch++) {
                    this.updateMiniBatch(miniBatches.get(batch), learningRate);
               }

               System.out.println(String.format("Epoch %d complete", i));

          }
     }

     // train the network with test data to check progress against
     public void gradientDescent(ArrayList<ArrayList<ArrayList<Double>>> trainingData, int epochs, int miniBatchSize,
               double learningRate, ArrayList<ArrayList<ArrayList<Double>>> testData) {

          for (int i = 0; i < epochs; i++) {
               // shuffle data and create miniBatches list
               Collections.shuffle(trainingData);
               ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> miniBatches = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();

               // fill miniBatches list with the mini batches
               int k = 0;
               int n = 0;
               ArrayList<ArrayList<ArrayList<Double>>> miniBatch = new ArrayList<ArrayList<ArrayList<Double>>>();
               while (k < trainingData.size()) {
                    miniBatch.add(trainingData.get(k));
                    k++;
                    n++;
                    if (n >= miniBatchSize || k == trainingData.size()) {
                         miniBatches.add(new ArrayList<ArrayList<ArrayList<Double>>>(miniBatch));
                         n = 0;
                         miniBatch.clear();
                    }
               }

               for (int batch = 0; batch < miniBatches.size(); batch++) {
                    this.updateMiniBatch(miniBatches.get(batch), learningRate);
               }

               System.out.println(String.format("Epoch %d = %d / %d", i, evaluate(testData), testData.size()));
          }

     }

     /*
      * preform gradient descent on the network given the inputs and expected results
      * as index 0 and 1 of each miniBatch element, and the learning rate
      */
     public void updateMiniBatch(ArrayList<ArrayList<ArrayList<Double>>> miniBatch, double learningRate) {
          // create lists of same size as weights and biases with zeros
          ArrayList<ArrayList<Double>> nablaB = new ArrayList<ArrayList<Double>>();
          for (int layer = 0; layer < this.biases.size(); layer++) {
               nablaB.add(new ArrayList<Double>(Collections.nCopies(this.biases.get(layer).size(), 0.0)));
          }
          ArrayList<ArrayList<ArrayList<Double>>> nablaW = new ArrayList<ArrayList<ArrayList<Double>>>();
          for (int layer = 0; layer < this.weights.size(); layer++) {
               nablaW.add(new ArrayList<ArrayList<Double>>());
               for (int outputNeuron = 0; outputNeuron < this.weights.get(layer).size(); outputNeuron++) {
                    nablaW.get(layer).add(new ArrayList<Double>(
                              Collections.nCopies(this.weights.get(layer).get(outputNeuron).size(), 0.0)));
               }
          }

          for (int batch = 0; batch < miniBatch.size(); batch++) {
               // call backprop with inputs and outputs to get gradients for weights and biases
               ArrayList<Double> inputs = miniBatch.get(batch).get(0);
               ArrayList<Double> outputs = miniBatch.get(batch).get(1);

               ArrayList<Object> gradients = backprop(inputs, outputs);
               ArrayList<ArrayList<Double>> deltaNablaB = (ArrayList<ArrayList<Double>>) gradients.get(0);
               ArrayList<ArrayList<ArrayList<Double>>> deltaNablaW = (ArrayList<ArrayList<ArrayList<Double>>>) gradients
                         .get(1);

               /*
                * set all values in nablaB to be nablaB + deltaNablaB, note that layer 0
                * corresponds to layer 1 in the actual network
                */
               for (int layer = 0; layer < deltaNablaB.size(); layer++) {
                    for (int neuron = 0; neuron < deltaNablaB.get(layer).size(); neuron++) {
                         double newBias = nablaB.get(layer).get(neuron) + deltaNablaB.get(layer).get(neuron);
                         nablaB.get(layer).set(neuron, newBias);
                    }
               }

               // set all values in nablaW to be nablaW + deltaNablaW
               for (int connection = 0; connection < deltaNablaW.size(); connection++) {
                    for (int neuron = 0; neuron < deltaNablaW.get(connection).size(); neuron++) {
                         for (int weight = 0; weight < deltaNablaW.get(connection).get(neuron).size(); weight++) {
                              double newWeight = nablaW.get(connection).get(neuron).get(weight)
                                        + deltaNablaW.get(connection).get(neuron).get(weight);
                              nablaW.get(connection).get(neuron).set(weight, newWeight);
                         }
                    }
               }

               ArrayList<ArrayList<Double>> oldBiases = new ArrayList<ArrayList<Double>>(this.biases);
               ArrayList<ArrayList<ArrayList<Double>>> oldWeights = new ArrayList<ArrayList<ArrayList<Double>>>(
                         this.weights);

               // update the networks weights and biases
               for (int layer = 0; layer < nablaB.size(); layer++) {
                    // System.out.println("===== LAYER " + layer + " ======");
                    for (int neuron = 0; neuron < nablaB.get(layer).size(); neuron++) {
                         // System.out.println("NEURON: " + neuron);
                         double currentBias = this.biases.get(layer).get(neuron);
                         double newBias = nablaB.get(layer).get(neuron);
                         double updatedBias = currentBias - (learningRate / miniBatch.size()) * newBias;
                         this.biases.get(layer).set(neuron, updatedBias);

                         // if ((this.biases.get(layer).get(neuron) - currentBias) == 0)
                         // System.out.println("bias didn't change");
                    }
               }

               for (int connection = 0; connection < nablaW.size(); connection++) {
                    // System.out.println("===== CONNECTION " + connection + " ======");
                    for (int neuron = 0; neuron < nablaW.get(connection).size(); neuron++) {
                         // System.out.println("=== OUTPUT NEURON: " + neuron);
                         for (int weight = 0; weight < nablaW.get(connection).get(neuron).size(); weight++) {
                              // System.out.println("=== INPUT NEURON: " + weight);
                              double currentWeight = this.weights.get(connection).get(neuron).get(weight);
                              double newWeight = nablaW.get(connection).get(neuron).get(weight);
                              double updatedWeight = currentWeight - (learningRate / miniBatch.size()) * newWeight;
                              this.weights.get(connection).get(neuron).set(weight, updatedWeight);

                              // if ((this.weights.get(connection).get(neuron).get(weight) - currentWeight) ==
                              // 0)
                              // System.out.println("weight didn't change");
                         }
                    }
               }

               // for (int layer = 0; layer < this.biases.size(); layer++) {
               // System.out.println("===== LAYER " + layer + " ======");
               // for (int neuron = 0; neuron < this.biases.get(layer).size(); neuron++) {
               // System.out.println("NEURON: " + neuron);
               // System.out.println("old bias: " + oldBiases.get(layer).get(neuron));
               // System.out.println("new bias: " + this.biases.get(layer).get(neuron));
               // }
               // }

               assert this.biases.equals(oldBiases) : "something changed!";
               assert this.weights.equals(oldWeights) : "something changed!";

          }
     }

     public ArrayList<Object> backprop(ArrayList<Double> inputs, ArrayList<Double> outputs) {
          // create lists of same size as weights and biases with zeros
          ArrayList<ArrayList<Double>> nablaB = new ArrayList<ArrayList<Double>>();
          for (int layer = 0; layer < this.biases.size(); layer++) {
               nablaB.add(new ArrayList<Double>(Collections.nCopies(this.biases.get(layer).size(), 0.0)));
          }
          ArrayList<ArrayList<ArrayList<Double>>> nablaW = new ArrayList<ArrayList<ArrayList<Double>>>();
          for (int layer = 0; layer < this.weights.size(); layer++) {
               nablaW.add(new ArrayList<ArrayList<Double>>());
               for (int outputNeuron = 0; outputNeuron < this.weights.get(layer).size(); outputNeuron++) {
                    nablaW.get(layer).add(new ArrayList<Double>(
                              Collections.nCopies(this.weights.get(layer).get(outputNeuron).size(), 0.0)));
               }
          }

          // feedforward starting with the inputs
          ArrayList<Double> activation = inputs;
          ArrayList<ArrayList<Double>> activations = new ArrayList<ArrayList<Double>>();
          activations.add(activation);
          ArrayList<ArrayList<Double>> zs = new ArrayList<ArrayList<Double>>();

          for (int connection = 0; connection < this.weights.size(); connection++) {
               ArrayList<Double> z = new ArrayList<Double>();
               for (int outputNeuron = 0; outputNeuron < this.weights.get(connection).size(); outputNeuron++) {
                    double zComponentSum = 0;
                    for (int inputNeuron = 0; inputNeuron < this.weights.get(connection).get(outputNeuron)
                              .size(); inputNeuron++) {
                         double zComponentValue = activation.get(inputNeuron)
                                   * this.weights.get(connection).get(outputNeuron).get(inputNeuron);
                         zComponentSum = zComponentSum + zComponentValue;
                    }
                    z.add(zComponentSum + this.biases.get(connection).get(outputNeuron));
               }
               // System.out.println("NEW Z: "+ z);
               zs.add(z);

               // make new activations sigmoid of every z
               for (int element = 0; element < z.size(); element++) {
                    z.set(element, sigmoid(z.get(element)));
               }

               activation = z;
               activations.add(activation);
          }

          // calculate the output error
          ArrayList<Double> sigmoidPrimeZs = new ArrayList<Double>();
          for (int z = 0; z < activations.get(activations.size() - 1).size(); z++) {
               double sigmoidPrimeZ = sigmoidPrime(activations.get(activations.size() - 1).get(z));
               sigmoidPrimeZs.add(sigmoidPrimeZ);
          }

          ArrayList<Double> partialDeriviatives = this.costDerivative(activations.get(activations.size() - 1), outputs);
          ArrayList<Double> delta = new ArrayList<Double>();
          for (int i = 0; i < sigmoidPrimeZs.size(); i++) {
               double deltaComponent = sigmoidPrimeZs.get(i) * partialDeriviatives.get(i);
               delta.add(deltaComponent);
          }

          // inner dot product of delta and second to last layer of activations transposed
          ArrayList<ArrayList<Double>> dotProduct = new ArrayList<ArrayList<Double>>();
          for (int d = 0; d < delta.size(); d++) {
               ArrayList<Double> componentList = new ArrayList<Double>();
               for (int a = 0; a < activations.get(activations.size() - 2).size(); a++) {
                    double component = delta.get(d) * activations.get(activations.size() - 2).get(a);
                    componentList.add(component);
               }
               dotProduct.add(componentList);
          }
          nablaW.set(nablaW.size() - 1, dotProduct);
          nablaB.set(nablaB.size() - 1, delta);

          // backpropogate the output error, stopping before input layer
          for (int layer = this.numLayers - 2; layer > 0; layer--) {
               ArrayList<Double> newDelta = new ArrayList<Double>();
               ArrayList<Double> z = zs.get(layer - 1);
               ArrayList<Double> sp = new ArrayList<Double>();
               for (int zIndex = 0; zIndex < z.size(); zIndex++)
                    sp.add(sigmoidPrime(z.get(zIndex)));

               for (int outputNeuron = 0; outputNeuron < this.weights.get(layer).size(); outputNeuron++) {
                    double dVal = 0;
                    for (int inputNeuron = 0; inputNeuron < this.weights.get(layer).get(outputNeuron)
                              .size(); inputNeuron++) {

                         dVal = dVal + delta.get(outputNeuron)
                                   * this.weights.get(layer).get(outputNeuron).get(inputNeuron);

                    }
                    dVal = dVal * sp.get(outputNeuron);
                    newDelta.add(dVal);
               }
               delta = new ArrayList<Double>(newDelta);
               ArrayList<ArrayList<Double>> nablaWeightsLayer = new ArrayList<ArrayList<Double>>();
               for (int d = 0; d < delta.size(); d++) {
                    ArrayList<Double> line = new ArrayList<Double>();
                    for (int a = 0; a < activations.get(layer - 1).size(); a++) {
                         line.add(delta.get(d) * activations.get(layer - 1).get(a));
                    }
                    nablaWeightsLayer.add(line);
               }

               nablaW.set(layer - 1, nablaWeightsLayer);
               nablaB.set(layer, delta);
          }

          ArrayList<Object> ret = new ArrayList<Object>();
          ret.add(nablaB);
          ret.add(nablaW);
          return ret;

     }

     /*
      * testData is a list containing lists of inputs (index 0) and outputs (index 1)
      * together in a list
      */
     public int evaluate(ArrayList<ArrayList<ArrayList<Double>>> testData) {
          int numCorrect = 0;
          for (int test = 0; test < testData.size(); test++) {
               // get max index of the results

               ArrayList<Double> testResults = feedForward(testData.get(test).get(0));
               int maxIndexResults = 0;
               for (int i = 0; i < testResults.size(); i++) {
                    if (testResults.get(i) > testResults.get(maxIndexResults))
                         maxIndexResults = i;
               }
               int maxIndexTest = 0;
               for (int i = 0; i < testData.get(test).get(1).size(); i++) {
                    if (testData.get(test).get(1).get(i) > testData.get(test).get(1).get(maxIndexTest))
                         maxIndexTest = i;
               }
               if (maxIndexResults == maxIndexTest)
                    numCorrect++;

          }

          return numCorrect;
     }

     public ArrayList<Double> costDerivative(ArrayList<Double> outputActivations, ArrayList<Double> expectedOutputs) {
          ArrayList<Double> partialDeriviatives = new ArrayList<Double>();
          for (int i = 0; i < outputActivations.size(); i++)
               partialDeriviatives.add((outputActivations.get(i) - expectedOutputs.get(i)));
          return partialDeriviatives;
     }

     // compute and return sigmoid of z
     public static double sigmoid(double z) {
          return 1.0 / (1.0 + Math.exp(-z));
     }

     // return derivative of sigmoid function
     public static double sigmoidPrime(double z) {
          return sigmoid(z) * (1 - sigmoid(z));
     }

     // return gaussian list of length x
     public static ArrayList<Double> getGaussianList(int x) {
          Random rando = new Random();
          ArrayList<Double> list = new ArrayList<Double>();

          for (int i = 0; i < x; i++)
               list.add(rando.nextGaussian());

          return list;
     }

     // main method
     public static void main(String[] args) throws IOException {

          Data data = new Data();
          data.setTrainingImageData("./data/train-images.idx3-ubyte");
          data.setTrainingLabelData("./data/train-labels.idx1-ubyte");
          System.out.println("set training data");
          data.setTestImageData("./data/t10k-images.idx3-ubyte");
          data.setTestLabelData("./data/t10k-labels.idx1-ubyte");
          System.out.println("set test data");

          // ArrayList<ArrayList<ArrayList<Double>>> trainingSubset = new
          // ArrayList<ArrayList<ArrayList<Double>>>(
          // data.trainingData.subList(0, 6));
          //
          // ArrayList<ArrayList<ArrayList<Double>>> testSubset = new
          // ArrayList<ArrayList<ArrayList<Double>>>(
          // data.testData.subList(0, 99));

          int[] netSizes = { 784, 30, 10 };
          Network net = new Network(netSizes);
          net.gradientDescent(data.trainingData, 30, 10, 3.0, data.testData);
          System.out.println("done training");

     }
}

class Data {

     public ArrayList<ArrayList<ArrayList<Double>>> trainingData = new ArrayList<ArrayList<ArrayList<Double>>>();
     public ArrayList<ArrayList<ArrayList<Double>>> testData = new ArrayList<ArrayList<ArrayList<Double>>>();

     public void setTrainingImageData(String filename) throws IOException {
          File file = new File(filename);
          byte[] fileContent = Files.readAllBytes(file.toPath());

          int pixel = 0;
          ArrayList<Double> image = new ArrayList<Double>();
          // image data starts at byte 16
          for (int i = 16; i < fileContent.length; i++) {
               // since java is cool and doesn't support unsigned bytes, convert to int first
               int value = Byte.toUnsignedInt(fileContent[i]);
               // convert to value between 0 - 1
               double pixelValue = value / 256.0;
               image.add(Double.valueOf(pixelValue));
               pixel++;
               // each image is 28 x 28 = 784 pixels
               if (pixel >= 784) {
                    ArrayList<Double> pixels = new ArrayList<Double>(image);
                    ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
                    list.add(pixels);
                    this.trainingData.add(list);
                    pixel = 0;
                    image.clear();
               }

          }
     }

     public void setTestImageData(String filename) throws IOException {
          File file = new File(filename);
          byte[] fileContent = Files.readAllBytes(file.toPath());

          int pixel = 0;
          ArrayList<Double> image = new ArrayList<Double>();
          // image data starts at byte 16
          for (int i = 16; i < fileContent.length; i++) {
               // since java is cool and doesn't support unsigned bytes, convert to int first
               int value = Byte.toUnsignedInt(fileContent[i]);
               // convert to value between 0 - 1
               double pixelValue = value / 256.0;
               image.add(Double.valueOf(pixelValue));
               pixel++;
               // each image is 28 x 28 = 784 pixels
               if (pixel >= 784) {
                    ArrayList<Double> pixels = new ArrayList<Double>(image);
                    ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
                    list.add(pixels);
                    this.testData.add(list);
                    pixel = 0;
                    image.clear();
               }

          }
     }

     public void setTrainingLabelData(String filename) throws IOException {
          File file = new File(filename);
          byte[] fileContent = Files.readAllBytes(file.toPath());

          // label data starts at byte 8 of the file
          for (int i = 8; i < fileContent.length; i++) {
               ArrayList<Double> outputs = new ArrayList<Double>();
               for (int k = 0; k < 10; k++)
                    outputs.add(0.0);

               int index = fileContent[i];
               outputs.set(index, 1.0);
               this.trainingData.get(i - 8).add(outputs);
          }

     }

     public void setTestLabelData(String filename) throws IOException {
          File file = new File(filename);
          byte[] fileContent = Files.readAllBytes(file.toPath());

          // label data starts at byte 8 of the file
          for (int i = 8; i < fileContent.length; i++) {
               ArrayList<Double> outputs = new ArrayList<Double>();
               for (int k = 0; k < 10; k++)
                    outputs.add(0.0);

               int index = fileContent[i];
               outputs.set(index, 1.0);
               this.testData.get(i - 8).add(outputs);
          }

     }

}
