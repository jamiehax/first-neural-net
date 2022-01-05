import org.ejml.equation.Equation;
import org.ejml.simple.SimpleMatrix;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

public class Network {

	/*
	 * DATA FIELDS OF NETWORK: 
	 * numLayers: number of layers in network 
	 * sizes: array of number of neurons in each layer (layer = index) 
	 * biases: list of l (n, 1) matrices for the n neurons in the layer l, starting at 
	 * networks second layer, as no biases should be set for input layer 
	 * weights: list of numLayers - 1 (n(l), n(l - 1)) matrices for each layer, starting 
	 * at the second layer, with dimensions of the number of neurons in the layer x the number 
	 * of neurons in the previous layer.
	 * 
	 * the 'best' version of each data field is used to store the networks best performing
	 * setup so that it can be saved after training is over
	 */

	int numLayers;
	int[] sizes;
	ArrayList<SimpleMatrix> biases = new ArrayList<SimpleMatrix>();
	ArrayList<SimpleMatrix> weights = new ArrayList<SimpleMatrix>();
	double percentCorrect;
	
	int bestNumLayers;
	int[] bestSizes;
	ArrayList<SimpleMatrix> bestBiases = new ArrayList<SimpleMatrix>();
	ArrayList<SimpleMatrix> bestWeights = new ArrayList<SimpleMatrix>();
	double bestPercentCorrect;

	// loads the network with the values from the passed file
	public Network(String filename) throws IOException {
		loadNetwork(filename);
	}
	
	// creates a new network with random weights / biases
	public Network(int[] sizes) {
		this.sizes = sizes;
		numLayers = sizes.length;
		percentCorrect = 0.0;
		bestPercentCorrect = 0.0;

		// initialize weights and biases randomly (no biases for input layer)
		for (int layer = 1; layer < numLayers; layer++) {
			SimpleMatrix biasMatrix = new SimpleMatrix(sizes[layer], 1);
			biasMatrix.setColumn(0, 0, getGaussianList(sizes[layer]));
			biases.add(biasMatrix);
		}

		for (int layer = 0; layer < numLayers - 1; layer++) {
			SimpleMatrix weightMatrix = new SimpleMatrix(sizes[layer + 1], sizes[layer]);

			// set all rows to random gaussian numbers
			for (int row = 0; row < weightMatrix.numRows(); row++)
				weightMatrix.setRow(row, 0, getGaussianList(sizes[layer]));

			weights.add(weightMatrix);
		}

	}

	// return output of network, given (n, 1) matrix inputs
	public SimpleMatrix feedFoward(SimpleMatrix inputs) {
		for (int layer = 0; layer < biases.size(); layer++)
			inputs = sigmoid((weights.get(layer).mult(inputs)).plus(biases.get(layer)));

		return inputs;
	}

	/*
	 * perform stochastic gradient descent on the network with mini-batches from the
	 * training data. trainingData is a list training examples which are stored as 2
	 * element lists, with the inputs to the network as index 0 and the expected
	 * outputs as index 1. test data is in same format and provided to check networks progress
	 */
	public void gradientDescent(ArrayList<ArrayList<SimpleMatrix>> trainingData, int epochs, int miniBatchSize,
			double learningRate, ArrayList<ArrayList<SimpleMatrix>> testData) {
		for (int epoch = 0; epoch < epochs; epoch++) {
			Collections.shuffle(trainingData);
			for (int batch = 0; batch < trainingData.size() / miniBatchSize; batch++) {
				ArrayList<ArrayList<SimpleMatrix>> miniBatch = new ArrayList<ArrayList<SimpleMatrix>>(
						trainingData.subList(batch * miniBatchSize, (batch * miniBatchSize) + miniBatchSize));
				updateMiniBatch(miniBatch, learningRate);
			}
			
			// update percent correct
			int numCorrect = selfEvaluate(testData);
			System.out.println(String.format("Epoch %d: %d / %d", epoch, numCorrect, testData.size()));
			double dCorrect = numCorrect;
			percentCorrect = dCorrect / testData.size();
			
			// store network values if this network performed better
			if (percentCorrect > bestPercentCorrect) {
				bestNumLayers = numLayers;
				bestSizes = sizes;
			    bestPercentCorrect = percentCorrect;
			    
			    for (int layer = 0; layer < biases.size(); layer++) {
			    	SimpleMatrix biasMatrix = new SimpleMatrix(biases.get(layer));
			    	bestBiases.add(biasMatrix);
			    }
			    for (int layer = 0; layer < weights.size(); layer++) {
			    	SimpleMatrix weightMatrix = new SimpleMatrix(weights.get(layer));
			    	bestWeights.add(weightMatrix);
			    }
			}
		}
	}

	/*
	 * update the networks weights and biases based on the results from backprop of
	 * each single batch miniBatch is a list of training data with each element of
	 * the list being a length 2 list with inputs at index 0 and expected outputs at
	 * index 1
	 */
	public void updateMiniBatch(ArrayList<ArrayList<SimpleMatrix>> miniBatch, double learningRate) {

		// create zeroed versions of weights and biases
		ArrayList<SimpleMatrix> nablaB = new ArrayList<SimpleMatrix>();
		ArrayList<SimpleMatrix> nablaW = new ArrayList<SimpleMatrix>();
		for (int layer = 0; layer < numLayers - 1; layer++) {
			SimpleMatrix zeroBias = biases.get(layer).copy();
			zeroBias.zero();
			nablaB.add(zeroBias);
			SimpleMatrix zeroWeights = weights.get(layer).copy();
			zeroWeights.zero();
			nablaW.add(zeroWeights);
		}

		// calculate neurons error's using backpropagation
		for (int batch = 0; batch < miniBatch.size(); batch++) {
			SimpleMatrix inputs = miniBatch.get(batch).get(0);
			SimpleMatrix outputs = miniBatch.get(batch).get(1);

			ArrayList<ArrayList<SimpleMatrix>> gradients = backprop(inputs, outputs);
			ArrayList<SimpleMatrix> deltaNablaB = gradients.get(0);
			ArrayList<SimpleMatrix> deltaNablaW = gradients.get(1);

			for (int layer = 0; layer < numLayers - 1; layer++) {
				nablaB.set(layer, nablaB.get(layer).plus(deltaNablaB.get(layer)));
				nablaW.set(layer, nablaW.get(layer).plus(deltaNablaW.get(layer)));
			}
		}

		// update the networks weights and biases
		for (int layer = 0; layer < numLayers - 1; layer++) {
			SimpleMatrix newBiases = biases.get(layer)
					.minus((nablaB.get(layer).scale(learningRate / miniBatch.size())));
			biases.set(layer, newBiases);
			SimpleMatrix newWeights = weights.get(layer)
					.minus((nablaW.get(layer).scale(learningRate / miniBatch.size())));
			weights.set(layer, newWeights);
		}

	}

	public ArrayList<ArrayList<SimpleMatrix>> backprop(SimpleMatrix inputs, SimpleMatrix outputs) {
		// create zeroed versions of weights and biases
		ArrayList<SimpleMatrix> nablaB = new ArrayList<SimpleMatrix>();
		ArrayList<SimpleMatrix> nablaW = new ArrayList<SimpleMatrix>();
		for (int layer = 0; layer < numLayers - 1; layer++) {
			SimpleMatrix zeroBias = biases.get(layer).copy();
			zeroBias.zero();
			nablaB.add(zeroBias);
			SimpleMatrix zeroWeights = weights.get(layer).copy();
			zeroWeights.zero();
			nablaW.add(zeroWeights);
		}

		// feedfoward the inputs through the network
		SimpleMatrix activation = inputs;
		ArrayList<SimpleMatrix> activations = new ArrayList<SimpleMatrix>();
		activations.add(activation);
		ArrayList<SimpleMatrix> zs = new ArrayList<SimpleMatrix>();
		for (int layer = 0; layer < numLayers - 1; layer++) {
			SimpleMatrix z = (weights.get(layer).mult(activation)).plus(biases.get(layer));
			zs.add(z);
			activation = sigmoid(z);
			activations.add(activation);
		}

		// output error
		SimpleMatrix delta = (costDerivative(activations.get(activations.size() - 1), outputs))
				.elementMult(sigmoidPrime(zs.get(zs.size() - 1)));
		nablaB.set(nablaB.size() - 1, delta);
		nablaW.set(nablaW.size() - 1, delta.kron(activations.get(activations.size() - 2).transpose()));

		// backpropogate the error
		for (int layer = numLayers - 2; layer > 0; layer--) {
			SimpleMatrix z = zs.get(layer - 1);
			SimpleMatrix sp = sigmoidPrime(z);

			SimpleMatrix dot = new SimpleMatrix(weights.get(layer).transpose().numRows(), 1);
			for (int row = 0; row < weights.get(layer).transpose().numRows(); row++) {
				SimpleMatrix rowVec = weights.get(layer).transpose().extractVector(true, row);
				SimpleMatrix deltVec = delta.transpose();
				dot.set(row, 0, deltVec.dot(rowVec));
			}
			delta = dot.elementMult(sp);
			nablaB.set(layer - 1, delta);
			nablaW.set(layer - 1, delta.kron(activations.get(layer - 1).transpose()));
		}
		ArrayList<ArrayList<SimpleMatrix>> ret = new ArrayList<ArrayList<SimpleMatrix>>();
		ret.add(nablaB);
		ret.add(nablaW);
		return ret;
	}

	/*
	 * return number of correct classifications given list of test data, with each
	 * element of the list being a length 2 list with inputs at index 0 and expected
	 * outputs at index 1
	 */
	public int selfEvaluate(ArrayList<ArrayList<SimpleMatrix>> testData) {
		int numCorrect = 0;
		for (int test = 0; test < testData.size(); test++) {
			SimpleMatrix results = feedFoward(testData.get(test).get(0));
			SimpleMatrix expectedOutputs = testData.get(test).get(1);

			int expectedMaxIndex = 0;
			for (int i = 0; i < expectedOutputs.numRows(); i++) {
				if (expectedOutputs.get(i, 0) > expectedOutputs.get(expectedMaxIndex, 0))
					expectedMaxIndex = i;
			}
			int resultsMaxIndex = 0;
			for (int i = 0; i < results.numRows(); i++) {
				if (results.get(i, 0) > results.get(resultsMaxIndex, 0))
					resultsMaxIndex = i;
			}
			if (resultsMaxIndex == expectedMaxIndex)
				numCorrect++;
		}
		return numCorrect;
	}
	
	// return networks output for given input
	public int indetify(SimpleMatrix inputs) {
		SimpleMatrix results = feedFoward(inputs);
		int resultsMaxIndex = 0;
		for (int i = 0; i < results.numRows(); i++) {
			if (results.get(i, 0) > results.get(resultsMaxIndex, 0))
				resultsMaxIndex = i;
		}
		return resultsMaxIndex;
	}
	
	// return networks output for random test input as length 2 list
	// index 0 is networks output, index 1 is expected output
	public int[] indetify(ArrayList<ArrayList<SimpleMatrix>> testData) {
		Random rando = new Random();
		int test = rando.nextInt(testData.size());
		SimpleMatrix results = feedFoward(testData.get(test).get(0));
		SimpleMatrix expectedOutputs = testData.get(test).get(1);
		
		int expectedMaxIndex = 0;
		for (int i = 0; i < expectedOutputs.numRows(); i++) {
			if (expectedOutputs.get(i, 0) > expectedOutputs.get(expectedMaxIndex, 0))
				expectedMaxIndex = i;
		}
		int resultsMaxIndex = 0;
		for (int i = 0; i < results.numRows(); i++) {
			if (results.get(i, 0) > results.get(resultsMaxIndex, 0))
				resultsMaxIndex = i;
		}
		int[] ret = new int[2];
		ret[0] = resultsMaxIndex;
		ret[1] = expectedMaxIndex;
		return ret;
	}

	public int identify(SimpleMatrix inputs) {
		SimpleMatrix outputs = feedFoward(inputs);
		int outputsMaxIndex = 0;
		for (int row = 0; row < outputs.numRows(); row++) {
			if (outputs.get(row, 0) > outputs.get(outputsMaxIndex, 0))
				outputsMaxIndex = row;
		}
		return outputsMaxIndex;
	}

	// return deriviative of the quadratic cost function
	public static SimpleMatrix costDerivative(SimpleMatrix outputActivations, SimpleMatrix expectedOutputs) {
		return outputActivations.minus(expectedOutputs);
	}

	// return derivative of sigmoid function
	public static SimpleMatrix sigmoidPrime(SimpleMatrix z) {
		return sigmoid(z).elementMult(sigmoid(z).negative().plus(1.0));
	}

	// return matrix resulting from applying sigmoid element wise to passed matrix
	public static SimpleMatrix sigmoid(SimpleMatrix z) {
		Equation eq = new Equation();
		eq.alias(z, "z");
		eq.process("r = 1.0 / (1.0 + exp(-z))");
		SimpleMatrix r = eq.lookupSimple("r");
		return r;
	}

	// return x length list of random gaussian numbers
	public static double[] getGaussianList(int x) {
		Random rando = new Random();
		double[] list = new double[x];
		for (int i = 0; i < x; i++) {
			list[i] = rando.nextGaussian();
		}
		return list;
	}

	
	
	
	/*  
	 * =============================================
	 * 		LOADING / SAVING NETWORK METHODS
	 * =============================================
	 * 
	 * file format:
	 * line 1: percent correct
	 * line 2: networks layer sizes 
	 * line 3: networks biases
	 * line 4: networks weights
	 */
	
	// saves the best network to the passed filename
	public void saveNetwork(String filename) throws IOException {
		File outputFile = new File(filename);
		outputFile.createNewFile();
		FileWriter fw = new FileWriter(filename);

		// write percentCorrect
		fw.write(String.format("%.17f", bestPercentCorrect));
		fw.write("\n");
		
		// write layer sizes
		String sizesString = "";
		for (int layer = 0; layer < bestSizes.length; layer++) {
			sizesString = sizesString.concat(String.format("%d, ", bestSizes[layer]));
		}
		fw.write(sizesString);
		fw.write("\n");

		// write biases
		for (int layer = 0; layer < bestNumLayers - 1; layer++) {
			for (int row = 0; row < bestBiases.get(layer).numRows(); row++) {
				fw.write(String.format("%.17f, ", bestBiases.get(layer).get(row, 0)));
			}
		}
		fw.write("\n");

		// write weights
		for (int layer = 0; layer < bestNumLayers - 1; layer++) {
			for (int row = 0; row < bestWeights.get(layer).numRows(); row++) {
				for (int col = 0; col < bestWeights.get(layer).numCols(); col++) {
					fw.write(String.format("%.17f, ", bestWeights.get(layer).get(row, col)));
				}
			}
		}
		fw.write("\n");
		fw.close();

	}

	// sets the networks weights / biases to be the ones from the file
	public void loadNetwork(String filename) throws IOException {
		File inputFile = new File(filename);
		inputFile.createNewFile();
		Scanner fr = new Scanner(inputFile);

		String sizesString = "";
		String biasesString = "";
		String weightsString = "";
		ArrayList<Integer> sizesList = new ArrayList<Integer>();
		ArrayList<SimpleMatrix> biasesList = new ArrayList<SimpleMatrix>();
		ArrayList<SimpleMatrix> weightsList = new ArrayList<SimpleMatrix>();
		int line = 0;
		while (fr.hasNextLine()) {
			
			// set percent correct
			if (line == 0) {
				String percentCorrectString = fr.nextLine();
				double percentCorrectParse = Double.parseDouble(percentCorrectString);
				percentCorrect = percentCorrectParse;
				bestPercentCorrect = percentCorrectParse;
			}
			
			// parse sizes string to set sizes
			else if (line == 1) {
				sizesString = fr.nextLine();
				String[] s = sizesString.split(", ");
				for (int i = 0; i < s.length; i++)
					sizesList.add(Integer.parseInt(s[i]));
			}

			else if (line == 2)
				biasesString = biasesString.concat(fr.nextLine());

			else
				weightsString = weightsString.concat(fr.nextLine());

			line++;
		}
		
		// create biases matrix list
		for (int layer = 1; layer < sizesList.size(); layer++) {
			SimpleMatrix biasMatrix = new SimpleMatrix(sizesList.get(layer), 1);
			String[] s = biasesString.split(", ");
			int i = 0;
			for (int row = 0; row < sizesList.get(layer); row++) {
				Double biasVal = Double.parseDouble(s[i]);
				biasMatrix.set(row, 0, biasVal);
				i++;
			}
			biasesList.add(biasMatrix);
		}
		
		// create weights matrix list
		int i = 0;
		for (int layer = 0; layer < sizesList.size() - 1; layer++) {
			SimpleMatrix weightMatrix = new SimpleMatrix(sizesList.get(layer + 1), sizesList.get(layer));
			String[] s = weightsString.split(", ");
			for (int row = 0; row < weightMatrix.numRows(); row++) {
				for (int col = 0; col < weightMatrix.numCols(); col++) {
					Double biasVal = Double.parseDouble(s[i]);
					weightMatrix.set(row, col, biasVal);
					i++;
				}
			}
			weightsList.add(weightMatrix);
		}

		numLayers = sizesList.size();
		bestNumLayers = sizesList.size();
		this.sizes = new int[sizesList.size()];
		this.bestSizes = new int[sizesList.size()];
		for (int k = 0; k < sizesList.size(); k++) {
			this.sizes[k] = sizesList.get(k);
			this.bestSizes[k] = sizesList.get(k);
		}

		biases = biasesList;
		bestBiases = biasesList;
		weights = weightsList;
		bestWeights = weightsList;
	}

}
