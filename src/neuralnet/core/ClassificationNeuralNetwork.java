package neuralnet.core;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

/**
 * A neural network that classifies "Classifiables".
 * @param <T> - The type of the objects to be classified by this network. Has to implement Classifiable.
 */
public class ClassificationNeuralNetwork<T extends Classifiable> implements Cloneable {
	@Override
	public Object clone() {
		return new ClassificationNeuralNetwork<T>(this);
	}
	
	//Pre-defined activation and cost functions
	protected static class SigmoidActivation implements ActivationFunction {
		@Override
		public double activation(double z) {
			return 1.0 / (1.0 + Math.exp(-z));
		}
		@Override
		public double activationDerivative(double z) {
			return activation(z) * (1 - activation(z));
		}
		@Override
		public byte getCode() {
			return 0;
		}
	}
	protected static class QuadraticCost implements CostFunction {
		@Override
		public double costDerivative(double y, double a) {
			return (a - y);
		}
		@Override
		public byte getCode() {
			return 0;
		}
	}
	protected static class CrossEntropySigmoidCost implements CostFunction {
		@Override
		public double costDerivative(double y, double a) {
			return (1 - y) / (1 - a) - y / a;
		}
		@Override
		public byte getCode() {
			return 1;
		}
	}
	
	/**
	 * Sigmoid activation function.
	 */
	public static final ActivationFunction SIGMOID_ACTIVATION = new SigmoidActivation();
	/**
	 * Simple quadratic cost function.
	 */
	public static final CostFunction QUADRATIC_COST = new QuadraticCost();
	/**
	 * Sigmoid cross-entropy cost function.
	 */
	public static final CostFunction CROSSENTROPY_SIGMOID_COST = new CrossEntropySigmoidCost();
	
	public static final byte SAVE_FORMAT_VER = 0x01;
	protected static final ActivationFunction[] ACTIVATION_LIST = new ActivationFunction[] {
			SIGMOID_ACTIVATION
	};
	protected static final CostFunction[] COST_LIST	= new CostFunction[] {
			QUADRATIC_COST,
			CROSSENTROPY_SIGMOID_COST
	};
	/*
	 * Although the input layer does not have weights and biases, space is still allocated for them
	 * so the indices are less confusing
	 */
	protected int layers;
	//the number of neurons in each layer
	protected int[] neuronCounts;
	//most number of neurons in a layer
	protected int neuronMax;
	//bias of each neuron, from layer and number
	//e.g. biases[1][1] is the bias of the 2nd neuron in the 1st hidden layer
	protected double[][] biases;
	//weight of each connection, from layer, number and number
	//e.g. weights[1][1][2] is the weight of the connection between
	//the 2nd neuron in the 1st hidden layer and the 3rd neuron in the layer before it
	protected double[][][] weights;
	//The activation and cost functions
	protected ActivationFunction activationFunction;
	protected CostFunction costFunction;
	
	protected static int getMax(int[] arr) {
		int max = 0;
		for(int i : arr) {
			if(i > max) {
				max = i;
			}
		}
		return max;
	}
	protected static double dotProduct(double[] a, double[] b) {
		double result = 0;
		for(int i = 0; i < a.length; i ++) {
			result += a[i] * b[i];
		}
		return result;
	}
	protected static double dotProduct(final double[] a, final double[] b, final int len) {
		double result = 0;
		for(int i = 0; i < len; i ++) {
			result += a[i] * b[i];
		}
		return result;
	}
	
	//Creates a non-rectangular array
	protected static double[][] createJaggedArray(int[] lengths) {
		double[][] arr = new double[lengths.length][];
		for(int i = 0; i < arr.length; i ++)
			arr[i] = new double[lengths[i]];
		return arr;
	}
	//Used to create a jagged array of rectangular arrays
	protected static double[][][] createJaggedArray3d(int[] widths, int[] heights) {
		if(widths.length != heights.length)
			throw new IllegalArgumentException("Dimension arrays provided are not of the same length");
		double[][][] arr = new double[widths.length][][];
		for(int i = 0; i < arr.length; i ++)
			arr[i] = new double[widths[i]][heights[i]];
		return arr;
	}
	
	/**
	 * Creates a 3-dimensional array in the shape of the weights matrix.
	 * This comes in useful not just for storing weights, but also other related things,
	 * such as the gradient for the weights.
	 * @return A 3-dimensional array in the shape of the weights matrix.
	 */
	protected double[][][] createWeightsArray() {
		int[] sizes2 = new int[layers];
		for(int i = 0; i < layers - 1; i ++)
			sizes2[i + 1] = neuronCounts[i];
		//First layer contains no weights, so each neuron is connected to 0 others
		sizes2[0] = 0;
		return createJaggedArray3d(neuronCounts, sizes2);
	}
	/**
	 * Creates a 2-dimensional array in the shape of the biases matrix.
	 * This comes in useful not just for storing biases, but also other related things,
	 * such as the gradient for the biases.
	 * @return A 2-dimensional array in the shape of the biases matrix.
	 */
	protected double[][] createBiasesArray() {
		return createJaggedArray(neuronCounts);
	}
	
	/**
	 * Creates a new neural network with the specified structure, activation function and cost function.
	 * The weights are initialized to a gaussian with mean 0 and standard deviation 1/sqrt(Nin), 
	 * where Nin is the number of weights attached to the same neuron as the weight in question.
	 * Biases are initialized to a gaussian with mean 0 and standard deviation 1.
	 * @param neuronCounts - The structure of the neural network. Each element represents the number of neurons in
	 * that layer
	 * @param activation - The activation function
	 * @param cost - The cost function
	 */
	public ClassificationNeuralNetwork(int[] neuronCounts, ActivationFunction activation, CostFunction cost) {
		Random r = new Random();
		activationFunction = activation;
		costFunction = cost;
		this.layers = neuronCounts.length;
		this.neuronCounts = neuronCounts;
		neuronMax = getMax(neuronCounts);
		//The first (input) layer has no biases, but memory is still allocated to keep indices simple
		this.biases = createBiasesArray();
		this.weights = createWeightsArray();
		//No need to initialize the first layer's weights and biases
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				biases[i][j] = r.nextGaussian();
				for(int k = 0; k < neuronCounts[i - 1]; k ++)
					//Initialize each weight to a gaussian with mean 0 and standard deviation 1/sqrt(Nin)
					//Where Nin the number of weights the neuron has
					weights[i][j][k] = r.nextGaussian() / Math.sqrt(neuronCounts[i - 1]);
			}
		}
	}
	/**
	 * Loads a neural network from a file.
	 * Note that with the current file format, each pre-defined activation and cost function has its unique code.
	 * This code is saved with the file and used to find the correct function when loading the network.
	 * However, in the case with custom activation and cost functions returning codes that cannot be matched,
	 * the activation and cost functions of the network will be set to <em>null</em>, and thus <b>must</b> be
	 * set later <b>manually</b> with setActivationFunction() and/or setCostFunction().
	 * @param f - The file to load from
	 * @throws IOException If reading the file was not successful
	 * @throws NeuralNetworkException If the format of the file is not supported
	 */
	public ClassificationNeuralNetwork(File f) throws IOException, NeuralNetworkException {
		DataInputStream in = new DataInputStream(new FileInputStream(f));
		byte version = in.readByte();
		switch(version) {
		case 0x01:
		{
			ArrayList<Integer> countsList = new ArrayList<Integer>();
			int count;
			while((count = in.readInt()) != 0) {
				countsList.add(count);
			}
			this.neuronCounts = new int[countsList.size()];
			for(int i = 0; i < neuronCounts.length; i ++)
				neuronCounts[i] = countsList.get(i);
			this.neuronMax = getMax(neuronCounts);
			this.layers = neuronCounts.length;
			byte activationType = in.readByte();
			byte costType = in.readByte();
			
			boolean found = false;
			for(int i = 0; i < ACTIVATION_LIST.length; i ++) {
				if(ACTIVATION_LIST[i].getCode() == activationType) {
					this.activationFunction = ACTIVATION_LIST[i];
					found = true;
					break;
				}
			}
			if(!found) {
				this.activationFunction = null;
			}
			found = false;
			for(int i = 0; i < COST_LIST.length; i ++) {
				if(COST_LIST[i].getCode() == costType) {
					this.costFunction = COST_LIST[i];
					found = true;
					break;
				}
			}
			if(!found) {
				this.costFunction = null;
			}
			
			weights = createWeightsArray();
			biases = createBiasesArray();
			for(int i = 1; i < layers; i ++) {
				for(int j = 0; j < neuronCounts[i]; j ++) {
					for(int k = 0; k < neuronCounts[i - 1]; k ++) {
						weights[i][j][k] = in.readDouble();
					}
				}
			}
			for(int i = 1; i < layers; i ++) {
				for(int j = 0; j < neuronCounts[i]; j ++) {
					biases[i][j] = in.readDouble();
				}
			}
		}
			break;
		default: in.close(); throw new NeuralNetworkException("Unsupported format");
		}
		in.close();
	}
	/**
	 * Constructs a neural network by copying an existing one.
	 * @param otherNet - Another ClassificationNeuralNetwork
	 */
	public ClassificationNeuralNetwork(final ClassificationNeuralNetwork<?> otherNet) {
		copyFrom(otherNet);
	}
	/**
	 * Constructs this object but does not initialize anything. 
	 * Not meant to be used outside of this class.
	 */
	protected ClassificationNeuralNetwork() {
	}
	
	/**
	 * Loads a neural network from file. For details, see ClassificationNeuralNetwork(File f).
	 * @param f - The file to load from
	 * @throws IOException - If reading the file was not successful
	 * @throws NeuralNetworkException - If the format of the file is not supported
	 */
	public void loadFile(File f) throws IOException, NeuralNetworkException {
		this.copyFrom(new ClassificationNeuralNetwork<T>(f));
	}
	/**
	 * Sets this network to be an exact copy of another neural network, changing the structure if necessary.
	 * @param otherNet - The neural network to copy from
	 */
	public void copyFrom(final ClassificationNeuralNetwork<?> otherNet) {
		this.layers = otherNet.layers;
		if(!Arrays.equals(this.neuronCounts, otherNet.neuronCounts)) {
			this.neuronCounts = otherNet.neuronCounts.clone();
		}
		this.neuronMax = otherNet.neuronMax;
		this.activationFunction = otherNet.activationFunction;
		this.costFunction = otherNet.costFunction;
		
		this.weights = otherNet.weights.clone();
		this.biases = otherNet.biases.clone();
	}
	/**
	 * Sets the activation function of this neural network.
	 * @param a - The new activation function
	 */
	public void setActivationFunction(ActivationFunction a) {
		this.activationFunction = a;
	}
	/**
	 * Sets the cost function of this neural network
	 * @param c - The new cost function
	 */
	public void setCostFunction(CostFunction c) {
		this.costFunction = c;
	}
	
	/**
	 * Feedforwards the network with a specified input and returns the "classification" of that input.
	 * The classification is generated by calling the toClassification() method of the input with the output of the network.
	 * @param obj - The input
	 * @return The "classification" of the input
	 */
	public Object classify(T obj) {
		double[] lastActivations = new double[neuronMax];
		double[] input = obj.asNeuralNetworkInput();
		for(int i = 0; i < input.length; i ++) {
			lastActivations[i] = input[i];
		}
		double[] activations = new double[neuronMax];
		
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				activations[j] = activationFunction.activation(
						dotProduct(lastActivations, weights[i][j], neuronCounts[i - 1])
						+ biases[i][j]);
			}
			lastActivations = activations.clone();
		}
		return obj.toClassification(lastActivations);
	}
	/**
	 * Classifies each element of the input array with classify() and returns the number of items correctly classified.
	 * @param data - An array containing objects to be classified
	 * @return The number of objects correctly classified
	 */
	public int evaluate(T[] data) {
		int total = 0;
		for(T obj : data)
			if(this.classify(obj).equals(obj.getClassification()))
				total ++;
		return total;
	}
	
	/**
	 * Stochastic gradient descent using L2 regularization.<br>
	 * Equivalent to calling SGD(trainingData, batchSize, learningRate, regularizationConstant, epochs, null, false)
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param epochs - The number of epochs to train for
	 */
	public void SGD(T[] trainingData, int batchSize, double learningRate, double regularizationConstant, int epochs) {
		SGD(trainingData, batchSize, learningRate, regularizationConstant, epochs, null);
	}
	/**
	 * Stochastic gradient descent using L2 regularization. The performance is evaluated and printed to stdout 
	 * for each epoch, unless evalData is null.<br>
	 * Equivalent to calling SGD(trainingData, batchSize, learningRate, regularizationConstant, epochs, evalData, false)
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param epochs - The number of epochs to train for
	 * @param evalData - The data to evaluate the network's performance with
	 */
	public void SGD(T[] trainingData, int batchSize, double learningRate, double regularizationConstant, int epochs, T[] evalData) {
		SGD(trainingData, batchSize, learningRate, regularizationConstant, epochs, evalData, false);
	}
	/**
	 * Stochastic gradient descent using L2 regularization. The performance is evaluated and printed to stdout 
	 * for each epoch, unless evalData is null. A performance graph is also generated in the end if generateGraph is true.
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param epochs - The number of epochs to train for
	 * @param evalData - The data to evaluate the network's performance with
	 * @param generateGraph - If true, generates a classification rate vs time graph.
	 */
	@SuppressWarnings("unchecked")
	public void SGD(T[] trainingData, int batchSize, double learningRate, double regularizationConstant, int epochs, T[] evalData, boolean generateGraph) {
		double maxPercentage = 0.0;
		int maxEpoch = -1;
		double[] percentages = null;
		if(evalData != null) {
			percentages = new double[epochs];
			System.out.println("No Training:\nEvaluating...");
			double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
			System.out.println(percentage + "% correctly classified.");
		}
		for(int epoch = 1; epoch <= epochs; epoch ++) {
			List<T> l = Arrays.asList(trainingData);
			Collections.shuffle(l);
			
			if(evalData != null) {
				System.out.println("Epoch #" + epoch);
				System.out.println("Learning...");
			}
			
			//Separate the shuffled training samples into mini-batches and train with each mini-batch
			for(int i = 0; i < trainingData.length; i += batchSize) {
				List<T> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
				T[] miniBatch = (T[]) new Classifiable[miniBatchList.size()];
				miniBatchList.toArray(miniBatch);
				learnFromMiniBatch(miniBatch, learningRate, regularizationConstant, trainingData.length);
			}
			
			if(evalData != null) {
				System.out.println("Evaluating...");
				double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
				System.out.println(percentage + "% correctly classified.");
				if(percentage > maxPercentage) {
					maxPercentage = percentage;
					maxEpoch = epoch;
				}
				percentages[epoch - 1] = percentage;
			}
		}
		if(evalData != null)
			System.out.printf("Max classification rate: %f%%, reached at Epoch #%d", maxPercentage, maxEpoch);
		if(generateGraph) {
			BufferedImage graph = new BufferedImage(percentages.length * 10, 500, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = (Graphics2D) graph.getGraphics();
			g.setPaint(Color.WHITE);
			g.fillRect(0, 0, percentages.length * 10, 500);
			g.setPaint(Color.RED);
			for(int i = 0; i < percentages.length - 1; i ++) {
				g.drawLine(i * 10, (int) (500 - (percentages[i] * 5)), (i + 1) * 10, (int) (500 - (percentages[i + 1] * 5)));
			}
			try {
				ImageIO.write(graph, "png", new File("classification_rate_progression.png"));
			} 
			catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	/**
	 * Performs stochastic gradient descent with L2 regularization and saves the best network.<br>
	 * After each training epoch, the data of the network is stored as a temporary file that is deleted when the VM exits.
	 * When the training is finished, the network will load and make permanent the temporary file that stores the network with the best
	 * performance compared to all other epochs, even if it might not be the network from the last epoch.
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param epochs - The number of epochs to train for
	 * @param evalData - The data to evaluate the network's performace with. Unlike SGD(), it cannot be null.
	 * @param outFile - The file to save the final network as. Can be null.
	 * @throws IOException If saving the temporary files or the final file is unsuccessful
	 */
	@SuppressWarnings("unchecked")
	public void SGDAndSave(T[] trainingData, int batchSize, double learningRate, double regularizationConstant, int epochs, T[] evalData, File outFile) throws IOException {
		double maxPercentage = 0.0;
		int maxEpoch = -1;
		
		File[] netData = new File[epochs];
		
		System.out.println("No Training:\nEvaluating...");
		double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
		System.out.println(percentage + "% correctly classified.");
		
		for(int epoch = 1; epoch <= epochs; epoch ++) {
			List<T> l = Arrays.asList(trainingData);
			Collections.shuffle(l);

			System.out.println("Epoch #" + epoch);
			System.out.println("Learning...");
			
			//Separate the shuffled training samples into mini-batches and train with each mini-batch
			for(int i = 0; i < trainingData.length; i += batchSize) {
				List<T> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
				T[] miniBatch = (T[]) new Classifiable[miniBatchList.size()];
				miniBatchList.toArray(miniBatch);
				learnFromMiniBatch(miniBatch, learningRate, regularizationConstant, trainingData.length);
			}
			
			System.out.println("Evaluating...");
			percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
			System.out.println(percentage + "% correctly classified.");
			if(percentage > maxPercentage) {
				maxPercentage = percentage;
				maxEpoch = epoch;
			}
			File f = File.createTempFile("tmpnet", null);
			saveData(f);
			netData[epoch - 1] = f;
			f.deleteOnExit();
		}
		System.out.printf("Max classification rate: %f%%, reached at Epoch #%d", maxPercentage, maxEpoch);
		if(outFile != null) {
			if(outFile.exists())
				outFile.delete();
			Files.copy(netData[maxEpoch - 1].toPath(), outFile.toPath());
		}
		
		try {
			this.loadFile(netData[maxEpoch - 1]);
		} 
		catch (NeuralNetworkException e) {
			System.err.println("Unexpected exception in SGDAndSave(): " + e.toString());
		}
	}
	/**
	 * Performs stochastic gradient descent with L2 regularization, with momentum.<br>
	 * Equivalent to calling MomentumSGD(trainingData, batchSize, learningRate, regularizationConstant, momentumCoefficient, epochs, null)
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param momentumCoefficient - The momentum coefficient (mu)
	 * @param epochs - The number of epochs to train for
	 */
	public void MomentumSGD(T[] trainingData, int batchSize, double learningRate, double regularizationConstant, double momentumCoefficient, int epochs) {
		MomentumSGD(trainingData, batchSize, learningRate, regularizationConstant, momentumCoefficient, epochs, null);
	}
	/**
	 * Performs stochastic gradient descent with L2 regularization, with momentum.<br>
	 * The performance after each epoch is evaluated and printed to stdout if evalData is not null.
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param momentumCoefficient - The momentum coefficient (mu)
	 * @param epochs - The number of epochs to train for
	 * @param evalData - The data to evaluate the network's performance with
	 */
	@SuppressWarnings("unchecked")
	public void MomentumSGD(T[] trainingData, int batchSize, double learningRate, double regularizationConstant, double momentumCoefficient, int epochs, T[] evalData) {
		double[][][] velocity = createWeightsArray();
		double maxPercentage = 0.0;
		int maxEpoch = -1;
		double[] percentages = null;
		if(evalData != null) {
			percentages = new double[epochs];
			System.out.println("No Training:\nEvaluating...");
			double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
			System.out.println(percentage + "% correctly classified.");
		}
		for(int epoch = 1; epoch <= epochs; epoch ++) {
			List<T> l = Arrays.asList(trainingData);
			Collections.shuffle(l);
			
			if(evalData != null) {
				System.out.println("Epoch #" + epoch);
				System.out.println("Learning...");
			}
			
			//Separate the shuffled training samples into mini-batches and train with each mini-batch
			for(int i = 0; i < trainingData.length; i += batchSize) {
				List<T> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
				T[] miniBatch = (T[]) new Classifiable[miniBatchList.size()];
				miniBatchList.toArray(miniBatch);
				learnFromMiniBatch(miniBatch, learningRate, regularizationConstant, trainingData.length, velocity, momentumCoefficient);
			}
			
			if(evalData != null) {
				System.out.println("Evaluating...");
				double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
				System.out.println(percentage + "% correctly classified.");
				if(percentage > maxPercentage) {
					maxPercentage = percentage;
					maxEpoch = epoch;
				}
				percentages[epoch - 1] = percentage;
			}
		}
		if(evalData != null)
			System.out.printf("Max classification rate: %f%%, reached at Epoch #%d", maxPercentage, maxEpoch);
	}
	/**
	 * Performs stochastic gradient descent with L2 regularization with a changing/scheduled learning rate.
	 * For each "cycle", the network learns and is evaluated. If there is no improvement for a certain
	 * number of epochs, the learning rate is multiplied by a factor and the next cycle starts.
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param initLearningRate - The initial learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param evalData - The data to evaluate the network's performance with. Cannot be null.
	 * @param schedule - The number of epochs with no performance increase before moving to the next cycle
	 * @param newRateFactor - The scalar the learning rate is multiplied by for each cycle
	 * @param cycles - The number of cycles to continue for
	 */
	@SuppressWarnings("unchecked")
	public void SGDScheduledEta(T[] trainingData, int batchSize, double initLearningRate, double regularizationConstant, T[] evalData, int schedule, double newRateFactor, int cycles) {
		int epoch = 1;
		double eta = initLearningRate;
		int lastMaxEpoch = 1;
		double lastMaxRate = 0.0;
		
		double allTimeBest = 0.0;
		int bestCycle = -1;
		int bestEpoch = -1;
		
		for(int cycle = 1; cycle <= cycles; cycle ++) {
			System.out.printf("Cycle #%d (eta = %f):\n", cycle, eta);
			while(true) {
				List<T> l = Arrays.asList(trainingData);
				Collections.shuffle(l);
				System.out.printf("Cycle #%d, Epoch #%d:\nLearning...\n", cycle, epoch);
				
				for(int i = 0; i < trainingData.length; i += batchSize) {
					List<T> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
					T[] miniBatch = (T[]) new Classifiable[miniBatchList.size()];
					miniBatchList.toArray(miniBatch);
					learnFromMiniBatch(miniBatch, eta, regularizationConstant, trainingData.length);
				}
				
				double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
				System.out.printf("%f%% correctly classified.\n", percentage);
				if(percentage > allTimeBest) {
					allTimeBest = percentage;
					bestCycle = cycle;
					bestEpoch = epoch;
				}
				if(percentage > lastMaxRate) {
					lastMaxRate = percentage;
					lastMaxEpoch = epoch;
				}
				else {
					if(epoch - lastMaxEpoch >= schedule) {
						lastMaxEpoch = 1;
						lastMaxRate = 0.0;
						epoch = 1;
						break;
					}
				}
				epoch ++;
			}
			
			eta *= newRateFactor;
		}
		System.out.printf("Training finished.\nAll-time best was %f%% at Cycle #%d, Epoch #%d.\n", allTimeBest, bestCycle, bestEpoch);
	}
	/**
	 * Performs stochastic gradient descent with L2 regularization and momentum, with a changing/scheduled learning rate.
	 * For each "cycle", the network learns and is evaluated. If there is no improvement for a certain
	 * number of epochs, the learning rate is multiplied by a factor and the next cycle starts.
	 * @param trainingData - The training data
	 * @param batchSize - The size of each mini-batch
	 * @param initLearningRate - The initial learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param momentumCoefficient - The momentum coefficient (mu)
	 * @param evalData - The data to evaluate the network's performance with. Cannot be null.
	 * @param schedule - The number of epochs with no performance increase before moving to the next cycle
	 * @param newRateFactor - The scalar the learning rate is multiplied by for each cycle
	 * @param cycles - The number of cycles to continue for
	 */
	@SuppressWarnings("unchecked")
	public void SGDScheduledEta(T[] trainingData, int batchSize, double initLearningRate, double regularizationConstant, double momentumCoefficient, T[] evalData, int schedule, double newRateFactor, int cycles) {
		double[][][] velocity = createWeightsArray();
		
		int epoch = 1;
		double eta = initLearningRate;
		int lastMaxEpoch = 1;
		double lastMaxRate = 0.0;
		
		double allTimeBest = 0.0;
		int bestCycle = -1;
		int bestEpoch = -1;
		
		for(int cycle = 1; cycle <= cycles; cycle ++) {
			System.out.printf("Cycle #%d (eta = %f):\n", cycle, eta);
			while(true) {
				List<T> l = Arrays.asList(trainingData);
				Collections.shuffle(l);
				System.out.printf("Cycle #%d, Epoch #%d:\nLearning...\n", cycle, epoch);
				
				for(int i = 0; i < trainingData.length; i += batchSize) {
					List<T> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
					T[] miniBatch = (T[]) new Classifiable[miniBatchList.size()];
					miniBatchList.toArray(miniBatch);
					learnFromMiniBatch(miniBatch, eta, regularizationConstant, trainingData.length, velocity, momentumCoefficient);
				}
				
				double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
				System.out.printf("%f%% correctly classified.\n", percentage);
				if(percentage > allTimeBest) {
					allTimeBest = percentage;
					bestCycle = cycle;
					bestEpoch = epoch;
				}
				if(percentage > lastMaxRate) {
					lastMaxRate = percentage;
					lastMaxEpoch = epoch;
				}
				else {
					if(epoch - lastMaxEpoch >= schedule) {
						lastMaxEpoch = 1;
						lastMaxRate = 0.0;
						epoch = 1;
						break;
					}
				}
				epoch ++;
			}
			
			eta *= newRateFactor;
		}
		System.out.printf("Training finished.\nAll-time best was %f%% at Cycle #%d, Epoch #%d.\n", allTimeBest, bestCycle, bestEpoch);
	}
	
	/**
	 * Applies a single step of gradient descent with L2 regularization. <br>
	 * Equivalent to calling learnFromMiniBatch(miniBatch, learningRate, regularizationConstant, dataSize, null, 0)
	 * @param miniBatch - The mini-batch to learn from
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param dataSize - The total size of the training data, for L2 regularization.
	 */
	protected void learnFromMiniBatch(T[] miniBatch, double learningRate, double regularizationConstant, int dataSize) {
		learnFromMiniBatch(miniBatch, learningRate, regularizationConstant, dataSize, null, 0);
	}
	/**
	 * Applies a single step of gradient descent with L2 regularization and momentum. <br>
	 * @param miniBatch - The mini-batch to learn from
	 * @param learningRate - The learning rate (eta)
	 * @param regularizationConstant - The regularization constant (lambda)
	 * @param dataSize - The total size of the training data, for L2 regularization.
	 * @param velocity - A 3-dimensional array in the shape of the weights matrix. Each element represent the "velocity"
	 * of that weight. This array is updated in the process. If null, momentum is not applied.
	 * @param momentumCoefficient - The momentum coefficient (mu)
	 */
	protected void learnFromMiniBatch(T[] miniBatch, double learningRate, double regularizationConstant, int dataSize, double[][][] velocity, double momentumCoefficient) {
		//The size of the batch
		//Only incremented for values that are non-null
		int batchSize = 0;
		//Summed dC/db and dC/dw
		double[][] biasDerivativesTotal = createBiasesArray();
		double[][][] weightDerivativesTotal = createWeightsArray();
		
		for(T trainingSample : miniBatch) {
			if(trainingSample != null) {
				batchSize ++;
				//Expected output
				double[] y = trainingSample.generateExpectedOutput();
				//Activations
				double[][] a = createBiasesArray();
				//Weighted sums
				double[][] z = createBiasesArray();
				//Errors
				double[][] e = createBiasesArray();
				
				//Feedforward
				a[0] = trainingSample.asNeuralNetworkInput();
				for(int i = 1; i < layers; i ++) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						//Dot product of last layer's activations with this layer's weights added to the bias 
						z[i][j] = dotProduct(a[i - 1], weights[i][j]) + biases[i][j];
						//Put through the activation function
						a[i][j] = activationFunction.activation(z[i][j]);
					}
				}
				//Calculate error for output layer
				for(int j = 0; j < neuronCounts[layers - 1]; j ++) {
					//The error for a neuron in the output layer =
					//activation'(z) * dC/da
					e[layers - 1][j] = activationFunction.activationDerivative(z[layers - 1][j])
							* costFunction.costDerivative(y[j], a[layers - 1][j]);
				}
				//Backpropagate
				for(int i = layers - 2; i >= 0; i --) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						//Perform the sigma
						double err = 0.0;
						for(int k = 0; k < neuronCounts[i + 1]; k ++) {
							//The error of a neuron in the next layer * the weight connecting them
							err += e[i + 1][k] * weights[i + 1][k][j];
						}
						//dC/da * da/dz = dC/dz
						err *= activationFunction.activationDerivative(z[i][j]);
						e[i][j] = err;
					}
				}
				//Calculate the weight and bias derivatives and add to total
				//Skip input layer
				for(int i = 1; i < layers; i ++) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						biasDerivativesTotal[i][j] += e[i][j];
						for(int k = 0; k < neuronCounts[i - 1]; k ++) {
							weightDerivativesTotal[i][j][k] = e[i][j] * a[i - 1][k];
						}
					}
				}
			}
		}
		
		//Divide to take the average
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				biasDerivativesTotal[i][j] /= (double)batchSize;
				for(int k = 0; k < neuronCounts[i - 1]; k ++) {
					weightDerivativesTotal[i][j][k] /= (double)batchSize;
				}
			}
		}
		//Update the new weights and biases
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				//b -> b' = b - eta * gradient
				biases[i][j] = biases[i][j] - learningRate * biasDerivativesTotal[i][j];
				for(int k = 0; k < neuronCounts[i - 1]; k ++) {
					//Calculate momentum if the velocity matrix is not null
					if(velocity != null) {
						//v -> v' = mu * v - eta * gradient
						velocity[i][j][k] = momentumCoefficient * velocity[i][j][k]
								- learningRate * weightDerivativesTotal[i][j][k];
						//w -> w' = w * (1 - (eta * lambda / n)) + v
						weights[i][j][k] = weights[i][j][k] * (1 - learningRate * regularizationConstant / dataSize)
								+ velocity[i][j][k];
					}
					else {
						//w -> w' = w * (1 - (eta * lambda / n)) - eta * gradient
						weights[i][j][k] = weights[i][j][k] * (1 - learningRate * regularizationConstant / dataSize)
								- learningRate * weightDerivativesTotal[i][j][k];
					}
				}
			}
		}
	}
	
	/**
	 * Saves the network's data in a file with the latest format.<br>
	 * <br>
	 * Format for version 0x01:<br>
	 * Version code - 1 byte<br>
	 * Input layer neuron count - 4 bytes<br>
	 * Hidden layer 1 neuron count - 4 bytes<br>
	 * Hidden layer 2 neuron count - 4 bytes<br>
	 * ...<br>
	 * Output layer neuron count - 4 bytes<br>
	 * 0 - 4 bytes<br>
	 * Activation type code - 1 byte<br>
	 * Cost type code - 1 byte<br>
	 * Weight[1][0][0] - 8 bytes<br>
	 * Weight[1][0][1] - 8 bytes<br>
	 * ...<br>
	 * Bias[1][0] - 8 bytes<br>
	 * Bias[1][1] - 8 bytes<br>
	 * ...<br>
	 * 
	 * @param f - The file to save the data in. If it does not exist, a new file will be created. Existing files will be overwritten.
	 * @throws IOException If writing to the file was unsuccessful
	 */
	public void saveData(File f) throws IOException {
		if(!f.exists())
			f.createNewFile();
		DataOutputStream out = new DataOutputStream(new FileOutputStream(f));
		out.writeByte(SAVE_FORMAT_VER);
		
		for(int i = 0; i < layers; i ++)
			out.writeInt(neuronCounts[i]);
		out.writeInt(0);
		
		out.writeByte(activationFunction.getCode());
		out.writeByte(costFunction.getCode());
		
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				for(int k = 0; k < neuronCounts[i - 1]; k ++) {
					out.writeDouble(weights[i][j][k]);
				}
			}
		}
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				out.writeDouble(biases[i][j]);
			}
		}
		
		out.close();
	}
}
