package neuralnet.core;

/**
 * The cost function for a neural network.<br>
 * Implement this class and put into a neural network to use.
 */
public interface CostFunction {
	/**
	 * The partial derivative of this cost function with respect to the output of a neuron in the output layer.
	 * @param y - The "expected" output of a neuron
	 * @param a - The actual output of that neuron
	 * @return The rate of change of the cost at that point
	 */
	public double costDerivative(double y, double a);
	/**
	 * The code of this cost function.<br>
	 * Each cost function has a unique code for it to be identified.
	 * This value is used when saving and loading neural networks; it is saved in the file.
	 * The ClassificationNeuralNetwork class keeps a list of these functions to look up the correct one when loading.
	 * For custom cost functions not included in the list, loading a network causes the cost function
	 * of that particular network to be set to <em>null</em>, and thus it <b>must</b> be set later <b>manually</b>
	 * through the setCostFunction() method.
	 * @return The code of this cost function
	 */
	public byte getCode();
}
