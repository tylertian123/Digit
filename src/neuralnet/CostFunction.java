package neuralnet;

/*
 * The cost function for a neural network
 * Implement this class and put into a neural network to use
 */
public interface CostFunction {
	//The derivative of the cost function
	//y represents the expected output of a neuron and a represents the actual activation
	public double costDerivative(double y, double a);
	//The cost code
	public byte getCode();
}
