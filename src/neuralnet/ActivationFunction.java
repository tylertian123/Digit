package neuralnet;

/*
 * The activation function for a neural network
 * Implement this and pass into the constructor of a neural network to use
 */
public interface ActivationFunction {
	//The activation function
	//E.g. sigmoid
	public double activation(double z);
	//The derivative of the activation function
	public double activationDerivative(double z);
	//The activation code
	public byte getCode();
}
