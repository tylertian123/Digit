package neuralnet;

public interface ActivationFunction {
	public double activation(double z);
	public double activationDerivative(double z);
}
