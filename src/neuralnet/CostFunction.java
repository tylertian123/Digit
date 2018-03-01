package neuralnet;

public interface CostFunction {
	public double costDerivative(double y, double a);
}
