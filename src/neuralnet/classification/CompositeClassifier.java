package neuralnet.classification;

import java.util.HashMap;

import neuralnet.core.Classifiable;
import neuralnet.core.ClassificationNeuralNetwork;

/**
 * A "classifier" object is one that can "classify" a Classifiable object.<br>
 * The CompositeClassifier class combines multiple ClassificationNeuralNetworks in an attempt to make classification more accurate.
 * @param <T> - The type of object to classify
 */
public class CompositeClassifier<T extends Classifiable> {
	ClassificationNeuralNetwork<T>[] networks;
	
	/**
	 * Creates a new CompositeClassifier with the specified ClassificationNeuralNetworks.
	 * @param n - The networks to classify with
	 */
	@SafeVarargs
	public CompositeClassifier(ClassificationNeuralNetwork<T>... n) {
		networks = n;
	}
	
	/**
	 * "Classifies" an object. This is done by calling classify() on each of the networks that make up this CompositeClassifier
	 * with the object, and returning the result most of the networks agree on. 
	 * If there is a tie, the networks that are first in the array of networks that make up this classifier are favored.
	 * @param obj - The object to classify
	 * @return The "classification" of the object
	 */
	public Object classify(T obj) {
		Object[] classifications = new Object[networks.length];
		for(int i = 0; i < networks.length; i ++)
			classifications[i] = networks[i].classify(obj);
		HashMap<Object, Integer> occurrences = new HashMap<Object, Integer>();
		for(Object c : classifications) {
			if(occurrences.containsKey(c))
				occurrences.put(c, occurrences.get(c) + 1);
			else
				occurrences.put(c, 1);
		}
		Object maxClassification = 0;
		int maxVal = 0;
		for(Object o : occurrences.keySet()) {
			if(occurrences.get(o) > maxVal) {
				maxVal = occurrences.get(o);
				maxClassification = o;
			}
		}
		
		return maxClassification;
	}
}
