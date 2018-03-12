package neuralnet;

import java.util.HashMap;

public class CompositeClassifier<T extends Classifiable> {
	ClassificationNeuralNetwork<T>[] networks;
	
	@SafeVarargs
	public CompositeClassifier(ClassificationNeuralNetwork<T>... n) {
		networks = n;
	}
	
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
