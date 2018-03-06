package neuralnet;

import java.util.HashMap;

public class CompositeClassifier<T extends Classifiable> {
	ClassificationNeuralNetwork<T>[] networks;
	
	@SafeVarargs
	public CompositeClassifier(ClassificationNeuralNetwork<T>... n) {
		networks = n;
	}
	
	public int classify(T obj) {
		int[] classifications = new int[networks.length];
		for(int i = 0; i < networks.length; i ++)
			classifications[i] = networks[i].classify(obj);
		HashMap<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
		for(int c : classifications) {
			if(occurrences.containsKey(c))
				occurrences.put(c, occurrences.get(c) + 1);
			else
				occurrences.put(c, 1);
		}
		int maxClassification = 0;
		int maxVal = 0;
		for(int o : occurrences.keySet()) {
			if(occurrences.get(o) > maxVal) {
				maxVal = occurrences.get(o);
				maxClassification = o;
			}
		}
		
		return maxClassification;
	}
}
