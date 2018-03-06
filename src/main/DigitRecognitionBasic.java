package main;

import mnist.MNISTLoader;

import java.io.File;

import mnist.MNISTImage;
import neuralnet.ClassificationNeuralNetwork;
import neuralnet.CompositeClassifier;

public class DigitRecognitionBasic {

	public static void main(String[] args) {
		try {
			MNISTImage[] trainingData = MNISTLoader.loadTrainingImages();
			MNISTImage[] evalData = MNISTLoader.loadTestingImages();
			MNISTImage[] validationData = MNISTLoader.loadValidationImages();
			
			/*ClassificationNeuralNetwork<MNISTImage> net = 
					new ClassificationNeuralNetwork<MNISTImage>(new int[] {MNISTImage.PIXEL_COUNT, 100, 10}, 
							ClassificationNeuralNetwork.SIGMOID_ACTIVATION,
							ClassificationNeuralNetwork.CROSSENTROPY_SIGMOID_COST);*/
			//net.SGDScheduledEta(trainingData, 10, 0.5, 5.0, evalData, 3, 0.5, 8);
			//net.SGDAndSave(trainingData, 2, 0.10, 20, 5.0, evalData, new File("sgdscheduled.ann"));
			//net.SGDScheduledEta(trainingData, 1, 0.06, 5.0, evalData, 4, 0.5, 5);
			//net.saveData(new File("sgdscheduledonline.ann"));
			
			//ClassificationNeuralNetwork net = new ClassificationNeuralNetwork(new File("sgdscheduled.ann"));
			//System.out.println(net.evaluate(validationData));
			
			CompositeClassifier<MNISTImage> classifier = new CompositeClassifier<MNISTImage> (
					new ClassificationNeuralNetwork<MNISTImage>(new File("trained networks\\98.18%.ann"))
					//new ClassificationNeuralNetwork<MNISTImage>(new File("trained networks\\98.18%.ann")),
					//new ClassificationNeuralNetwork<MNISTImage>(new File("trained networks\\98.05%.ann")),
					//new ClassificationNeuralNetwork<MNISTImage>(new File("trained networks\\97.76%.ann")),
					//new ClassificationNeuralNetwork<MNISTImage>(new File("trained networks\\97.07%.ann")),
					//new ClassificationNeuralNetwork<MNISTImage>(new File("trained networks\\96.02%.ann"))
					);
			int correct = 0;
			for(MNISTImage img : validationData) {
				if(classifier.classify(img) == img.getClassification())
					correct ++;
			}
			System.out.printf("%f%% correctly classified\n", ((double) correct) / validationData.length * 100);
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
