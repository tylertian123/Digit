package main;

import mnist.MNISTLoader;

import java.io.File;

import mnist.MNISTImage;
import neuralnet.ClassificationNeuralNetwork;

public class DigitRecognitionBasic {

	public static void main(String[] args) {
		try {
			MNISTImage[] trainingData = MNISTLoader.loadTrainingImages();
			MNISTImage[] evalData = MNISTLoader.loadTestingImages();
			MNISTImage[] validationData = MNISTLoader.loadValidationImages();
			
			ClassificationNeuralNetwork<MNISTImage> net = 
					new ClassificationNeuralNetwork<MNISTImage>(new int[] {MNISTImage.PIXEL_COUNT, 100, 10}, 
							ClassificationNeuralNetwork.SIGMOID_ACTIVATION,
							ClassificationNeuralNetwork.CROSSENTROPY_SIGMOID_COST);
			//net.SGDScheduledEta(trainingData, 10, 0.5, 5.0, evalData, 3, 0.5, 8);
			//net.saveData(new File("sgdscheduled.ann"));
			net.SGD(trainingData, 2, 0.10, 20, 5.0, evalData, true);
			
			//ClassificationNeuralNetwork net = new ClassificationNeuralNetwork(new File("sgdscheduled.ann"));
			//System.out.println(net.evaluate(validationData));
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
