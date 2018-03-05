package main;

import mnist.MNISTLoader;

import java.io.File;
import java.io.IOException;

import mnist.MNISTImage;
import neuralnet.DigitRecognitionNeuralNetwork;
import neuralnet.NeuralNetworkException;

public class DigitRecognitionBasic {

	public static void main(String[] args) {
		try {
			MNISTImage[] trainingData = MNISTLoader.loadTrainingImages();
			MNISTImage[] evalData = MNISTLoader.loadTestingImages();
			MNISTImage[] validationData = MNISTLoader.loadValidationImages();
			
			DigitRecognitionNeuralNetwork net = 
					new DigitRecognitionNeuralNetwork(new int[] {MNISTImage.PIXEL_COUNT, 70, 30, 10}, 
							DigitRecognitionNeuralNetwork.SIGMOID_ACTIVATION,
							DigitRecognitionNeuralNetwork.CROSSENTROPY_SIGMOID_COST);
			//net.SGDAndSave(trainingData, 10, 0.5, 30, 3.0, evalData, new File("network.ann"));
			net.SGD(trainingData, 10, 0.5, 10, 3.0, evalData, true);
			net.SGD(trainingData, 10, 0.25, 10, 3.0, evalData);
			net.SGDAndSave(trainingData, 10, 0.125, 5, 3.0, evalData, new File("network.ann"));
			
			/*DigitRecognitionNeuralNetwork net = new DigitRecognitionNeuralNetwork(new File("94.93%.ann"));
			System.out.println(net.evaluate(validationData));*/
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
