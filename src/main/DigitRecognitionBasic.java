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
					new DigitRecognitionNeuralNetwork(new int[] {MNISTImage.PIXEL_COUNT, 100, 10}, 
							DigitRecognitionNeuralNetwork.SIGMOID_ACTIVATION,
							DigitRecognitionNeuralNetwork.CROSSENTROPY_SIGMOID_COST);
			net.SGDAndSave(trainingData, 10, 0.5, 40, 5.0, evalData, new File("network.ann"));
			//net.SGD(trainingData, 10, 0.5, 40, 5.0, evalData, true);
			
			/*DigitRecognitionNeuralNetwork net = new DigitRecognitionNeuralNetwork(new File("94.93%.ann"));
			System.out.println(net.evaluate(validationData));*/
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
