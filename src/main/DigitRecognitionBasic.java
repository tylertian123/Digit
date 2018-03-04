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
			DigitRecognitionNeuralNetwork net = 
					new DigitRecognitionNeuralNetwork(new int[] {MNISTImage.PIXEL_COUNT, 50, 10}, 
							DigitRecognitionNeuralNetwork.SIGMOID_ACTIVATION,
							DigitRecognitionNeuralNetwork.CROSSENTROPY_SIGMOID_COST);
			//net.SGD(trainingData, 10, 0.50, 50, evalData, true);
			//net.saveData(new File("net1.ann"));
			net.SGDAndSave(trainingData, 10, 0.50, 50, evalData, new File("net1.ann"));
			
			/*DigitRecognitionNeuralNetwork net = new DigitRecognitionNeuralNetwork(new File("ann1.ann"));
			System.out.println(net.evaluate(evalData));*/
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
