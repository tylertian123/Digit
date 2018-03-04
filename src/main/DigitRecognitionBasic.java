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
			net.SGD(trainingData, 10, 0.80, 40, evalData, true);
			net.saveDataAs(new File("net1.ann"));
			/*DigitRecognitionNeuralNetwork net = new DigitRecognitionNeuralNetwork(new File("net1.ann"));
			System.out.println(net.evaluate(evalData));*/
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
