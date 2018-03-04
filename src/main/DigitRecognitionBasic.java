package main;

import mnist.MNISTLoader;

import java.io.IOException;

import mnist.MNISTImage;
import neuralnet.DigitRecognitionNeuralNetwork;

public class DigitRecognitionBasic {

	public static void main(String[] args) {
		try {
			MNISTImage[] trainingData = MNISTLoader.loadTrainingImages();
			MNISTImage[] evalData = MNISTLoader.loadTestingImages();
			DigitRecognitionNeuralNetwork net = 
					new DigitRecognitionNeuralNetwork(new int[] {MNISTImage.PIXEL_COUNT, 50, 10}, 
							DigitRecognitionNeuralNetwork.SIGMOID_ACTIVATION,
							DigitRecognitionNeuralNetwork.CROSSENTROPY_SIGMOID_COST);
			net.SGD(trainingData, 10, 0.5, 40, evalData);
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
		
	}

}
