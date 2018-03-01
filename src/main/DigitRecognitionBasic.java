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
			DigitRecognitionNeuralNetwork net = new DigitRecognitionNeuralNetwork(new int[] {MNISTImage.PIXEL_COUNT, 16, 10});
			net.SGD(trainingData, 10, 1.0, 20, evalData);
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
		
	}

}
