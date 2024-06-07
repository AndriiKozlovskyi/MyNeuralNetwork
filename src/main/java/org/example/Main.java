package org.example;

import org.example.ai.activation_functions.ActivationFunction;
import org.example.ai.data.MLDataSet;
import org.example.ai.entities.NeuralNetwork;

public class Main {
    private static final double[][] XOR_INPUT = {
            {1, 1},
            {1, 0},
            {0, 1},
            {0, 0}
    };

    private static final double[][] XOR_IDEAL = {
            {0},
            {1},
            {1},
            {0}
    };
    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 10, 1);
        neuralNetwork.init();
        neuralNetwork.setLearningRate(0.01);
        neuralNetwork.setMomentum(0.5);
        neuralNetwork.setActivationFunction(ActivationFunction.SIGMOID);


        MLDataSet dataSet = new MLDataSet(XOR_INPUT, XOR_IDEAL);
        neuralNetwork.train(dataSet, 100000);

        neuralNetwork.predict(1, 1);
        neuralNetwork.predict(1, 0);
        neuralNetwork.predict(0, 1);
        neuralNetwork.predict(0, 0);
    }
}