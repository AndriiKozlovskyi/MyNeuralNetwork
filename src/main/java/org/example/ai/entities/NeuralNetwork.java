package org.example.ai.entities;

import lombok.Setter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.example.ai.activation_functions.*;
import org.example.ai.data.MLData;
import org.example.ai.data.MLDataSet;

import java.util.*;

public class NeuralNetwork {
    private static final Logger logger = LogManager.getLogger(NeuralNetwork.class);


    private final int inputSize;
    private final int outputSize;
    private final int hiddenSize;

    private final List<Neuron> inputLayer;
    private final List<Neuron> hiddenLayer;
    private final List<Neuron> outputLayer;

    @Setter
    private double learningRate = 0.01;
    @Setter
    private double momentum = 0.5;
    private IActivationFunction activationFunction = new Sigmoid();
    private boolean initialized = false;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.inputLayer = new ArrayList<>();
        this.hiddenLayer = new ArrayList<>();
        this.outputLayer = new ArrayList<>();
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        switch (activationFunction) {
            case LEAKY_RELU:
                this.activationFunction = new LeakyReLu();
                break;
            case TANH:
                this.activationFunction = new TanH();
                break;
            case SIGMOID:
                this.activationFunction = new Sigmoid();
                break;
            case SWISH:
                this.activationFunction = new Swish();
                break;
        }
    }

    public void init() {
        for (int i = 0; i < inputSize; i ++) {
            this.inputLayer.add(new Neuron());
        }
        for (int i = 0; i < hiddenSize; i ++) {
            this.hiddenLayer.add(new Neuron(this.inputLayer, activationFunction));
        }
        for (int i = 0; i < outputSize; i ++) {
            this.outputLayer.add(new Neuron(this.hiddenLayer, activationFunction));
        }

        this.initialized = true;
        logger.info("Network initialized");
    }

    public void train(MLDataSet set, int epoch) {
        if (!initialized) {
            this.init();
        }
        logger.info("Training started");
        for(int i = 0; i < epoch; i ++) {
            Collections.shuffle(set.getData());

            for (MLData datum : set.getData()) {
                forward(datum.getInputs());
                backward(datum.getTargets());
            }
        }
    }

    public void backward(double[] targets) {
        int i = 0;
        for(Neuron neuron : outputLayer) {
            neuron.calculateGradient(targets[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateGradient();
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
        for (Neuron neuron : outputLayer) {
            neuron.updateConnections(learningRate, momentum);
        }
    }

    public void forward(double[] inputs) {
        int i = 0;
        for (Neuron neuron : inputLayer) {
            neuron.setOutput(inputs[i++]);
        }
        for (Neuron neuron : hiddenLayer) {
            neuron.calculateOutput();
        }
        for (Neuron neuron : outputLayer) {
            neuron.calculateOutput();
        }
    }

    public double[] predict(double... inputs) {
        forward(inputs);
        double[] output = new double[outputLayer.size()];
        for (int i = 0; i < output.length; i ++) {
            output[i] = outputLayer.get(i).getOutput();
        }
        logger.info("Input : " + Arrays.toString(inputs) + " Predicted : " + Arrays.toString(output));
        return output;
    }

}
