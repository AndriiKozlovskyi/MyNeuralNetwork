package org.example.ai.activation_functions;

public class Sigmoid implements IActivationFunction {
    @Override
    public double output(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double outputDerivative(double x) {
        return x * (1 - x);
    }
}
