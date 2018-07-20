using static ZNet.NeuralNet.Util.NetUtil;
using ZNet.NeuralNet.Util;
using System;

namespace ZNet.NeuralNet.FeedForward {

    public delegate float Activation(float x);

    public class FeedForwardConfig {

        public float LearningRate { get; set; }
        public Activation HiddenActivation { get; set; }
        public Activation OutputActivation { get; set; }
        public Activation HiddenActivationDerivative { get; set; }
        public Activation OutputActivationDerivative { get; set; }
        public Random random = new Random();

        public FeedForwardConfig(float learningRate, ActivationType hiddenActivation, ActivationType outputActivation) {
            this.LearningRate = learningRate;

            switch (hiddenActivation) {
                case ActivationType.Sigmoid:
                    HiddenActivation = NetMath.Sigmoid;
                    HiddenActivationDerivative = NetMath.SigmoidDerivative;
                    break;
                case ActivationType.Tanh:
                    HiddenActivation = NetMath.Tanh;
                    HiddenActivationDerivative = NetMath.TanhDerivative;
                    break;
                case ActivationType.ReLU:
                    HiddenActivation = NetMath.ReLU;
                    HiddenActivationDerivative = NetMath.ReLUDerivative;
                    break;
                case ActivationType.LeakyReLU:
                    HiddenActivation = NetMath.LeakyReLU;
                    HiddenActivationDerivative = NetMath.LeakyReLUDerivative;
                    break;
                case ActivationType.Linear:
                    HiddenActivation = NetMath.Linear;
                    HiddenActivationDerivative = NetMath.LinearDerivative;
                    break;
                case ActivationType.Binary:
                    HiddenActivation = NetMath.Binary;
                    HiddenActivationDerivative = NetMath.BinaryDerivative;
                    break;
            }

            switch (outputActivation) {
                case ActivationType.Sigmoid:
                    OutputActivation = NetMath.Sigmoid;
                    OutputActivationDerivative = NetMath.SigmoidDerivative;
                    break;
                case ActivationType.Tanh:
                    OutputActivation = NetMath.Tanh;
                    OutputActivationDerivative = NetMath.TanhDerivative;
                    break;
                case ActivationType.ReLU:
                    OutputActivation = NetMath.ReLU;
                    OutputActivationDerivative = NetMath.ReLUDerivative;
                    break;
                case ActivationType.LeakyReLU:
                    OutputActivation = NetMath.LeakyReLU;
                    OutputActivationDerivative = NetMath.LeakyReLUDerivative;
                    break;
                case ActivationType.Linear:
                    OutputActivation = NetMath.Linear;
                    OutputActivationDerivative = NetMath.LinearDerivative;
                    break;
                case ActivationType.Binary:
                    OutputActivation = NetMath.Binary;
                    OutputActivationDerivative = NetMath.BinaryDerivative;
                    break;
            }
        }
    }
}