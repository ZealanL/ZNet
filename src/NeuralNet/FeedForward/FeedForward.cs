using System;
using ZNet.NeuralNet.FeedForward;
using ZNet.NeuralNet.Util;

    namespace ZNet.NeuralNet.FeedForward
    {
        public class FeedForward
        {
            public FeedForwardConfig Config = new FeedForwardConfig(0.2f, NetUtil.ActivationType.Tanh, NetUtil.ActivationType.Sigmoid);

            public float[][][] Weights;
            public float[][] Biases; 

            public float LearningRate = 0.2f;

            //TODO: Implement
            //public float Momentum = 0f;

            public FeedForward(int inputAmount, int[] layerSizes, FeedForwardConfig config) {

                this.Config = config;

                //Initialize current input size, will be reassigned after completing layer initialization
                int currentInput = inputAmount;

                Weights = new float[layerSizes.Length][][];
                Biases = new float[layerSizes.Length][];

                //Loop through layers and create weight arrays for each neuron
                //Weight size is previous layer size (currentInput)
                for (int layer = 0; layer < layerSizes.Length; layer++) {

                    //Initialize layer array
                    Weights[layer] = new float[layerSizes[layer]][];

                    //Initialize biases
                    Biases[layer] = new float[layerSizes[layer]];

                    //Initialize weights
                    for (int neuron = 0; neuron < layerSizes[layer]; neuron++) {
                        Weights[layer][neuron] = new float[currentInput];
                    }
                    currentInput = layerSizes[layer];
                }

                this.InitializeWeights();
            }

            private void InitializeWeights() {
                for (int l = 0; l < Weights.Length; l++) {
                    for(int n = 0; n < Weights [l].Length; n++) {
                        for (int w = 0; w < Weights[l][n].Length; w++) {

                            //Initialize weight from -0.5f to 0.5f
                            Weights[l][n][w] = (float)Config.random.NextDouble() - 0.5f;
                        }
                    }
                }
            }

            ///<summary>
            ///Feed forward through the network and return the output. Should not be used excessively, Calculate(float[] input) is more preformant.
            ///</summary>
            public float[] CalculateOutput(float[] input) {
                return Calculate(input)[Weights.Length - 1];
            }

            ///<summary>
            ///Feed forward through the network and return the outputs of each neuron in all layers.
            ///</summary>
            public float[][] Calculate(float[] input) {

                float[][] neuronOutputs = new float[Weights.Length][];
                for (int l = 0; l < Weights.Length; l++) {

                    //Initialize array of outputs from current layer
                    float[] neuronOutput = new float[Weights[l].Length];

                    for (int n = 0; n < Weights[l].Length; n++) {
                        float sum = Biases[l][n];
                        //Set the output for the neuron as the (sum + bias) through the activation function (Sigmoid)
                        for (int w = 0; w < Weights[l][n].Length; w++) {
                            sum += Weights[l][n][w] * input[w];
                        }

                        if (l == Weights.Length - 1) {
                            neuronOutput[n] = Config.OutputActivation(sum);
                        } else {
                            neuronOutput[n] = Config.HiddenActivation(sum);
                        }
                    }

                    input = neuronOutput;
                    neuronOutputs[l] = neuronOutput;
                }

                //When layer parsing is complete, return the input from the last layer (the final output)
                return neuronOutputs;
            }

            public void Train(FeedForwardTrainingSet trainingSet, int iterations) {

                //Initialize previous output array for each neuron's previous output
                float[][] previousOutput = new float[Weights.Length][];
                for (int i = 0; i < Weights.Length; i++) {
                    previousOutput[i] = new float[Weights[i].Length];
                }

                for (int iteration = 0; iteration < iterations; iteration++) {
                    for (int trainingIndex = 0; trainingIndex < trainingSet.TrainingData.Length; trainingIndex++) {

                        float[] input = trainingSet.TrainingData[trainingIndex];
                        float[] target = trainingSet.TargetOutput[trainingIndex];
                        float[][] output = Calculate(input);
                        
                        //Get error of output layer (output[output.Length - 1] == output layer neurons output)
                        float[] outputLayerError = new float[output[output.Length - 1].Length];
                        outputLayerError = NetMath.GetVectorError(target, output[output.Length - 1]);

                        //Compute network total error
                        float totalError = 0;
                        NetMath.VectorSumFast(outputLayerError, totalError);

                        AdjustWeights(input, BackpropagateNeuronError(output, input, target), output);
                    }
                }
            }

            ///<summary>
            ///Backpropagate to find individual neuron error, returns error signals for each neuron.
            ///<summary>
            private float[][] BackpropagateNeuronError(float[][] outputs, float[] input, float[] target) {
                float[][] error = new float[Weights.Length][];

                //Backpropagate error of output layer
                int outputLayer = Weights.Length - 1;
                error[outputLayer] = new float[Weights[outputLayer].Length];
                for (int outputNeuron = 0; outputNeuron < Weights[outputLayer].Length; outputNeuron++) {
                    //Error for output layer is (derivative of output) * (target - output), basically how much the output neuron screwed up
                    float neuronError = target[outputNeuron] - outputs[outputLayer][outputNeuron];
                    error[outputLayer][outputNeuron] = neuronError * Config.OutputActivationDerivative(outputs[outputLayer][outputNeuron]);
                }

                //However, the output neurons aren't the only thing we need to change
                // -> Backpropagate error of hidden layer(s)
                for (int hiddenLayer = Weights.Length - 2; hiddenLayer > -1; hiddenLayer--) {
                    error[hiddenLayer] = new float[Weights[hiddenLayer].Length];
                    for (int hiddenNeuron = 0; hiddenNeuron < Weights[hiddenLayer].Length; hiddenNeuron++) {
                        //Need to keep track of hiddenNeuron's outputs' net weights (how important we (the neuron) are),
                        //More importance in next layer means less of a shift is needed to achieve the same total change
                        float sum = 0;

                        //In order to do that ^, we need to check with the layer ahead of us to see how much our output is valued
                        int nextLayer = hiddenLayer + 1;
                        for (int nextLayerNeuron = 0; nextLayerNeuron < Weights[nextLayer].Length; nextLayerNeuron++) {
                            //How much they error/signal * how much they care about our output (their weight from our output)
                            sum += error[nextLayer][nextLayerNeuron] * Weights[nextLayer][nextLayerNeuron][hiddenNeuron];
                        }

                        //Ok, so we know how much the next layer cares about us (this hidden neuron)
                        // -> Now we can just get our derivative and multiply it by the sum for our error signal
                        error[hiddenLayer][hiddenNeuron] = sum * Config.HiddenActivationDerivative(outputs[hiddenLayer][hiddenNeuron]);
                    }
                }

                //Return the error for all neurons that we just calculated, will be used to adjust weights and biases
                return error;
            }

            ///<summary>
            ///Adjust neuron weights from neuron error.
            ///<summary>
            private void AdjustWeights(float[] inputs, float[][] errorSignals, float[][] outputs) {
                //Go through every layer except the input layer (we will do that later)
                for (int layer = 1; layer < Weights.Length; layer++) {
                    for (int neuron = 0; neuron < Weights[layer].Length; neuron++) {
                        float delta = -Config.LearningRate * errorSignals[layer][neuron];
                        Biases[layer][neuron] += delta; //Adjust bias
                        for (int previousNeuron = 0; previousNeuron < Weights[layer-1].Length; previousNeuron++) {
                            //Adjust weights of current neuron
                            Weights[layer][neuron][previousNeuron] += delta * outputs[layer-1][previousNeuron];
                            //Console.WriteLine("Adjusted Weight {2} of Neuron {1} of Layer {0} by {3}", layer+1, neuron+1, previousNeuron, delta * outputs[layer-1][previousNeuron]);
                        }
                    }
                }

                //Go through input layer and adjust weights
                for (int inputNeuron = 0; inputNeuron < Weights[0].Length; inputNeuron++) {
                    float delta = -Config.LearningRate * errorSignals[0][inputNeuron];
                    Biases[0][inputNeuron] += delta; //Adjust bias
                    for (int input = 0; input < inputs.Length; input++) {
                        //Adjust weights of first layer (input layer) to delta * the input for that weight
                        Weights[0][inputNeuron][input] += delta * inputs[input];
                        //Console.WriteLine("Adjusted Weight {2} of Neuron {1} of Layer {0} by {3}", 1, inputNeuron+1, input, delta * inputs[input]);
                    }
                }
            }
        }

        public struct FeedForwardTrainingSet {
            public float[][] TrainingData { get; set; }
            public float[][] TargetOutput { get; set; }
            public FeedForwardTrainingSet(float[][] trainingData, float[][] targetOutput) {
                if (trainingData.Length != targetOutput.Length) {
                    throw new Exception("TrainingSet Exception: Training data size must match expected output size");
                }
                
                this.TrainingData = trainingData;
                this.TargetOutput = targetOutput;
            }
        }
    }
