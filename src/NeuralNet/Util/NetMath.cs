using System;

namespace ZNet.NeuralNet.Util {
    public class NetMath {
        public static float Sigmoid(float x) {
            return 1 / (1 + (float)Math.Exp(x));
        }

        public static float SigmoidDerivative(float x) {
            return x * (1 - x);
        }

        public static float Tanh(float x) {
            return (float)Math.Tanh(x);
        }

        public static float TanhDerivative(float x) {
            return (1 + x) * (1 - x);
        }

        public static float ReLU(float x) {
            if (x < 0) return 0;
            return x;
        }

        public static float LeakyReLU(float x) {
            if (x < 0) return x * 0.1f;
            return x;
        }

        public static float ReLUDerivative(float x) {
            if (x < 0) return 0;
            return 1;
        }

        public static float LeakyReLUDerivative(float x) {
            if (x < 0) return 0.1f;
            return 1;
        }

        public static float Linear(float x) {
            return x;
        }

        public static float LinearDerivative(float x) {
            return 1;
        }

        public static float Binary(float x) {
            if (x > 0) {
                return 1;
            } else {
                return 0;
            }
        }

        public static float BinaryDerivative(float x) {
            return 1;
        }

        public static void VectorMultiply(float[] input1, float[] input2, float[] output) {
            for (int i = 0; i < input1.Length; i++) {
                output[i] = input1[i] * input2[i];
            }
        }

        public static void VectorSumFast(float[] input, float output) {
            for (int i = 0; i < input.Length; i++) output += input[i];
        }

        public static void VectorAddFast(float[] input1, float[] input2, float[] output) {
            for (int i = 0; i < input1.Length; i++) {
                output[i] = input1[i] + input2[i];
            }
        }

        public static void VectorSubtractFast(float[] input1, float[] input2, float[] output) {
            for (int i = 0; i < input1.Length; i++) {
                output[i] = input1[i] - input2[i];
            }
        }

        public static void SquareVectorFast(float[] input) {
            for (int i = 0; i < input.Length; i++) {
                input[i] *= input[i];
            }
        }

        public static float[] GetVectorError(float[] input, float[] target) {
            float[] output = new float[input.Length];

            for (int i = 0; i < input.Length; i++) {
                output[i] = (target[i] - input[i])*(target[i] - input[i])/2;
            }

            return output;
        }

        public static float[][] MatrixTranspose(float[][] squareMatrix) {
            float[][] transposedMatrix = squareMatrix;

            for (int i = 0; i < squareMatrix.Length; i++) {
                for (int j = 0; j < squareMatrix[i].Length; i++) {
                    transposedMatrix[i][j] = squareMatrix[j][i];
                }
            }

            return transposedMatrix;
        }
    }
}