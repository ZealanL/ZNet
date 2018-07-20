namespace ZNet.NeuralNet.Util {
    public class NetUtil {
        //Different activation types usable
        public enum ActivationType { Sigmoid, Tanh, ReLU, LeakyReLU, Binary, Linear }
    }

    [System.Serializable]
    public class UnimplementedException : System.Exception
    {
        public UnimplementedException() { }
        public UnimplementedException(string message) : base(message) { }
        public UnimplementedException(string message, System.Exception inner) : base(message, inner) { }
        protected UnimplementedException(
            System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
    }
}