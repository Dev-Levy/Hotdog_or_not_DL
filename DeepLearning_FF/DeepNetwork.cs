using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace DeepLearning_FF
{
    internal class DeepNetwork
    {
        readonly int[] layers = new int[] { ImgDataSet.InputSize, 50, 50, 50, ImgDataSet.OutputSize };

        readonly int batchSize = 10;
        readonly int epochCount = 1000;

        readonly Variable x;
        readonly Function y;

        public DeepNetwork()
        {
            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float, "X");

            Function lastLayer = x;
            for (int i = 0; i < layers.Length - 1; i++)
            {
                Parameter weight = new Parameter(new int[] { layers[i + 1], layers[i] }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                Parameter bias = new Parameter(new int[] { layers[i + 1] }, DataType.Float, CNTKLib.GlorotNormalInitializer());

                lastLayer = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(weight, lastLayer), bias)); //(LAYER * WEIGHT) + BIAS
            }

            y = lastLayer;
        }

        public DeepNetwork(string filename)
        {
            y = Function.Load(filename, DeviceDescriptor.CPUDevice);
            x = y.Arguments.First(a => a.Name == "X");
        }

        public void Train(ImgDataSet imgds)
        {
            Variable yt = Variable.InputVariable(new int[] { ImgDataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.SquaredError(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, y);

            double learningRate = 0.001;
            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()),
                                                 new TrainingParameterScheduleDouble(learningRate, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner> { learner });

            var batchData = new Tuple<Value, Value>[imgds.Count / batchSize];

            for (int i = 0; i <= epochCount; i++)
            {
                double sumLoss = 0;
                double sumEval = 0;

                imgds.Shuffle();
                Parallel.For(0, imgds.Count / batchSize, j =>
                {
                    Value x_value = Value.CreateBatch(x.Shape, imgds.Input.GetRange(j * batchSize * ImgDataSet.InputSize, ImgDataSet.InputSize * batchSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, imgds.Output.GetRange(j * batchSize * ImgDataSet.OutputSize, ImgDataSet.OutputSize * batchSize), DeviceDescriptor.CPUDevice);

                    batchData[j] = Tuple.Create(x_value, yt_value);
                });

                foreach (var batch in batchData)
                {
                    var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, batch.Item1 },
                        { yt, batch.Item2 }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }

                //for (int j = 0; j < imgds.Count / batchSize; j++)
                //{
                //    Value x_value = Value.CreateBatch(x.Shape, imgds.Input.GetRange(j * batchSize * ImgDataSet.InputSize, ImgDataSet.InputSize * batchSize), DeviceDescriptor.CPUDevice);
                //    Value yt_value = Value.CreateBatch(yt.Shape, imgds.Output.GetRange(j * batchSize * ImgDataSet.OutputSize, ImgDataSet.OutputSize * batchSize), DeviceDescriptor.CPUDevice);

                //    var inputDataMap = new UnorderedMapVariableValuePtr()
                //    {
                //        { x, x_value },
                //        { yt, yt_value }
                //    };
                //    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                //    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                //    sumEval += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                //}
                Console.WriteLine($"{i}.\tloss: {sumLoss / imgds.Count}\teval: {sumEval / imgds.Count}");
            }
        }

        public void Save(string filename)
        {
            y.Save(filename);
        }

        public double Evaluate(ImgDataSet imgds)
        {
            Variable yt = Variable.InputVariable(new int[] { 10 }, DataType.Float);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, y);
            Evaluator evaluator = CNTKLib.CreateEvaluator(y_yt_equal);

            double sumEval = 0;
            for (int i = 0; i < 100 / batchSize; i++)
            {
                Value x_value = Value.CreateBatch(x.Shape, imgds.Input.GetRange(i * batchSize * ImgDataSet.InputSize, imgds.Count),
                                                        DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(x.Shape, imgds.Output.GetRange(i * batchSize * ImgDataSet.OutputSize, imgds.Count),
                                                    DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                {
                    {x, x_value },
                    {yt,yt_value }
                };
                sumEval += evaluator.TestMinibatch(inputDataMap, DeviceDescriptor.CPUDevice) * batchSize;
            }
            return sumEval / imgds.Count;
        }
    }
}
