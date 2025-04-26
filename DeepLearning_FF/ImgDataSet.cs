using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace DeepLearning_FF
{
    internal class ImgDataSet
    {
        public static readonly int InputSize = 299 * 299;
        public static readonly int OutputSize = 1;
        public List<float> Input { get; set; }
        public List<float> Output { get; set; }

        public int Count { get; set; }

        public ImgDataSet()
        {
            Input = new List<float>();
            Output = new List<float>();
        }
        public ImgDataSet(string folderName, Classification clas)
        {
            Input = new List<float>();
            Output = new List<float>();
            LoadInput(folderName);
            LoadOutput(clas);
        }

        public static ImgDataSet Union(List<ImgDataSet> dataSets)
        {
            ImgDataSet dataSet = new ImgDataSet();
            foreach (var set in dataSets)
            {
                dataSet.Input.AddRange(set.Input);
                dataSet.Output.AddRange(set.Output);
                dataSet.Count += set.Count;
            }
            return dataSet;
        }

        private void LoadInput(string folderName)
        {
            int count = 0;
            string[] files = Directory.GetFiles(folderName);

            ConcurrentBag<float[]> imgs = new ConcurrentBag<float[]>();

            Parallel.ForEach(files, filename =>
            {
                float[] floatArray = ImageToFloatArray(filename);
                imgs.Add(floatArray);
                Interlocked.Increment(ref count);
            });

            foreach (var img in imgs)
            {
                Input.AddRange(img);
            }
            Count = count;
        }

        private void LoadOutput(Classification clas)
        {
            for (int i = 0; i < Count; i++)
            {
                Output.Add((float)clas);
            }
        }

        private float[] ImageToFloatArray(string imagePath)
        {
            using (Bitmap image = new Bitmap(imagePath))
            {
                float[] pixelData = new float[InputSize];
                int index = 0;

                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        Color pixel = image.GetPixel(x, y);
                        float grayValue = (pixel.R * 0.299f + pixel.G * 0.587f + pixel.B * 0.114f) / 255f;
                        pixelData[index] = grayValue;
                        index++;
                    }
                }
                return pixelData;
            }
        }

        internal void Shuffle()
        {
            Random random = new Random();
            for (int i = 0; i < Count; i++)
            {
                int a = random.Next(Count);
                int b = random.Next(Count);

                if (a != b)
                {
                    float T;
                    for (int j = 0; j < InputSize; j++)
                    {
                        T = Input[a * InputSize + j];
                        Input[a * InputSize + j] = Input[b * InputSize + j];
                        Input[b * InputSize + j] = T;
                    }
                    T = Output[a * OutputSize];
                    Output[a * OutputSize] = Output[b * OutputSize];
                    Output[b * OutputSize] = T;
                }
            }
        }
    }
}
