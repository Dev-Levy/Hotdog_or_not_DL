using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace DeepLearning_FF
{
    enum Classification { not_hotdog, hotdog }
    internal class Program
    {
        static void Main()
        {
            //string originalHotdogFolder = "C:\\Users\\horga\\Downloads\\hotdog-nothotdog\\hotdog-nothotdog\\train\\hotdog";
            string hotdogFolder = "C:\\Users\\horga\\Downloads\\hotdog-nothotdog\\hotdog-nothotdog\\train\\hotdog1x1";
            //ResizeImages(originalHotdogFolder, hotdogFolder);
            ImgDataSet hotdog = new ImgDataSet(hotdogFolder, Classification.hotdog);

            //string originalNot_hotdogFolder = "C:\\Users\\horga\\Downloads\\hotdog-nothotdog\\hotdog-nothotdog\\train\\not_hotdog";
            string not_hotdogFolder = "C:\\Users\\horga\\Downloads\\hotdog-nothotdog\\hotdog-nothotdog\\train\\not_hotdog1x1";
            //ResizeImages(originalNot_hotdogFolder, not_hotdogFolder);
            ImgDataSet not_hotdog = new ImgDataSet(not_hotdogFolder, Classification.not_hotdog);

            ImgDataSet dataSet = ImgDataSet.Union(new List<ImgDataSet>() { hotdog, not_hotdog });

            DeepNetwork deep = new DeepNetwork();
            deep.Train(dataSet);
            deep.Save("hotdog_DL.txt");
        }
        public static void ResizeImages(string sourceFolder, string destFolder)
        {
            int width_height = 299;
            string[] imageFiles = Directory.GetFiles(sourceFolder);
            Directory.CreateDirectory(destFolder);
            foreach (string imagePath in imageFiles)
            {
                string fileName = Path.GetFileNameWithoutExtension(imagePath);
                string destPath = Path.Combine(destFolder, fileName + ".png");

                using (Image originalImage = Image.FromFile(imagePath))
                {
                    if (originalImage.Width != width_height || originalImage.Height != width_height)
                    {
                        using (Bitmap resizedImage = new Bitmap(width_height, width_height))
                        {
                            resizedImage.SetResolution(originalImage.HorizontalResolution, originalImage.VerticalResolution);

                            using (Graphics graphics = Graphics.FromImage(resizedImage))
                            {
                                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

                                graphics.DrawImage(originalImage, 0, 0, width_height, width_height);
                            }

                            resizedImage.Save(destPath, ImageFormat.Png);
                            originalImage.Dispose();
                            resizedImage.Dispose();
                        }
                    }
                    else
                    {
                        originalImage.Save(destPath, ImageFormat.Png);
                    }
                }
            }
        }
    }
}
