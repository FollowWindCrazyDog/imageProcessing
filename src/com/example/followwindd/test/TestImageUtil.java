package com.example.followwindd.test;

import com.example.followwindd.util.ImageUtil;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;


/**
 * @author 11633
 * @date 2018/4/21 8:46
 */
public class TestImageUtil {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static Mat imread = Imgcodecs.imread("./photo/00005.jpg");

    public static void showImg(Mat imread, int len) {
        HighGui.namedWindow("test", HighGui.WINDOW_AUTOSIZE);
        HighGui.imshow("test", imread);
        HighGui.waitKey(len);
    }

    public static class TestBase {
        @Test
        public void testShiftSaturability() {
            for (int i = 0; i <= 255; i += 50)
                showImg(ImageUtil.Base.shiftSaturability(imread, i), 1000);
        }

        @Test
        public void testShiftContrast() {
            for (int i = 0; i <= 255; i += 50)
                showImg(ImageUtil.Base.shiftContrast(imread, i), 1000);
        }

        @Test
        public void testShiftBrightness() {
            for (int i = 0; i <= 255; i += 50) {
                showImg(ImageUtil.Base.shiftBrightness(imread, i), 1000);
            }
        }

        @Test
        public void testTransposition() {
            showImg(ImageUtil.Base.transposition(imread), 0);
        }
    }


    public static class TestFilter {
        @Test
        public void testGaussianFiltering() {
            for (int i = 3; i < 20; i += 2) {
                showImg(ImageUtil.Filter.gaussianFiltering(imread, i, 100), 100);
                System.out.println(i);
            }
        }


        @Test
        public void testAverageFiltering() {
            for (int i = 3; i < 20; i += 2) {
                showImg(ImageUtil.Filter.averageFiltering(imread, i), 100);
                System.out.println(i);
            }
        }

        @Test
        public void testSharpen() {
            for (int i = 3; i < 20; i += 2) {
                showImg(ImageUtil.Filter.sharpen(imread, i, 2), 100);
                System.out.println(i);
            }
        }

        @Test
        public void testDisFog() {
            showImg(ImageUtil.Filter.disFog(imread, 20), 0);
        }

        @Test
        public void testEdgeDetection() {
            showImg(ImageUtil.Filter.edgeDetection(imread), 0);
        }


        @Test
        public void testOilPainting() {
            showImg(ImageUtil.Filter.oilPainting(imread), 0);
        }
    }


}
