package com.example.followwindd.util;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.Core.BORDER_DEFAULT;
import static org.opencv.core.Core.addWeighted;
import static org.opencv.core.Core.convertScaleAbs;
import static org.opencv.core.CvType.CV_16S;
import static org.opencv.imgproc.Imgproc.Sobel;
import static org.opencv.imgproc.Imgproc.cvtColor;

/**
 * 图像的基本处理函数
 *
 * @author 11633
 * @date 2018/4/20 21:29
 */
public class ImageUtil {


    public static final int MIN_VAL = 0;

    public static final int MAX_VAL = 255;

    public static final int BASE_VAL = (MIN_VAL + MAX_VAL) >>> 1;//大于中间值表示增强效果,小于中间值表示消减效果


    /**
     * 图像处理的基本功能
     */
    public static class Base {

        /**
         * @Description: 图像饱和度调节
         * @Param: Mat 传入的图像矩阵
         * @Param: val 调节系数
         * @Return: 调节完成的图像矩阵
         * @Author: followWindDog
         * @Date: 2018/4/18
         */
        public static Mat shiftSaturability(Mat mat, int val) {
            if (val == BASE_VAL) return mat;
            Mat hsv = new Mat();
            double f;
            boolean flag = false;
            if (val > BASE_VAL) {
                flag = true;
                f = 1.0 * (val - BASE_VAL) / BASE_VAL;
            } else {
                f = 1.0 * val / BASE_VAL;
            }

            Imgproc.cvtColor(mat, hsv, Imgproc.COLOR_BGR2HSV);

            for (int i = 0; i < hsv.height(); i++) {
                for (int j = 0; j < hsv.width(); j++) {
                    double[] vals = hsv.get(i, j);
                    if (flag) {
                        vals[1] = vals[1] + (MAX_VAL - vals[1]) * f;
                    } else {
                        vals[1] = vals[1] * f;
                    }
                    hsv.put(i, j, vals);
                }
            }
            Mat ret = new Mat();
            Imgproc.cvtColor(hsv, ret, Imgproc.COLOR_HSV2BGR);
            return ret;
        }

        /**
         * @Description: 图像的对比度调节
         * @Param: Mat 传入的图像矩阵
         * @Param: val 调节系数
         * @Return: 调节完成的图像矩阵
         * @Author: followWindDog
         * @Date: 2018/4/25
         */
        public static Mat shiftContrast(Mat img, int val) {
            if (val == BASE_VAL) return img;
            Mat ret = new Mat(img.height(), img.width(), img.type());
            double maxx = -1, minn = 300.0;
            for (int i = 0; i < img.height(); i++) {
                for (int j = 0; j < img.width(); j++) {
                    double[] vals = img.get(i, j);
                    for (int k = 0; k < vals.length; k++) {
                        maxx = maxx > vals[k] ? maxx : vals[k];
                        minn = minn < vals[k] ? minn : vals[k];
                    }
                }
            }
            double midd = (maxx + minn) / 2;
            double a;
            if(val>BASE_VAL){
                if(val==MAX_VAL)val--;
                a = BASE_VAL/(1.0*(MAX_VAL-val));
            }else{
                a = (1.0 * val) / BASE_VAL;
            }
            double b = midd * (1 - a);
            for (int i = 0; i < img.height(); i++) {
                for (int j = 0; j < img.width(); j++) {
                    double[] vals = img.get(i, j);
                    for (int k = 0; k < vals.length; k++) {
                        vals[k] = a * vals[k] + b;
                    }
                    ret.put(i, j, vals);
                }
            }
            return ret;
        }

        /**
         * @Description: 图像亮度调节
         * @Param: Mat 传入的图像矩阵
         * @Param: val 调节系数
         * @Return: 调节完成的图像矩阵
         * @Author: followWindDog
         * @Date: 2018/4/25
         */
        public static Mat shiftBrightness(Mat img, int val) {
            if (val == BASE_VAL)
                return img;
            Mat ret = new Mat(img.height(), img.width(), img.type());
            int hei = img.height();
            int wid = img.width();
            double f;
            boolean flag = false;
            if (val > BASE_VAL) {
                flag = true;
                f = 1.0 * (MAX_VAL - val) / BASE_VAL;
            } else {
                f = 1.0 * val / BASE_VAL;
            }
            for (int i = 0; i < hei; i++) {
                for (int j = 0; j < wid; j++) {
                    double[] vals = img.get(i, j);
                    for (int k = 0; k < vals.length; k++) {
                        if (flag) {
                            vals[k] = 255 - (vals[k] * f);
                        } else {
                            vals[k] = vals[k] * f;
                        }
                    }
                    ret.put(i, j, vals);
                }
            }
            return ret;
        }

        /**
         * @Description: 将矩阵进行转置
         * @Param: mat 要转置的图像矩阵
         * @Return: 转置过后的图像矩阵
         * @Author: followWindDog
         * @Date: 2018/4/18
         */
        public static Mat transposition(Mat mat) {
            Mat ret = new Mat(mat.width(), mat.height(), mat.type());
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    double[] vals = mat.get(i, j);
                    ret.put(j, i, vals);
                }
            }
            return ret;
        }
    }

    public static class Filter {

        private static int standardizeCellSize(int cellSize) {
            return cellSize % 2 == 0 ? cellSize + 1 : cellSize;
        }


        /**
         * @Description: 高斯函数
         * @Param: x 到达中心的x方向距离
         * @Param: y 到达中心的y方向距离
         * @Param: y 到达中心的y方向距离
         * @Return: variance 方差
         * @Author: followWindDog
         * @Date: 2018/4/20
         */
        private static double gaussianFunction(double x, double y, double variance) {
            return 1.0 / (2 * Math.PI * variance * variance) * Math.exp((-x * x - y * y) / (2 * variance * variance));
        }

        private static boolean chackRange(int i, int j, int hei, int wid) {
            return i >= 0 && i < hei && j >= 0 && j < wid;
        }

        private static Mat filtering(Mat mat, double[][] weightMatrix) {
            Mat ret = new Mat(mat.height(), mat.width(), mat.type());
            double[][][] m = new double[mat.height()][mat.width()][mat.get(0, 0).length];
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    m[i][j] = mat.get(i, j);
                }
            }

            int halfCellSize = weightMatrix.length >>> 1;
            int hei = mat.height();
            int wid = mat.width();
            for (int i = 0; i < hei; i++) {
                for (int j = 0; j < wid; j++) {
                    double[] suma = new double[3];
                    double sumb = 0;
                    for (int k = 0; k < weightMatrix.length; k++) {
                        for (int l = 0; l < weightMatrix.length; l++) {
                            int loci = i - (halfCellSize - k);
                            int locj = j - (halfCellSize - l);
                            if (chackRange(loci, locj, mat.height(), mat.width())) {
                                sumb += weightMatrix[k][l];
                                for (int n = 0; n < suma.length; n++) {
                                    suma[n] += weightMatrix[k][l] * m[loci][locj][n];
                                }
                            }
                        }
                    }
                    for (int k = 0; k < suma.length; k++) {
                        suma[k] /= sumb;
                    }
                    ret.put(i, j, suma);
                }
            }
            return ret;
        }

        public static Mat gaussianFiltering(Mat mat, int cellSize, double variance) {
            //通过高斯函数计算每一个格子的权值
            cellSize = standardizeCellSize(cellSize);
            int hCellSize = cellSize >>> 1;
            double[][] tCell = new double[cellSize][cellSize];
            double sum = 0;
            for (int i = 0; i < tCell.length; i++) {
                for (int j = 0; j < tCell[i].length; j++) {
                    tCell[i][j] = gaussianFunction(hCellSize - i, hCellSize - j, variance);
                    sum += tCell[i][j];
                }
            }
            for (int i = 0; i < tCell.length; i++) {
                for (int j = 0; j < tCell[i].length; j++) {
                    tCell[i][j] = tCell[i][j] * 100 / sum;
                }
            }
            return filtering(mat, tCell);
        }

        /**
         * @Description: 最普通的均值滤波
         * @Param: mat 要滤波的图像矩阵
         * @Param: cellSize 滤波方框的大小
         * @Return: 滤波完成后的矩阵
         * @Author: followWindDog
         * @Date: 2018/4/20
         */
        public static Mat averageFiltering(Mat mat, int cellSize) {
            //TODO 性能优化
            Mat ret = new Mat(mat.height(), mat.width(), mat.type());
            double[][][] mm = new double[mat.height()][mat.width()][mat.get(0, 0).length];
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    mm[i][j] = mat.get(i, j);
                }
            }
            cellSize = standardizeCellSize(cellSize);
            int halfCellSiz = cellSize >> 1;
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    double[] to = new double[3];
                    int fi = i - halfCellSiz, ti = i + halfCellSiz, fj = j - halfCellSiz, tj = j + halfCellSiz;
                    for (int k = 0; k < to.length; k++) {
                        double sum = 0;
                        int cnt = 0;
                        for (int l = fi; l <= ti; l++) {
                            for (int m = fj; m <= tj; m++) {
                                if (l >= 0 && l < mat.height() && m >= 0 && m < mat.width()) {
                                    sum += mm[l][m][k];
                                    cnt++;
                                }
                            }
                        }
                        to[k] = sum / cnt;
                    }
                    ret.put(i, j, to);
                }
            }
            return ret;
        }

        /**
         * @Description: 将图片进行锐化
         * @Param: mat 传入的图像矩阵
         * @Param: cellSize 滤波使用的方框大小
         * @Param: factor 锐化强度
         * @Return: 锐化过后的矩阵
         * @Author: followWindDog
         * @Date: 2018/4/18
         */
        public static Mat sharpen(Mat mat, int cellSize, int factor) {
            Mat filter = averageFiltering(mat, cellSize);
            Mat ret = new Mat(mat.height(), mat.width(), mat.type());
            for (int i = 0; i < ret.height(); i++) {
                for (int j = 0; j < ret.width(); j++) {
                    double[] rgb = mat.get(i, j);
                    double[] frgb = filter.get(i, j);
                    for (int k = 0; k < rgb.length; k++) {
                        rgb[k] += factor * (rgb[k] - frgb[k]);
                    }
                    ret.put(i, j, rgb);
                }
            }
            return ret;
        }

        /*
        * 求图像的暗通道
        * */
        private static double[][] getDarkChannel(Mat mat, int cellSize) {
            if (cellSize % 2 == 0)
                cellSize++;
            int hcellSize = cellSize >>> 1;
            double[][] doubles = new double[mat.height()][mat.width()];
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    double[] vals = mat.get(i, j);
                    doubles[i][j] = vals[0];
                    for (int k = 1; k < vals.length; k++) {
                        doubles[i][j] = (doubles[i][j] < vals[k]) ? doubles[i][j] : vals[k];
                    }
                }
            }

            double[][] ret = new double[mat.height()][mat.width()];
            for (int i = 0; i < mat.height(); i++) {
                for (int j = 0; j < mat.width(); j++) {
                    double maxx = 300;
                    for (int k = 0; k < cellSize; k++) {
                        for (int l = 0; l < cellSize; l++) {
                            int locx = i + (k - hcellSize);
                            int locy = j + (l - hcellSize);
                            if (locx >= 0 && locx < ret.length && locy >= 0 && locy < ret[locx].length)
                                maxx = (maxx < doubles[locx][locy]) ? maxx : doubles[locx][locy];
                        }
                    }
                    ret[i][j] = maxx;
                }
            }
            return ret;
        }



        /**
         * @Description: 图像进行去雾处理(何凯铭去雾)
         * @Param: mat 等待去雾的图像
         * @Param: cellSize 去雾时使用的方框大小
         * @Return: 去雾完成的图像
         * @Author: followWindDog
         * @Date: 2018/4/25
         */
        public static Mat disFog(Mat mat, int cellSize) {
            double[][] dackCha = getDarkChannel(mat, cellSize);
            Mat ret = new Mat(mat.height(), mat.width(), mat.type());

            for (int i = 0; i < ret.height(); i++) {
                for (int j = 0; j < ret.width(); j++) {
                    double[] valsTo = new double[3];
                    double[] valsFrom = mat.get(i, j);
                    double f = dackCha[i][j] / 255;
                    for (int k = 0; k < valsFrom.length; k++) {
                        valsTo[k] = (valsFrom[k] - 127 * f) / (1 - f);
                    }
                    ret.put(i, j, valsTo);
                }
            }
            return ret;
        }



        /**
         * @Description: 为图像进行边缘检测
         * @Param: img 等待边缘检测的图像矩阵
         * @Return: 表示图像边缘的矩阵
         * @Author: followWindDog
         * @Date: 2018/4/25
         */
        public static Mat edgeDetection(Mat img) {
            Mat clone = img.clone();
            int sizex = 5, sizey = 5;
            double dx = 0, dy = 0;
            Imgproc.GaussianBlur(clone, clone, new Size(sizex, sizey), dx, dy, BORDER_DEFAULT);
            Mat grad_x = new Mat(), grad_y = new Mat(), src_gray = new Mat();
            Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();
            cvtColor(clone, src_gray, Imgproc.COLOR_BGR2GRAY);
            int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;
            Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
            convertScaleAbs(grad_x, abs_grad_x);

            Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
            convertScaleAbs(grad_y, abs_grad_y);
            Mat ret = new Mat();
            addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, ret);
            return ret;
        }



        /**
         * @Description: 为图像添加类似油画的效果
         * @Param: img 等待处理的图像
         * @Return: 经过油画处理的图像
         * @Author: followWindDog
         * @Date: 2018/4/25
         */
        public static Mat oilPainting(Mat img) {

            Mat mat1 = edgeDetection(img);
            Imgproc.GaussianBlur(img, img, new Size(11, 11), 20, 20, Core.BORDER_DEFAULT);
            for (int i = 0; i < img.height(); i++) {
                for (int j = 0; j < img.width(); j++) {
                    double[] doubles = img.get(i, j);
                    double[] doubles1 = mat1.get(i, j);
                    for (int k = 0; k < doubles.length; k++) {
                        doubles[k] -= 1 * doubles1[0];
                    }
                    img.put(i, j, doubles);
                }
            }
            return img;
        }

    }
}
