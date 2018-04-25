// Stitch.cpp : 定义控制台应用程序的入口点。
//

#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/stitching/stitcher.hpp"

using namespace std;
using namespace cv;

vector<Mat> imgs;
string result_name = "result.jpg";

void printUsage();
int parseCmdArgs(int argc, char** argv);

int main(int argc, char* argv[])
{
	//读取分析命令行
    int retval = parseCmdArgs(argc, argv);
    if (retval) return -1;

    Mat pano; //输出全景图

	//进行图像拼接
    Stitcher stitcher = Stitcher::createDefault();
    Stitcher::Status status = stitcher.stitch(imgs, pano);

    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return -1;
    }

	imshow("panoramas", pano);
    imwrite(result_name, pano);
	waitKey();
    return 0;
}


void printUsage()
{
    cout <<
        "stitching img1 img2 [...imgN]\n\n"
        "Flags:\n"
        "  --output <result_img>\n"
        "      默认'result.jpg'.\n";
}


int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }      
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
        {
            Mat img = imread(argv[i]);
            if (img.empty())
            {
                cout << "Can't read image '" << argv[i] << "'\n";
                return -1;
            }
            imgs.push_back(img);
        }
    }
    return 0;
}