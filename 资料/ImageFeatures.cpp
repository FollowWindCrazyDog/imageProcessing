#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\n展示了两张图像之间的特征点提取和匹配方法，并利用RANSAC进行几何校验.\n" << argv[0] << "[image1] [image2]\n" << endl;
}

const string winName = "Correspondences";

enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };

//简单单向匹配
static void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                     const Mat& descriptors1, const Mat& descriptors2,
                     vector<DMatch>& matches12 )
{
    vector<DMatch> matches;
    descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

//双向匹配，即如果第一张图像中的特征F1，能够在第二张图像中寻找到特征F2与之匹配，表示单向匹配上
//如果同时第二张图像的F2也能在第一张图像中寻找到F1，那么F1和F2实现了双向匹配（准确性更高）
static void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                         const Mat& descriptors1, const Mat& descriptors2,
                         vector<DMatch>& filteredMatches12, int knn=1 )
{
    filteredMatches12.clear();
    vector<vector<DMatch> > matches12, matches21;
    descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}

int main(int argc, char** argv)
{
    if( argc != 3)
    {
        help(argv);
        return -1;
    }

    cv::initModule_nonfree();

    double ransacReprojThreshold = -1;

    cout << "< 创建特征、描述和匹配类型 ..." << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
    int mactherFilterType = CROSS_CHECK_FILTER;
    
	if( detector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty()  )
    {
        cout << "错误参数" << endl;
        return -1;
    }

    cout << "< 读取图像..." << endl;
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2] );
    
	cout << ">" << endl;

    if( img1.empty() || img2.empty())
    {
        cout << "读取错误" << endl;
        return -1;
    }

    cout << endl << "< 提取第一张图像特征点..." << endl;
    vector<KeyPoint> keypoints1;
    detector->detect( img1, keypoints1 );
    cout << keypoints1.size() << " points" << endl << ">" << endl;

    cout << "< 计算第一张图像特征点描述..." << endl;
    Mat descriptors1;
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    cout << ">" << endl;

	cout << endl << "< 提取第二张图像特征点..." << endl;
    vector<KeyPoint> keypoints2;
    detector->detect( img2, keypoints2 );
    cout << keypoints2.size() << " points" << endl << ">" << endl;

    cout << "< 计算第二张图像特征点描述..." << endl;
    Mat descriptors2;
    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
    cout << ">" << endl;

	cout << "< 匹配特征点..." << endl;
    vector<DMatch> filteredMatches;
    switch( mactherFilterType )
    {
    case CROSS_CHECK_FILTER :
        crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
        break;
    default :
        simpleMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches );
    }
    cout << ">" << endl;


	Mat H12;
    vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
		
	if( ransacReprojThreshold >= 0 )
    {
        cout << "< 计算RANSAC投影矩阵..." << endl;
		
		for( size_t i = 0; i < filteredMatches.size(); i++ )
		{
			queryIdxs[i] = filteredMatches[i].queryIdx;
			trainIdxs[i] = filteredMatches[i].trainIdx;
		}

        vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
        H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
        cout << ">" << endl;
    }

	Mat drawImg;
    if( !H12.empty() )
    {
		cout << "< 根据RANSAC投影矩阵，过滤错误匹配..." << endl;
        vector<char> matchesMask( filteredMatches.size(), 0 );
        vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
        vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
        Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);

        double maxInlierDist = ransacReprojThreshold < 0 ? 3 : ransacReprojThreshold;
        for( size_t i1 = 0; i1 < points1.size(); i1++ )
        {
            if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= maxInlierDist ) // inlier
                matchesMask[i1] = 1;
        }

		cout << "< 显示匹配情况..." << endl;
        drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask);
    }
    else
        drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg );

	namedWindow(winName, CV_WINDOW_AUTOSIZE);
	float scale = 0.75f;
	cv::Size newSize = cv::Size(drawImg.cols*scale, drawImg.rows*scale);
	resize(drawImg, drawImg, newSize);
    imshow( winName, drawImg );
	waitKey();

    return 0;
}
