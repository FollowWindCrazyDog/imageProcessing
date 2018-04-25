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
    cout << "\nչʾ������ͼ��֮�����������ȡ��ƥ�䷽����������RANSAC���м���У��.\n" << argv[0] << "[image1] [image2]\n" << endl;
}

const string winName = "Correspondences";

enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1 };

//�򵥵���ƥ��
static void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                     const Mat& descriptors1, const Mat& descriptors2,
                     vector<DMatch>& matches12 )
{
    vector<DMatch> matches;
    descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

//˫��ƥ�䣬�������һ��ͼ���е�����F1���ܹ��ڵڶ���ͼ����Ѱ�ҵ�����F2��֮ƥ�䣬��ʾ����ƥ����
//���ͬʱ�ڶ���ͼ���F2Ҳ���ڵ�һ��ͼ����Ѱ�ҵ�F1����ôF1��F2ʵ����˫��ƥ�䣨׼ȷ�Ը��ߣ�
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

    cout << "< ����������������ƥ������ ..." << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
    int mactherFilterType = CROSS_CHECK_FILTER;
    
	if( detector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty()  )
    {
        cout << "�������" << endl;
        return -1;
    }

    cout << "< ��ȡͼ��..." << endl;
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2] );
    
	cout << ">" << endl;

    if( img1.empty() || img2.empty())
    {
        cout << "��ȡ����" << endl;
        return -1;
    }

    cout << endl << "< ��ȡ��һ��ͼ��������..." << endl;
    vector<KeyPoint> keypoints1;
    detector->detect( img1, keypoints1 );
    cout << keypoints1.size() << " points" << endl << ">" << endl;

    cout << "< �����һ��ͼ������������..." << endl;
    Mat descriptors1;
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    cout << ">" << endl;

	cout << endl << "< ��ȡ�ڶ���ͼ��������..." << endl;
    vector<KeyPoint> keypoints2;
    detector->detect( img2, keypoints2 );
    cout << keypoints2.size() << " points" << endl << ">" << endl;

    cout << "< ����ڶ���ͼ������������..." << endl;
    Mat descriptors2;
    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
    cout << ">" << endl;

	cout << "< ƥ��������..." << endl;
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
        cout << "< ����RANSACͶӰ����..." << endl;
		
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
		cout << "< ����RANSACͶӰ���󣬹��˴���ƥ��..." << endl;
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

		cout << "< ��ʾƥ�����..." << endl;
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
