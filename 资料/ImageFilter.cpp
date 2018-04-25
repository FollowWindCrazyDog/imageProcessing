
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/// 全局变量
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

Mat src; Mat dst;
char window_name[] = "Filter Demo 1";

/// 函数申明
int display_caption( char* caption );
int display_dst( int delay );

/**
 *  main 函数
 */
 int main( int argc, char** argv )
 {
   namedWindow( window_name, CV_WINDOW_AUTOSIZE );

   /// 载入原图像
   src = imread( "..//..//Images//ljx.jpg", 1 );

   if( display_caption( "Original Image" ) != 0 ) { return 0; }

   dst = src.clone();
   if( display_dst( DELAY_CAPTION ) != 0 ) { return 0; }

   /// 使用 均值平滑
   if( display_caption( "Homogeneous Blur" ) != 0 ) { return 0; }

   for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
       { blur( src, dst, Size( i, i ), Point(-1,-1) );
         if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

    /// 使用高斯平滑
    if( display_caption( "Gaussian Blur" ) != 0 ) { return 0; }

    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
        { GaussianBlur( src, dst, Size( i, i ), 0, 0 );
          if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

     /// 使用中值平滑
     if( display_caption( "Median Blur" ) != 0 ) { return 0; }

     for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { medianBlur ( src, dst, i );
           if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

     /// 使用双边平滑
     if( display_caption( "Bilateral Blur" ) != 0 ) { return 0; }

     for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { bilateralFilter ( src, dst, i, i*2, i/2 );
           if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

	 //图像简单锐化
	 if( display_caption( "Sharpening" ) != 0 ) { return 0; }
	 for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { bilateralFilter ( src, dst, i, i*2, i/2 );
		   dst = dst + 3*abs(src - dst);
           if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

	 //Sobel边缘检测
	 if( display_caption( "Sobel Edge" ) != 0 ) { return 0; }
	 for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
	 {
		 GaussianBlur( src, src, Size(3, 3), 0, 0, BORDER_DEFAULT );

		 Mat src_gray;
		 cvtColor( src, src_gray, CV_RGB2GRAY );
		 
		  /// Generate grad_x and grad_y
		  Mat grad_x, grad_y;
		  Mat abs_grad_x, abs_grad_y;

		  int scale = 1;
		  int delta = 0;
		  int ddepth = CV_16S;
		  /// Gradient X
		  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		  convertScaleAbs( grad_x, abs_grad_x );

		  /// Gradient Y
		  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		  convertScaleAbs( grad_y, abs_grad_y );

		  /// Total Gradient (approximate)
		  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );

		  if( display_dst( DELAY_BLUR ) != 0 ) { return 0; }
	 }

     /// 等待用户输入
     display_caption( "End: Press a key!" );

     waitKey(0);
     return 0;
 }

 int display_caption( char* caption )
 {
   dst = Mat::zeros( src.size(), src.type() );
   putText( dst, caption,
            Point( src.cols/4, src.rows/2),
            CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );

   imshow( window_name, dst );
   int c = waitKey( DELAY_CAPTION );
   if( c >= 0 ) { return -1; }
   return 0;
  }

  int display_dst( int delay )
  {
    imshow( window_name, dst );
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
  }