#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;

Mat SourceImg;
Mat OurImage, thresh, gray, edges, morph, morph2;

int area, figure_no;
int area_max = 0;

vector<Point2f> fig_mass_center;
vector<int> fig_area;


int main()
{
	string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set2\\NOVELTY_DATA\\*.jpg"); // sciezka do uzupelnienia
													
	vector<String> image;
	vector<Mat> data;
	cv::glob(path, image, true); // recurse
	for (size_t k = 0; k<image.size(); ++k)
	{
		SourceImg = imread(image[k]);
		string fileName = image[k].substr(image[k].find_last_of('\\') + 1);
		if (SourceImg.empty()) continue; //only proceed if sucsessful
		cout << "Zdjecia zaladowane o Pani" << endl;

		cvtColor(SourceImg, gray, CV_BGR2GRAY);
		//blur(gray, gray, Size(5, 5));
		//Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 6));
		//morphologyEx(gray, morph, 0, kernel);
		Mat kernel2 = getStructuringElement(MORPH_RECT, Size(6, 6));
		morphologyEx(gray, morph2, 3, kernel2);

		threshold(morph2, thresh, 80, 255, 0); //90 good
		Canny(thresh, edges, 100, 300, 3);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));
		RNG rng(time(NULL));

		vector<Moments> mu(contours.size());
		vector<Point2f> mc(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			mu[i] = moments(contours[i], false);
			mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}

		int a = mc[figure_no].x; //with bound checking .at(i).x ??wtf??
		int b = mc[figure_no].y;
		cout << "a: " << a << endl;
		cout << "b: " << b << endl;
		fig_mass_center.push_back(Point2f(a, b));

		Mat drawing = Mat::zeros(edges.size(), CV_8UC3);

		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
			circle(drawing, mc[i], 4, color, -1, 8, 0);

		}

		/// Separate the image in 3 places ( B, G and R )
		vector<Mat> bgr_planes;
		split(SourceImg, bgr_planes);

		/// Establish the number of bins
		int histSize = 256;

		/// Set the ranges ( for B,G,R) )
		float range[] = { 0, 256 };
		const float* histRange = { range };

		bool uniform = true; bool accumulate = false;

		Mat b_hist, g_hist, r_hist;

		/// Compute the histograms:
		calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

		// Draw the histograms for B, G and R
		int hist_w = 512; int hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		/// Draw for each channel
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
		}

		/// Display
		namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
		imshow("calcHist Demo", histImage);

		waitKey(0);


		namedWindow("WINDOW", CV_WINDOW_AUTOSIZE);
		imshow("WINDOW", thresh);

		// Showing the result
		data.push_back(SourceImg);
	}

	waitKey(0);
}
