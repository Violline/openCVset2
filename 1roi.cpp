#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat OurImage, thresh, gray, edges, morph, morph2;

int main()
{
	string Destination = "C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set2\\TRAINING_DATA\\IMG_3547.jpg";
	OurImage = imread(Destination, CV_LOAD_IMAGE_COLOR);
	//string Dest = "C:\\Users\\Olka\\Documents\\STUDIA 3\\MD\\example.jpg";
	//OurImage = imread(Dest, CV_LOAD_IMAGE_COLOR);


	if (!OurImage.data)
	{
		printf("No image!");
		getchar();
		return -1;
	}

	cvtColor(OurImage, gray, CV_BGR2GRAY);
	blur(gray, gray, Size(5, 5));
	Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 6));
	morphologyEx(gray, morph, 0, kernel);
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(6, 6));
	morphologyEx(morph, morph2, 2, kernel2);

	threshold(morph2, thresh, 90, 255, 0); //90 good

	Canny(thresh, edges, 100, 300, 3);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));
	RNG rng(time(NULL));
	
	//for (int i = 0; i<contours.size(); i++)
	cout << "hierarchy: " << contours[1] << endl;
	
	/// Get the moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	int a = mu[1].m10 / mu[1].m00;
	int b = mu[1].m01 / mu[1].m00;

	Mat drawing = Mat::zeros(edges.size(), CV_8UC3);
	
	for (int i = 0; i < contours.size(); i++)
	{
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	circle(drawing, mc[i], 4, color, -1, 8, 0);
	
	}
	cout << "a: " << a << endl;
	cout << "b: " << b << endl;

	Mat roi = OurImage(Rect(a-110, b-80, 220, 160));
	imshow("Example1", roi);

	/*
	for (auto vec : contours)
		for (auto v : vec)
			cout << v << endl;
	for (auto vec : hierarchy)
		cout << vec << endl;

*/

	namedWindow("WINDOW", CV_WINDOW_AUTOSIZE);
	imshow("WINDOW", drawing);

	if (waitKey(30) >= 0)
	destroyAllWindows();

	waitKey(0);
}
