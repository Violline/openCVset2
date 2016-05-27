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

Mat SourceImg, edges;

int figure_no;

vector<Point2f> fig_mass_center;
vector<int> fig_area;

vector<Mat> training_data;
vector<Mat> testing_data;
vector<Mat> novelty_data;

vector<String> image;
vector<Mat> dataa;

vector<String> dataOpen1() {
	string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set2\\TRAINING_DATA\\*.jpg"); // sciezka do uzupelnienia
	cv::glob(path, image, true); // recurse
	return image;
}
vector<String> dataOpen2() {
	string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set2\\TESTING_DATA\\*.jpg"); // sciezka do uzupelnienia
	cv::glob(path, image, true); // recurse
	return image;
}
vector<String> dataOpen3() {
	string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set2\\NOVELTY_DATA\\*.jpg"); // sciezka do uzupelnienia
	cv::glob(path, image, true); // recurse
	return image;
}

void txtWrite(int no) {
	ofstream txtFile;
	switch (no) {
	case 1:
		txtFile.open("C:\\Users\\Olka\\Desktop\\TRAINING.txt"); //przygotowanie pliku tekstowego
		break;
	case 2:
		txtFile.open("C:\\Users\\Olka\\Desktop\\TESTING.txt"); //przygotowanie pliku tekstowego
		break;
	case 3:
		txtFile.open("C:\\Users\\Olka\\Desktop\\NOVELTY.txt"); //przygotowanie pliku tekstowego
		break;
	}

	txtFile << left << setw(10) << "no.: ";
	txtFile << left << setw(15) << "figure name: ";
	txtFile << left << setw(15) << "figure area: ";
	txtFile << left << setw(15) << "figure mass center: " << endl;

	for (size_t k = 0; k < image.size(); ++k) {
		string fileName = image[k].substr(image[k].find_last_of('\\') + 1);

		txtFile << left << setw(10) << k + 1;
		txtFile << left << setw(15) << fileName;
		txtFile << left << setw(15) << fig_area.at(k);
		txtFile << left << setw(15) << fig_mass_center.at(k) << endl;
	}
	txtFile.close();
}

Mat imgProcessing(Mat SourceImg) {
	
	Mat gray, morph, morph2, thresh;

	cvtColor(SourceImg, gray, CV_BGR2GRAY);
	blur(gray, gray, Size(5, 5));
	Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 6));
	morphologyEx(gray, morph, 0, kernel);
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(6, 6));
	morphologyEx(morph, morph2, 2, kernel2);

	threshold(morph2, thresh, 120, 255, 0); //90 good
	Canny(thresh, edges, 100, 300, 3);
	return edges;
}

int imgArea(vector<vector<Point>> contours) {
	
	int area_max = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);

		if (area > area_max) {
			area_max = area;
			figure_no = i;
		}
	}

	fig_area.push_back(area_max);
	area_max = 0;
	return figure_no;
}

void imgMoments(vector<vector<Point>> contours, vector<Vec4i> hierarchy) {
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	int a = mc[figure_no].x; //with bound checking .at(i).x ??wtf??
	int b = mc[figure_no].y;
	fig_mass_center.push_back(Point2f(a, b));
	
	RNG rng(time(NULL));
	Mat drawing = Mat::zeros(edges.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i], 4, color, -1, 8, 0);
	}
	
	namedWindow("WINDOW", CV_WINDOW_AUTOSIZE);
	imshow("WINDOW", drawing);
}


void training(vector<String>image) {

	//vector<Mat> data;
	for (size_t k = 0; k<image.size(); ++k)
	{
		SourceImg = imread(image[k]);
		if (SourceImg.empty()) continue; //only proceed if sucsessful

		imgProcessing(SourceImg);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

		imgArea(contours);
		imgMoments(contours, hierarchy);

		Rect boundRect;
		boundRect = boundingRect(Mat(contours[figure_no]));

		Mat roi = SourceImg(boundRect);
		imshow("Example1", roi);

		dataa.push_back(roi);
	}
}


int main()
{
	dataOpen1();
	training(image);
	training_data = dataa;
	txtWrite(1);
	dataOpen2();
	training(image);
	testing_data = dataa;
	txtWrite(2);
	dataOpen3();
	training(image);
	novelty_data = dataa;
	txtWrite(3);
	
	waitKey(0);
}
