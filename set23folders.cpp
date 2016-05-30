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

Mat edges;
int figure_no;

vector<Point2f> fig_mass_center;
vector<int> fig_area;

vector<Mat> training_data;
vector<Mat> testing_data;
vector<Mat> novelty_data;

vector<String> image;
vector<Mat> dataa;
vector<int> draw_area;
vector<int> noveltyDetect;

vector<String> dataOpen1() {
	string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set2\\TESTING_DATA_ROI\\*.jpg"); // sciezka do uzupelnienia
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
		txtFile.open("C:\\Users\\Olka\\Desktop\\TEST.txt"); //przygotowanie pliku tekstowego
		break;
	case 2:
		txtFile.open("C:\\Users\\Olka\\Desktop\\TESTING.txt"); //przygotowanie pliku tekstowego
		break;
	case 3:
		txtFile.open("C:\\Users\\Olka\\Desktop\\NOVELTY.txt"); //przygotowanie pliku tekstowego
		break;
	}

	txtFile << left << setw(10) << "No.: ";
	txtFile << left << setw(15) << "Figure name: ";
	txtFile << left << setw(15) << "Novelty detection: \n" << endl;

	for (size_t k = 0; k < noveltyDetect.size(); ++k) {
		string fileName = image[k].substr(image[k].find_last_of('\\') + 1);
		if (noveltyDetect.at(k) == 1) {
			txtFile << left << setw(10) << k + 1;
			txtFile << left << setw(15) << fileName;
			txtFile << left << setw(15) << "1 - Novelty detected" << endl;
		}
		else {
			txtFile << left << setw(10) << k + 1;
			txtFile << left << setw(15) << fileName;
			txtFile << left << setw(15) << "0 - Nonnovelty" << endl;
		}
	}
	txtFile.close();
}

Mat imgProcessing(Mat SourceImg) {

	Mat gray, morph, morph2, morph3, morph4, thresh;

	cvtColor(SourceImg, gray, CV_BGR2GRAY);
	blur(gray, gray, Size(5, 5));
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(gray, morph, 0, kernel);
	Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(morph, morph2, 1, kernel2);

	threshold(morph2, thresh, 45, 255, 0); //90 good
	Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
	morphologyEx(thresh, morph3, 3, kernel3); 
	//Mat kernel4 = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	//morphologyEx(morph3, morph4, 1, kernel3);

	Canny(morph3, edges, 100, 300, 3);

	Mat imgHSV;
	Mat imgThresholded;
	
	//namedWindow("WINDOW", CV_WINDOW_AUTOSIZE);
	//imshow("WINDOW", edges);
	//waitKey();

	return edges;


}

int imgArea(vector<vector<Point>> contours) {

	int area_max = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		draw_area.push_back(area);
		//cout << "area: " << draw_area.at(i) << endl;

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
		if (draw_area.at(i) > 2000) {
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
			circle(drawing, mc[i], 4, color, -1, 8, 0);
		}
	}

	//namedWindow("WINDOW", CV_WINDOW_AUTOSIZE);
	//imshow("WINDOW", drawing);
	//waitKey(0);
}

void sort() {

	for (size_t k = 0; k<image.size(); ++k)
	{
		for (int i = 0; i < draw_area.size(); i++) {
			int a = draw_area.at(i);
			if (a>900 & a<1000 || a>1300)
			{
				noveltyDetect.push_back(1);
				break;
			}
			else if(a>700 & a<800 || a>1000 & a<1300)
			{
				noveltyDetect.push_back(0);
				break;
			}
		}
	}

}

void training(vector<String>image) {

	Mat SourceImg;
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
		sort();

		Rect boundRect;
		boundRect = boundingRect(Mat(contours[figure_no]));

		Mat roi = SourceImg(boundRect);
		//imshow("Example1", roi);

		dataa.push_back(roi);
	}
}


int main()
{
	dataOpen1();
	training(image);
	training_data = dataa;
	txtWrite(1);
	/*dataOpen2();
	training(image);
	testing_data = dataa;
	txtWrite(2);
	dataOpen3();
	training(image);
	novelty_data = dataa;
	txtWrite(3);*/

	waitKey(0);
}
