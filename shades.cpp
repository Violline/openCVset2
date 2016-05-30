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
vector<String> image;
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

	Mat gray, morph, morph2, morph3, thresh;

	cvtColor(SourceImg, gray, CV_BGR2GRAY);
	blur(gray, gray, Size(5, 5));
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(gray, morph, 0, kernel);
	Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	morphologyEx(morph, morph2, 1, kernel2);

	threshold(morph2, thresh, 45, 255, 0); //90 good
	Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
	morphologyEx(thresh, morph3, 3, kernel3);

	Canny(morph3, edges, 100, 300, 3);

	return edges;
}

vector<int> imgArea(vector<vector<Point>> contours) {

	vector<int> draw_area;
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		draw_area.push_back(area);
		cout << "area: " << draw_area.at(i) << endl;
	}
	return draw_area;
}

void sort(vector<int> draw_area) {

	int c = 0;

	for (int i = 0; i < draw_area.size(); i++) {
		int a = draw_area.at(i);

		if (c == 0) {

			if (a > 700 & a < 800 || a>1000 & a < 1300)
			{
				noveltyDetect.push_back(0);
				c++;
			}

			else if (a > 900 & a < 1000 || a>1300)
			{
				noveltyDetect.push_back(1);
				c++;
			}
		}
	}

}

void training(vector<String>image) {

	Mat SourceImg;
	for (size_t k = 0; k<image.size(); ++k)
	{
		SourceImg = imread(image[k]);
		if (SourceImg.empty()) continue; //only proceed if sucsessful

		imgProcessing(SourceImg);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

		vector<int> draw_area;
		draw_area = imgArea(contours);
		/*
		for (int i = 1; i < draw_area.size(); i++) {
			cout << "area: " << draw_area.at(i) << endl;
		}
		cout << "next" << endl;*/

		imshow("edges", edges);
		waitKey(0);
		sort(draw_area);
	}
}

int main()
{
	dataOpen3();
	image.size();
	training(image);
	cout << "im here" << endl;
	cout << "image size: " << image.size() << endl;
	cout << "novelty detect size: " << noveltyDetect.size() << endl;

	txtWrite(3);
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
