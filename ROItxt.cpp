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
	ofstream txtFile;
	string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set2\\TRAINING_DATA\\*.jpg"); // sciezka do uzupelnienia
	txtFile.open ("C:\\Users\\Olka\\Desktop\\TRAINING.txt"); //przygotowanie pliku tekstowego
	//string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set1\\TESTING_DATA\\*.jpg"); // sciezka do uzupelnienia
	//txtFile.open("C:\\Users\\Olka\\Desktop\\TESTING.txt"); //przygotowanie pliku tekstowego
														   //string path("C:\\Users\\Olka\\Desktop\\NoveltyDetectionData\\Set1\\NOVELTY_DATA\\*.jpg"); // sciezka do uzupelnienia
														   //txtFile.open ("C:\\Users\\Olka\\Desktop\\NOVELTY.txt"); //przygotowanie pliku tekstowego


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
		blur(gray, gray, Size(5, 5));
		Mat kernel = getStructuringElement(MORPH_RECT, Size(6, 6));
		morphologyEx(gray, morph, 0, kernel);
		Mat kernel2 = getStructuringElement(MORPH_RECT, Size(6, 6));
		morphologyEx(morph, morph2, 2, kernel2);

		threshold(morph2, thresh, 110, 255, 0); //90 good
		Canny(thresh, edges, 100, 300, 3);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));
		RNG rng(time(NULL));

		for (int i = 0; i < contours.size(); i++)
		{
			area = contourArea(contours[i]);
			cout << "area: " << area << endl;

			if (area > area_max) {
				area_max = area;
				figure_no = i;
				cout << "area_max: " << area_max << endl;
			}
		}
		
		fig_area.push_back(area_max);
		
		area_max = 0;
		cout << "figure_no: " << figure_no << endl;

		//for (int i = 0; i<contours.size(); i++)
		//cout << "hierarchy: " << contours[figure_no] << endl;

		/// Get the moments
		///  Get the mass centers:
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

		Rect boundRect;
		boundRect = boundingRect(Mat(contours[figure_no]));

		cout << "boundrect: " << boundRect << endl;

		Mat roi = SourceImg(boundRect);
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
		/*
		if (waitKey(30) >= 0)
			destroyAllWindows();

		waitKey(0);*/

		// Showing the result
		data.push_back(SourceImg);
	}

	txtFile << left << setw(10) << "no.: ";
	txtFile << left << setw(15) << "figure name: ";
	txtFile << left << setw(15) << "figure area: ";
	txtFile << left << setw(15) << "figure mass center: " << endl;

	for (size_t k = 0; k < image.size(); ++k) {
		cout << "figure area " << k << ": " << fig_area.at(k) << endl;
		cout << "figure center: " << fig_mass_center.at(k) << endl;
		string fileName = image[k].substr(image[k].find_last_of('\\') + 1);

		txtFile << left << setw(10) << k+1;
		txtFile << left << setw(15) << fileName;
		txtFile << left << setw(15) << fig_area.at(k);
		txtFile << left << setw(15) << fig_area.at(k) << endl;
	}

	txtFile.close();
	waitKey(0);
}
