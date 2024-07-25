#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

class ImageProcessor {
public:
    void process(const string& input_folder, const string& output_folder) {
        for (const auto& entry : fs::directory_iterator(input_folder)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                Mat image = imread(entry.path().string());
                if (image.empty()) {
                    cout << "无法打开或找到图像：" << entry.path() << endl;
                    continue;
                }

                Mat roi = applyROI(image);
                Mat blurred = applyGaussianBlur(roi);
                Mat gray = convertToGray(blurred);
                Mat edges = detectEdges(gray);
                vector<Vec4i> lines = detectLines(edges);
                Mat result = drawLines(image.size(), lines);
                
                // 先膨胀后腐蚀
                Mat morphed = applyMorphology(result);

                // 保存霍夫变换结果
                string output_filename = output_folder + "/" + entry.path().stem().string() + "_hough_morphed.jpg";
                imwrite(output_filename, morphed);

                // 显示各步骤结果（可选）
                imshow("原图", image);
                imshow("ROI", roi);
                imshow("高斯模糊后的图像", blurred);
                imshow("灰度图", gray);
                imshow("边缘检测", edges);
                imshow("霍夫变换检测到的线条", result);
                imshow("膨胀腐蚀后的结果", morphed);
                waitKey(100); // 短暂显示结果
            }
        }
    }

private:
    // 提取感兴趣区域
    Mat applyROI(const Mat& image) {
        Mat mask = Mat::zeros(image.size(), CV_8UC1);

        // 定义多边形顶点
        Point points[4];
        points[0] = Point(0, image.rows);
        points[1] = Point(image.cols * 0.1, image.rows * 0.4);
        points[2] = Point(image.cols * 0.5, image.rows * 0.4);
        points[3] = Point(image.cols, image.rows);

        // 填充多边形
        fillConvexPoly(mask, points, 4, Scalar(255));

        // 应用掩码
        Mat roi;
        image.copyTo(roi, mask);
        return roi;
    }

    // 应用高斯模糊
    Mat applyGaussianBlur(const Mat& image) {
        Mat blurred;
        GaussianBlur(image, blurred, Size(5, 5), 0);
        return blurred;
    }

    // 转换为灰度图
    Mat convertToGray(const Mat& image) {
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        return gray;
    }

    // 边缘检测
    Mat detectEdges(const Mat& gray) {
        Mat edges;
        Canny(gray, edges, 200, 230);
        return edges;
    }

    // 检测线条
    vector<Vec4i> detectLines(const Mat& edges) {
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1.5, CV_PI / 180, 13, 16, 10);
        return lines;
    }

    // 在空白图像上绘制线条
    Mat drawLines(const Size& imageSize, const vector<Vec4i>& lines) {
        Mat result = Mat::zeros(imageSize, CV_8UC3); // 创建空白图像
        for (size_t i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            line(result, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
        }
        return result;
    }

    // 应用膨胀和腐蚀
    Mat applyMorphology(const Mat& image) {
        Mat dilated, eroded;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        dilate(image, dilated, kernel);
        erode(dilated, eroded, kernel);
        return eroded;
    }
};

int main() {
    string input_folder = "/home/lty/第四次考核/dataset_任务二";
    string output_folder = "/home/lty/第四次考核/任务二结果";

    // 检查输入文件夹是否存在
    if (!fs::exists(input_folder)) {
        cerr << "输入文件夹不存在：" << input_folder << endl;
        return -1;
    }

    // 确保输出文件夹存在
    fs::create_directories(output_folder);

    ImageProcessor processor;
    
    // 获取总文件数
    int total_files = count_if(fs::directory_iterator(input_folder), fs::directory_iterator{}, 
        [](const auto& entry) { return entry.path().extension() == ".jpg" || entry.path().extension() == ".png"; });
    
    cout << "开始处理 " << total_files << " 个文件..." << endl;

    processor.process(input_folder, output_folder);

    cout << "处理完成。结果保存在：" << output_folder << endl;

    return 0;
}





