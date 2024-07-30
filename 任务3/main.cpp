#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main() {
    // 加载三个样本图像
    Mat sampleImage1 = imread("/home/lty/第四次考核/dataset_任务三/template/template_1.jpg", IMREAD_GRAYSCALE);
    Mat sampleImage2 = imread("/home/lty/第四次考核/dataset_任务三/template/template_2.jpg", IMREAD_GRAYSCALE);
    Mat sampleImage3 = imread("/home/lty/第四次考核/dataset_任务三/template/template_3.jpg", IMREAD_GRAYSCALE);

    if (sampleImage1.empty() || sampleImage2.empty() || sampleImage3.empty()) {
        cerr << "Failed to load sample images." << endl;
        return -1;
    }

    // 初始化SIFT特征检测器和描述符提取器
    Ptr<SIFT> sift = SIFT::create();

    // 检测并计算样本图像的关键点和描述符
    vector<KeyPoint> keypoints_sample1, keypoints_sample2, keypoints_sample3;
    Mat descriptors_sample1, descriptors_sample2, descriptors_sample3;
    sift->detectAndCompute(sampleImage1, noArray(), keypoints_sample1, descriptors_sample1);
    sift->detectAndCompute(sampleImage2, noArray(), keypoints_sample2, descriptors_sample2);
    sift->detectAndCompute(sampleImage3, noArray(), keypoints_sample3, descriptors_sample3);

    // 获取目标图像文件夹中的所有图像文件
    string targetFolderPath = "/home/lty/第四次考核/dataset_任务三/archive";
    vector<string> targetImagePaths;
    for (const auto& entry : fs::directory_iterator(targetFolderPath)) {
        if (entry.is_regular_file()) {
            targetImagePaths.push_back(entry.path().string());
        }
    }

    // 创建结果文件夹
    string resultFolderPath = "/home/lty/第四次考核/dataset_任务三/results";
    fs::create_directory(resultFolderPath);

    // 对每个目标图像进行三次特征匹配并保存匹配结果
    for (const auto& targetImagePath : targetImagePaths) {
        // 加载目标图像
        Mat targetImage = imread(targetImagePath, IMREAD_GRAYSCALE);
        if (targetImage.empty()) {
            cerr << "Failed to load target image: " << targetImagePath << endl;
            continue;
        }

        // 检测目标图像的关键点和计算描述符
        vector<KeyPoint> keypoints_target;
        Mat descriptors_target;
        sift->detectAndCompute(targetImage, noArray(), keypoints_target, descriptors_target);

        // 使用Brute-Force匹配器进行三次特征匹配并保存匹配结果
        for (int i = 0; i < 3; ++i) {
            vector<vector<DMatch>> knn_matches;
            BFMatcher matcher(NORM_L2);
            Mat descriptors_sample;
            vector<KeyPoint> keypoints_sample;
            Mat sampleImage;

            // 根据不同的i值，选择不同的样本图像和描述子
            switch(i) {
                case 0:
                    descriptors_sample = descriptors_sample1;
                    keypoints_sample = keypoints_sample1;
                    sampleImage = sampleImage1;
                    break;
                case 1:
                    descriptors_sample = descriptors_sample2;
                    keypoints_sample = keypoints_sample2;
                    sampleImage = sampleImage2;
                    break;
                case 2:
                    descriptors_sample = descriptors_sample3;
                    keypoints_sample = keypoints_sample3;
                    sampleImage = sampleImage3;
                    break;
                default:
                    break;
            }

            // 进行特征匹配
            matcher.knnMatch(descriptors_sample, descriptors_target, knn_matches, 2);

            // 进行比率测试以筛选好的匹配
            vector<DMatch> good_matches;
            float ratio_thresh = 0.7f;
            for (size_t j = 0; j < knn_matches.size(); j++) {
                if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance) {
                    good_matches.push_back(knn_matches[j][0]);
                }
            }

            // 绘制匹配结果
            Mat img_matches;
            drawMatches(sampleImage, keypoints_sample, targetImage, keypoints_target, good_matches, img_matches);

            // 保存匹配结果
            string targetImageName = fs::path(targetImagePath).filename().string();
            string resultImageName = resultFolderPath + "/match_" + targetImageName + "_sample" + to_string(i+1) + ".jpg";
            imwrite(resultImageName, img_matches);

            cout << "Saved match result: " << resultImageName << endl;
        }
    }

    cout << "All match results have been saved in the 'results' folder." << endl;

    return 0;
}






















