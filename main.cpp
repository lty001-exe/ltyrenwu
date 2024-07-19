#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>

// 定义合理的最大图像尺寸
const int MAX_IMAGE_WIDTH = 10000;
const int MAX_IMAGE_HEIGHT = 10000;

struct ImageParams {
    std::string imageName;
    std::string imgScale; // 存储缩放因子或尺寸
    std::string interpolationMethod;
    int imgHorizontal;
    int imgVertical;
    std::string rotationCenter;
    double rotationAngle;
};

// 根据字符串获取OpenCV插值方法
int getInterpolationMethod(const std::string& interpMethod) {
    if (interpMethod == "NEAREST")
        return cv::INTER_NEAREST;
    else if (interpMethod == "LINEAR")
        return cv::INTER_LINEAR;
    // 根据需要可以添加更多插值方法
    return cv::INTER_LINEAR; // 默认使用双线性插值
}

// 解析img_scale字段，返回尺寸
cv::Size parseScale(const std::string& scaleStr, const cv::Size& originalSize) {
    try {
        int width, height;
        if (std::sscanf(scaleStr.c_str(), "%d,%d", &width, &height) == 2) {
            // 如果成功解析出两个整数，返回尺寸
            if (width > 0 && height > 0 && width <= MAX_IMAGE_WIDTH && height <= MAX_IMAGE_HEIGHT) {
                return cv::Size(width, height);
            } else {
                throw std::invalid_argument("Width and height must be greater than zero and less than maximum allowed size");
            }
        } else {
            // 否则是单个缩放因子
            double scale = std::stod(scaleStr);
            if (scale <= 0) {
                throw std::invalid_argument("Scale must be greater than zero");
            }
            int newWidth = static_cast<int>(originalSize.width * scale);
            int newHeight = static_cast<int>(originalSize.height * scale);
            if (newWidth <= 0 || newHeight <= 0 || newWidth > MAX_IMAGE_WIDTH || newHeight > MAX_IMAGE_HEIGHT) {
                throw std::invalid_argument("Resulting dimensions must be greater than zero and less than maximum allowed size");
            }
            return cv::Size(newWidth, newHeight);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing scale value: " << e.what() << " (value: " << scaleStr << ")" << std::endl;
        return originalSize; // 返回原始尺寸以继续处理
    }
}

// 从 CSV 文件中读取参数
std::vector<ImageParams> readParameters(const std::string& csvFile) {
    std::vector<ImageParams> params;
    std::ifstream file(csvFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csvFile << std::endl;
        return params;
    }

    std::string line;
    std::getline(file, line); // 读取并忽略标题行

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ImageParams param;
        std::string field;
        std::getline(ss, param.imageName, ',');
        std::getline(ss, param.imgScale, ',');
        std::getline(ss, param.interpolationMethod, ',');
        ss >> param.imgHorizontal;
        ss.ignore(1); // 跳过逗号
        ss >> param.imgVertical;
        ss.ignore(1); // 跳过逗号
        std::getline(ss, param.rotationCenter, ',');
        ss >> param.rotationAngle;

        if (!param.imageName.empty()) {
            params.push_back(param);
        }
    }

    file.close();
    return params;
}

int main() {
    const std::string imageDir = "/home/lty/第四次考核/dataset_任务一"; // 图像所在目录
    const std::string outputDir = "/home/lty/第四次考核/任务一结果"; // 结果保存目录
    const std::string csvFile = "/home/lty/第四次考核/experiment1.csv"; // CSV 文件路径

    // 读取参数
    std::vector<ImageParams> params = readParameters(csvFile);

    // 遍历处理每张图像
    for (const auto& param : params) {
        std::string imageName = imageDir + "/" + param.imageName; // 构造图像文件的完整路径

        cv::Mat src = cv::imread(imageName, cv::IMREAD_UNCHANGED); // 读取图像

        if (src.empty()) {
            std::cerr << "Could not open or find the image " << imageName << std::endl;
            continue;
        }

        // 缩放图像
        cv::Mat scaled;
        cv::Size newSize = parseScale(param.imgScale, src.size());
        try {
            cv::resize(src, scaled, newSize, 0, 0, getInterpolationMethod(param.interpolationMethod));
        } catch (const std::exception& e) {
            std::cerr << "Error resizing image: " << e.what() << std::endl;
            continue;
        }

        // 旋转图像
        cv::Mat rotated;
        cv::Point2f center;
        if (param.rotationCenter == "center") {
            center = cv::Point2f(scaled.cols / 2.0, scaled.rows / 2.0);
        } else if (param.rotationCenter == "origin") {
            center = cv::Point2f(0.0, 0.0);
        }
        double scale = 1.0; // 根据需要调整缩放比例
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, param.rotationAngle, scale);
        try {
            cv::warpAffine(scaled, rotated, rotationMatrix, scaled.size(), getInterpolationMethod(param.interpolationMethod), cv::BORDER_REPLICATE);
        } catch (const std::exception& e) {
            std::cerr << "Error rotating image: " << e.what() << std::endl;
            continue;
        }

        // 平移图像
        cv::Mat translated;
        cv::Mat translationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, param.imgHorizontal, 0, 1, param.imgVertical);
        try {
            cv::warpAffine(rotated, translated, translationMatrix, rotated.size(), getInterpolationMethod(param.interpolationMethod), cv::BORDER_REPLICATE);
        } catch (const std::exception& e) {
            std::cerr << "Error translating image: " << e.what() << std::endl;
            continue;
        }

        // 构造保存结果的文件名
        std::string outputName = outputDir + "/" + "processed_" + param.imageName;

        // 保存处理后的图像
        bool isSaved = cv::imwrite(outputName, translated);
        if (!isSaved) {
            std::cerr << "Failed to write image to file: " << outputName << std::endl;
        } else {
            std::cout << "Image saved: " << outputName << std::endl;
        }
    }

    std::cout << "Processing completed." << std::endl;
    return 0;
}