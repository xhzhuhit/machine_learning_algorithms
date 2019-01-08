#include <opencv2/opencv.hpp>
#include <fstream>
#include <deque>

std::deque<float> _ratio_list;
cv::Ptr<cv::ml::SVM> _svm;
int _row = 305;
int _col = 50;

int main() {
    std::ifstream fin;
    fin.open("/home/zhuxiaohui/new/localization/config/lanechange_train_data.txt");
    cv::Mat train_data(_row, _col, CV_32FC1);
    cv::Mat train_label(_row, 1, CV_32SC1);
    float tmpfloat = 0.0;
    int tmpint = 0;
    for (int i = 0; i < _row; ++i) {
        for (int j = 0; j < _col; ++j) {
            fin >> tmpfloat;
            //std::cout << "ratio" << j << ":  " << tmpfloat << std::endl;
            train_data.at<float>(i, j) = tmpfloat;
        }
        fin >> tmpint;
        std::cout << "label" << i << ":  " << tmpint << " ";
        train_label.at<int>(i, 1) = tmpint;
    }
    _svm = cv::ml::SVM::create();
    //cv::ml::SVM::params;
    //params.svm_type = cv::ml::SVM::C_SVC; 
    //params.kernel_type = cv::ml::SVM::LINEAR;
    //_svm.train($train_data, &train_label, NULL, NULL, params);
    _svm->setType(cv::ml::SVM::Types::C_SVC);
    _svm->setKernel(cv::ml::SVM::KernelTypes::RBF);
    _svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    _svm->train(train_data, cv::ml::SampleTypes::ROW_SAMPLE, train_label);
    std::cout << "SVM init Done !"<< std::endl;
 
    int true_2_true = 0;
    int false_2_true = 0;  
    for (int i = 0; i < _row; ++i) { 
        cv::Mat sample = train_data.rowRange(i, i+1).clone();
        std::cout << sample.size() << std::endl; 
        float response = _svm->predict(sample);
        std::cout << i << "th biandaolema:" << int(response)  << "  lable: " 
                << train_label.at<int>(i, 1) << std::endl;
        if (int(response) == 1 && train_label.at<int>(i, 1) == 0) {
            ++false_2_true;
        } 
        if (int(response) == 1 && train_label.at<int>(i, 1) == 0) {
            ++true_2_true;
        } 
    }
    std::cout << "true_2_true: " << true_2_true 
            << "false_2_true: "  << false_2_true << std::endl;
    return 1;
    //return response>=0?true:false;
}
