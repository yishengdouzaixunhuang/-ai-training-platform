// C++ ONNX inference for semantic segmentation
// Build: cmake -B build && cmake --build build --config Release
// Requires: ONNX Runtime SDK

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

struct PredictResult {
    std::vector<int64_t> mask;  // H*W flat
    int width, height;
    double elapsed_ms;
};

class Predictor {
    Ort::Env env;
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    int tile_size = 512;
    int overlap = 64;

public:
    Predictor(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "cpp_infer") {
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try {
            opts.AppendExecutionProvider_CUDA(0);  // GPU first
        } catch (...) {
            std::cerr << "CUDA not available, using CPU" << std::endl;
        }
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        input_names.push_back(session->GetInputName(0, alloc));
        output_names.push_back(session->GetOutputName(0, alloc));
    }

    PredictResult predict(const std::string& image_path) {
        auto t0 = std::chrono::high_resolution_clock::now();

        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Cannot read: " << image_path << std::endl;
            return {};
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        int h = img.rows, w = img.cols;

        cv::Mat pred(h, w, CV_32SC1, cv::Scalar(0));

        if (w <= tile_size && h <= tile_size) {
            // Single pass
            cv::Mat tile;
            cv::resize(img, tile, cv::Size(tile_size, tile_size));
            cv::Mat blob = cv::dnn::blobFromImage(tile, 1.0/255.0, cv::Size(), cv::Scalar(), true, false);

            std::vector<int64_t> shape = {1, 3, tile_size, tile_size};
            Ort::Value input = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                (float*)blob.data, blob.total(), shape.data(), shape.size());

            auto output = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input, 1, output_names.data(), 1);
            float* out_data = output[0].GetTensorMutableData<float>();
            auto out_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
            int num_classes = out_shape[1];

            cv::Mat pred_small(tile_size, tile_size, CV_32SC1);
            for (int y = 0; y < tile_size; y++)
                for (int x = 0; x < tile_size; x++) {
                    int max_c = 0;
                    float max_v = out_data[y * tile_size + x];
                    for (int c = 1; c < num_classes; c++) {
                        float v = out_data[c * tile_size * tile_size + y * tile_size + x];
                        if (v > max_v) { max_v = v; max_c = c; }
                    }
                    pred_small.at<int>(y, x) = max_c;
                }
            cv::resize(pred_small, pred, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
        } else {
            // Tiled prediction with overlap
            int stride = tile_size - overlap;
            cv::Mat pred_sum(h, w, CV_32FC1, cv::Scalar(0));
            cv::Mat count_map(h, w, CV_32FC1, cv::Scalar(0));

            std::vector<cv::Mat> tiles;
            std::vector<cv::Rect> rois;
            for (int y = 0; y < h; y += stride) {
                for (int x = 0; x < w; x += stride) {
                    int x2 = std::min(x + tile_size, w);
                    int y2 = std::min(y + tile_size, h);
                    int x1 = std::max(0, x2 - tile_size);
                    int y1 = std::max(0, y2 - tile_size);
                    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
                    cv::Mat tile;
                    img(roi).copyTo(tile);
                    if (tile.cols < tile_size || tile.rows < tile_size) {
                        cv::copyMakeBorder(tile, tile, 0, tile_size - tile.rows, 0, tile_size - tile.cols, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
                    }
                    tiles.push_back(tile);
                    rois.push_back(roi);
                }
            }

            int batch_size = 16;
            for (size_t i = 0; i < tiles.size(); i += batch_size) {
                size_t n = std::min(tiles.size() - i, (size_t)batch_size);
                std::vector<cv::Mat> batch_tiles(tiles.begin() + i, tiles.begin() + i + n);

                cv::Mat blob = cv::dnn::blobFromImages(batch_tiles, 1.0/255.0, cv::Size(tile_size, tile_size), cv::Scalar(), true, false);
                std::vector<int64_t> shape = {(int64_t)n, 3, tile_size, tile_size};
                Ort::Value input = Ort::Value::CreateTensor<float>(
                    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                    (float*)blob.data, blob.total(), shape.data(), shape.size());

                auto output = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input, 1, output_names.data(), 1);
                float* out_data = output[0].GetTensorMutableData<float>();
                auto out_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
                int num_classes = out_shape[1];

                for (size_t j = 0; j < n; j++) {
                    auto& roi = rois[i + j];
                    for (int y = 0; y < roi.height; y++) {
                        for (int x = 0; x < roi.width; x++) {
                            int max_c = 0;
                            float max_v = out_data[j * num_classes * tile_size * tile_size + y * tile_size + x];
                            for (int c = 1; c < num_classes; c++) {
                                float v = out_data[j * num_classes * tile_size * tile_size + c * tile_size * tile_size + y * tile_size + x];
                                if (v > max_v) { max_v = v; max_c = c; }
                            }
                            pred_sum.at<float>(roi.y + y, roi.x + x) += max_c;
                            count_map.at<float>(roi.y + y, roi.x + x) += 1;
                        }
                    }
                }
            }
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    pred.at<int>(y, x) = count_map.at<float>(y, x) > 0 ? (int)(pred_sum.at<float>(y, x) / count_map.at<float>(y, x) + 0.5f) : 0;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        PredictResult res;
        res.width = w; res.height = h;
        res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        res.mask.assign((int64_t*)pred.data, (int64_t*)pred.data + w * h);

        // Save result
        fs::path out_path = fs::path(image_path).parent_path().parent_path() / "outputs" / (fs::path(image_path).stem().string() + "_pred.png");
        fs::create_directories(out_path.parent_path());
        cv::Mat pred_8u;
        pred.convertTo(pred_8u, CV_8U);
        cv::imwrite(out_path.string(), pred_8u);

        std::cout << image_path << ": " << res.elapsed_ms << "ms" << std::endl;
        return res;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: cpp_infer <model.onnx> <image1> [image2...]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "Loading " << model_path << "..." << std::endl;
    Predictor pred(model_path);

    for (int i = 2; i < argc; i++) {
        pred.predict(argv[i]);
    }
    return 0;
}
