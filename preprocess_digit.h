#ifndef PREPROCESS_DIGIT_H
#define PREPROCESS_DIGIT_H

#include <cstdint>
#include <string>
#include <vector>

struct GrayImage {
    int width;
    int height;
    std::vector<uint8_t> data;

    GrayImage();
    GrayImage(int h, int w, uint8_t value = 0);

    uint8_t& at(int y, int x);
    const uint8_t& at(int y, int x) const;
    bool empty() const;
};

struct BinaryImage {
    int width;
    int height;
    std::vector<uint8_t> data;

    BinaryImage();
    BinaryImage(int h, int w, uint8_t value = 0);

    uint8_t& at(int y, int x);
    const uint8_t& at(int y, int x) const;
};

struct BoundingBox {
    int min_x;
    int max_x;
    int min_y;
    int max_y;

    BoundingBox();
    BoundingBox(int minx, int maxx, int miny, int maxy);
};

struct ForegroundChoice {
    BinaryImage bw;
    std::string mode;
};

GrayImage load_bmp_as_grayscale(const std::string& path);
void save_pgm(const GrayImage& img, const std::string& path);

int otsu_threshold(const GrayImage& gray);
ForegroundChoice choose_foreground(const GrayImage& gray, int threshold);
BoundingBox bounding_box_from_binary(const BinaryImage& bw);
BoundingBox add_padding_to_box(BoundingBox box, int img_h, int img_w, int pad = 5);
GrayImage crop_gray(const GrayImage& img, const BoundingBox& box);
GrayImage emphasize_digit(const GrayImage& cropped_gray);
BoundingBox tight_box_from_grayscale_box(const GrayImage& img, int eps = 5);
GrayImage tight_box_from_grayscale(const GrayImage& img, int eps = 5);
GrayImage resize_to_mnist_box(const GrayImage& tight, int target_box_size, int& new_w, int& new_h);
GrayImage paste_to_canvas(const GrayImage& resized, int new_w, int new_h, int canvas_size = 64);
GrayImage center_of_mass_shift(const GrayImage& canvas, double target_x = 31.5, double target_y = 31.5);

GrayImage preprocess_roi_to_canvas_64(
    const GrayImage& input_gray,
    int pad = 5,
    int target_box_size = 48,
    int tight_eps = 5,
    bool verbose = false
);

#endif
