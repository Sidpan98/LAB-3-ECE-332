#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "preprocess_digit.h"

GrayImage::GrayImage() : width(0), height(0) {}

GrayImage::GrayImage(int h, int w, uint8_t value)
    : width(w), height(h), data(static_cast<size_t>(h) * w, value) {}

uint8_t& GrayImage::at(int y, int x) {
    return data[static_cast<size_t>(y) * width + x];
}

const uint8_t& GrayImage::at(int y, int x) const {
    return data[static_cast<size_t>(y) * width + x];
}

bool GrayImage::empty() const {
    return width <= 0 || height <= 0 || data.empty();
}

BinaryImage::BinaryImage() : width(0), height(0) {}

BinaryImage::BinaryImage(int h, int w, uint8_t value)
    : width(w), height(h), data(static_cast<size_t>(h) * w, value) {}

uint8_t& BinaryImage::at(int y, int x) {
    return data[static_cast<size_t>(y) * width + x];
}

const uint8_t& BinaryImage::at(int y, int x) const {
    return data[static_cast<size_t>(y) * width + x];
}

BoundingBox::BoundingBox() : min_x(0), max_x(0), min_y(0), max_y(0) {}

BoundingBox::BoundingBox(int minx, int maxx, int miny, int maxy)
    : min_x(minx), max_x(maxx), min_y(miny), max_y(maxy) {}

template <typename T>
T clamp_val(T v, T lo, T hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}


// struct GrayImage {
//     int width = 0;
//     int height = 0;
//     std::vector<uint8_t> data; // row-major

//     GrayImage() = default;

//     GrayImage(int h, int w, uint8_t value = 0)
//         : width(w), height(h), data(static_cast<size_t>(h) * w, value) {}

//     uint8_t& at(int y, int x) {
//         return data[static_cast<size_t>(y) * width + x];
//     }

//     const uint8_t& at(int y, int x) const {
//         return data[static_cast<size_t>(y) * width + x];
//     }

//     bool empty() const {
//         return width <= 0 || height <= 0 || data.empty();
//     }
// };

// struct BinaryImage {
//     int width = 0;
//     int height = 0;
//     std::vector<uint8_t> data; // 0 or 1

//     BinaryImage() = default;

//     BinaryImage(int h, int w, uint8_t value = 0)
//         : width(w), height(h), data(static_cast<size_t>(h) * w, value) {}

//     uint8_t& at(int y, int x) {
//         return data[static_cast<size_t>(y) * width + x];
//     }

//     const uint8_t& at(int y, int x) const {
//         return data[static_cast<size_t>(y) * width + x];
//     }
// };

// struct BoundingBox {
//     int min_x = 0;
//     int max_x = 0;
//     int min_y = 0;
//     int max_y = 0;
// };

// struct ForegroundChoice {
//     BinaryImage bw;
//     std::string mode;
// };

static uint16_t read_u16(std::ifstream& f) {
    uint8_t b0, b1;
    f.read(reinterpret_cast<char*>(&b0), 1);
    f.read(reinterpret_cast<char*>(&b1), 1);
    return static_cast<uint16_t>(b0 | (b1 << 8));
}

static uint32_t read_u32(std::ifstream& f) {
    uint8_t b0, b1, b2, b3;
    f.read(reinterpret_cast<char*>(&b0), 1);
    f.read(reinterpret_cast<char*>(&b1), 1);
    f.read(reinterpret_cast<char*>(&b2), 1);
    f.read(reinterpret_cast<char*>(&b3), 1);
    return static_cast<uint32_t>(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

static int32_t read_s32(std::ifstream& f) {
    return static_cast<int32_t>(read_u32(f));
}

// Supports uncompressed 8-bit grayscale BMP and 24-bit BMP.
GrayImage load_bmp_as_grayscale(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open BMP file: " + path);
    }

    char sig[2];
    f.read(sig, 2);
    if (sig[0] != 'B' || sig[1] != 'M') {
        throw std::runtime_error("Not a BMP file: " + path);
    }

    (void)read_u32(f);      // file size
    (void)read_u16(f);      // reserved1
    (void)read_u16(f);      // reserved2
    uint32_t pixel_offset = read_u32(f);

    uint32_t dib_size = read_u32(f);
    if (dib_size < 40) {
        throw std::runtime_error("Unsupported BMP DIB header size");
    }

    int32_t width = read_s32(f);
    int32_t height_signed = read_s32(f);
    uint16_t planes = read_u16(f);
    uint16_t bpp = read_u16(f);
    uint32_t compression = read_u32(f);
    uint32_t image_size = read_u32(f);
    (void)image_size;
    (void)read_s32(f);      // x ppm
    (void)read_s32(f);      // y ppm
    uint32_t colors_used = read_u32(f);
    (void)colors_used;
    (void)read_u32(f);      // important colors

    if (planes != 1) {
        throw std::runtime_error("Unsupported BMP: planes != 1");
    }
    if (compression != 0) {
        throw std::runtime_error("Unsupported BMP: compressed BMP not supported");
    }
    if (bpp != 8 && bpp != 24) {
        throw std::runtime_error("Unsupported BMP: only 8-bit and 24-bit supported");
    }

    const bool top_down = (height_signed < 0);
    const int height = std::abs(height_signed);
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid BMP dimensions");
    }

    GrayImage img(height, width, 0);

    // Palette for 8-bit BMP
    std::vector<std::array<uint8_t, 4>> palette;
    if (bpp == 8) {
        const std::streampos current = f.tellg();
        const std::streamoff palette_bytes = static_cast<std::streamoff>(pixel_offset) - current;
        const int palette_entries = static_cast<int>(palette_bytes / 4);
        palette.resize(std::max(0, palette_entries));

        for (int i = 0; i < palette_entries; ++i) {
            uint8_t bgrx[4];
            f.read(reinterpret_cast<char*>(bgrx), 4);
            palette[i][0] = bgrx[0];
            palette[i][1] = bgrx[1];
            palette[i][2] = bgrx[2];
            palette[i][3] = bgrx[3];
        }
    } else {
        f.seekg(pixel_offset, std::ios::beg);
    }

    const int row_bytes = ((width * bpp + 31) / 32) * 4;
    std::vector<uint8_t> row(static_cast<size_t>(row_bytes));

    for (int row_idx = 0; row_idx < height; ++row_idx) {
        f.read(reinterpret_cast<char*>(row.data()), row_bytes);
        if (!f) {
            throw std::runtime_error("Failed reading BMP pixel data");
        }

        const int y = top_down ? row_idx : (height - 1 - row_idx);

        if (bpp == 8) {
            for (int x = 0; x < width; ++x) {
                const uint8_t idx = row[x];
                uint8_t gray = idx;
                if (!palette.empty() && idx < palette.size()) {
                    const auto& p = palette[idx];
                    const float g = 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2];
                    gray = static_cast<uint8_t>(std::round(clamp_val(g, 0.0f, 255.0f)));
                }
                img.at(y, x) = gray;
            }
        } else { // 24-bit
            for (int x = 0; x < width; ++x) {
                const uint8_t b = row[3 * x + 0];
                const uint8_t g = row[3 * x + 1];
                const uint8_t r = row[3 * x + 2];
                const float gray_f = 0.114f * b + 0.587f * g + 0.299f * r;
                img.at(y, x) = static_cast<uint8_t>(std::round(clamp_val(gray_f, 0.0f, 255.0f)));
            }
        }
    }

    return img;
}

// Optional helper to inspect output on board/host without extra libraries.
void save_pgm(const GrayImage& img, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open output file: " + path);
    }
    out << "P5\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
}

int otsu_threshold(const GrayImage& gray) {
    std::array<int, 256> hist{};
    for (uint8_t v : gray.data) {
        hist[v]++;
    }

    const double total = static_cast<double>(gray.data.size());
    double sum_total = 0.0;
    for (int i = 0; i < 256; ++i) {
        sum_total += i * hist[i];
    }

    double sum_b = 0.0;
    double w_b = 0.0;
    double max_var = -1.0;
    int threshold = 0;

    for (int t = 0; t < 256; ++t) {
        w_b += hist[t];
        if (w_b == 0.0) {
            continue;
        }

        const double w_f = total - w_b;
        if (w_f == 0.0) {
            break;
        }

        sum_b += t * hist[t];
        const double m_b = sum_b / w_b;
        const double m_f = (sum_total - sum_b) / w_f;
        const double between_var = w_b * w_f * (m_b - m_f) * (m_b - m_f);

        if (between_var > max_var) {
            max_var = between_var;
            threshold = t;
        }
    }

    return threshold;
}

ForegroundChoice choose_foreground(const GrayImage& gray, int threshold) {
    BinaryImage bw_white_fg(gray.height, gray.width, 0);
    BinaryImage bw_black_fg(gray.height, gray.width, 0);

    size_t white_sum = 0;
    size_t black_sum = 0;

    for (int y = 0; y < gray.height; ++y) {
        for (int x = 0; x < gray.width; ++x) {
            const uint8_t v = gray.at(y, x);
            const uint8_t w = (v > threshold) ? 1 : 0;
            const uint8_t b = (v < threshold) ? 1 : 0;
            bw_white_fg.at(y, x) = w;
            bw_black_fg.at(y, x) = b;
            white_sum += w;
            black_sum += b;
        }
    }

    // Matches your Python:
    // if bw_white_fg.sum() < bw_black_fg.sum():
    //     return bw_white_fg, "white foreground"
    // else:
    //     return bw_black_fg, "dark foreground"
    if (white_sum < black_sum) {
        return {bw_white_fg, "white foreground"};
    } else {
        return {bw_black_fg, "dark foreground"};
    }
}

BoundingBox bounding_box_from_binary(const BinaryImage& bw) {
    bool found = false;
    BoundingBox box{bw.width, -1, bw.height, -1};

    for (int y = 0; y < bw.height; ++y) {
        for (int x = 0; x < bw.width; ++x) {
            if (bw.at(y, x) == 1) {
                found = true;
                box.min_x = std::min(box.min_x, x);
                box.max_x = std::max(box.max_x, x);
                box.min_y = std::min(box.min_y, y);
                box.max_y = std::max(box.max_y, y);
            }
        }
    }

    if (!found) {
        throw std::runtime_error("No foreground digit detected.");
    }

    return box;
}

BoundingBox add_padding_to_box(BoundingBox box, int img_h, int img_w, int pad) {
    box.min_x = std::max(box.min_x - pad, 0);
    box.max_x = std::min(box.max_x + pad, img_w - 1);
    box.min_y = std::max(box.min_y - pad, 0);
    box.max_y = std::min(box.max_y + pad, img_h - 1);
    return box;
}

GrayImage crop_gray(const GrayImage& img, const BoundingBox& box) {
    const int new_w = box.max_x - box.min_x + 1;
    const int new_h = box.max_y - box.min_y + 1;
    GrayImage out(new_h, new_w, 0);

    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            out.at(y, x) = img.at(box.min_y + y, box.min_x + x);
        }
    }

    return out;
}

GrayImage emphasize_digit(const GrayImage& cropped_gray) {
    GrayImage out(cropped_gray.height, cropped_gray.width, 0);

    float min_v = 255.0f;
    float max_v = 0.0f;
    std::vector<float> tmp(cropped_gray.data.size());

    for (size_t i = 0; i < cropped_gray.data.size(); ++i) {
        float v = 255.0f - static_cast<float>(cropped_gray.data[i]);
        tmp[i] = v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    for (float& v : tmp) {
        v -= min_v;
    }

    float new_max = 0.0f;
    for (float v : tmp) {
        new_max = std::max(new_max, v);
    }

    if (new_max > 0.0f) {
        for (size_t i = 0; i < tmp.size(); ++i) {
            float scaled = (tmp[i] / new_max) * 255.0f;
            out.data[i] = static_cast<uint8_t>(std::round(clamp_val(scaled, 0.0f, 255.0f)));
        }
    } else {
        std::fill(out.data.begin(), out.data.end(), 0);
    }

    return out;
}

BoundingBox tight_box_from_grayscale_box(const GrayImage& img, int eps) {
    bool found = false;
    BoundingBox box{img.width, -1, img.height, -1};

    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            if (img.at(y, x) > eps) {
                found = true;
                box.min_x = std::min(box.min_x, x);
                box.max_x = std::max(box.max_x, x);
                box.min_y = std::min(box.min_y, y);
                box.max_y = std::max(box.max_y, y);
            }
        }
    }

    if (!found) {
        throw std::runtime_error("No visible digit content found after emphasis.");
    }

    return box;
}

GrayImage tight_box_from_grayscale(const GrayImage& img, int eps) {
    return crop_gray(img, tight_box_from_grayscale_box(img, eps));
}

static double sinc(double x) {
    if (std::abs(x) < 1e-12) {
        return 1.0;
    }
    const double pix = M_PI * x;
    return std::sin(pix) / pix;
}

static double lanczos_kernel(double x, int a = 3) {
    x = std::abs(x);
    if (x >= a) {
        return 0.0;
    }
    return sinc(x) * sinc(x / a);
}

// Closer to PIL Lanczos behavior than nearest-neighbor/bilinear.
GrayImage resize_lanczos(const GrayImage& src, int new_h, int new_w, int a = 3) {
    if (new_h <= 0 || new_w <= 0) {
        throw std::runtime_error("resize_lanczos: invalid target size");
    }

    GrayImage dst(new_h, new_w, 0);

    const double scale_x = static_cast<double>(src.width) / new_w;
    const double scale_y = static_cast<double>(src.height) / new_h;

    for (int dy = 0; dy < new_h; ++dy) {
        for (int dx = 0; dx < new_w; ++dx) {
            const double src_x = (dx + 0.5) * scale_x - 0.5;
            const double src_y = (dy + 0.5) * scale_y - 0.5;

            const int x0 = static_cast<int>(std::floor(src_x));
            const int y0 = static_cast<int>(std::floor(src_y));

            double weighted_sum = 0.0;
            double weight_sum = 0.0;

            for (int ky = y0 - a + 1; ky <= y0 + a; ++ky) {
                if (ky < 0 || ky >= src.height) {
                    continue;
                }
                const double wy = lanczos_kernel(src_y - ky, a);
                if (wy == 0.0) {
                    continue;
                }

                for (int kx = x0 - a + 1; kx <= x0 + a; ++kx) {
                    if (kx < 0 || kx >= src.width) {
                        continue;
                    }
                    const double wx = lanczos_kernel(src_x - kx, a);
                    const double w = wx * wy;
                    if (w == 0.0) {
                        continue;
                    }

                    weighted_sum += w * src.at(ky, kx);
                    weight_sum += w;
                }
            }

            double value = 0.0;
            if (weight_sum != 0.0) {
                value = weighted_sum / weight_sum;
            }
            value = clamp_val(value, 0.0, 255.0);
            dst.at(dy, dx) = static_cast<uint8_t>(std::round(value));
        }
    }

    return dst;
}

GrayImage resize_to_mnist_box(const GrayImage& tight, int target_box_size, int& new_w, int& new_h) {
    const int w = tight.width;
    const int h = tight.height;

    if (h > w) {
        new_h = target_box_size;
        new_w = std::max(1, static_cast<int>(std::round(static_cast<double>(w) * target_box_size / h)));
    } else {
        new_w = target_box_size;
        new_h = std::max(1, static_cast<int>(std::round(static_cast<double>(h) * target_box_size / w)));
    }

    return resize_lanczos(tight, new_h, new_w, 3);
}

GrayImage paste_to_canvas(const GrayImage& resized, int new_w, int new_h, int canvas_size) {
    GrayImage canvas(canvas_size, canvas_size, 0);
    const int start_x = (canvas_size - new_w) / 2;
    const int start_y = (canvas_size - new_h) / 2;

    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            canvas.at(start_y + y, start_x + x) = resized.at(y, x);
        }
    }

    return canvas;
}

GrayImage center_of_mass_shift(const GrayImage& canvas, double target_x, double target_y) {
    double total_weight = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;

    for (int y = 0; y < canvas.height; ++y) {
        for (int x = 0; x < canvas.width; ++x) {
            const double w = static_cast<double>(canvas.at(y, x));
            if (w > 0.0) {
                total_weight += w;
                sum_x += x * w;
                sum_y += y * w;
            }
        }
    }

    if (total_weight == 0.0) {
        throw std::runtime_error("Canvas is empty. Cannot compute center of mass.");
    }

    const double com_x = sum_x / total_weight;
    const double com_y = sum_y / total_weight;

    const int shift_x = static_cast<int>(std::round(target_x - com_x));
    const int shift_y = static_cast<int>(std::round(target_y - com_y));

    GrayImage shifted(canvas.height, canvas.width, 0);

    const int src_y1 = std::max(0, -shift_y);
    const int src_y2 = std::min(canvas.height, canvas.height - shift_y);
    const int src_x1 = std::max(0, -shift_x);
    const int src_x2 = std::min(canvas.width, canvas.width - shift_x);

    const int dst_y1 = std::max(0, shift_y);
    const int dst_y2 = std::min(canvas.height, canvas.height + shift_y);
    const int dst_x1 = std::max(0, shift_x);
    const int dst_x2 = std::min(canvas.width, canvas.width + shift_x);

    for (int y = 0; y < (src_y2 - src_y1); ++y) {
        for (int x = 0; x < (src_x2 - src_x1); ++x) {
            shifted.at(dst_y1 + y, dst_x1 + x) = canvas.at(src_y1 + y, src_x1 + x);
        }
    }

    return shifted;
}

GrayImage preprocess_roi_to_canvas_64(
    const GrayImage& input_gray,
    int pad,
    int target_box_size,
    int tight_eps,
    bool verbose
) {
    if (input_gray.empty()) {
        throw std::runtime_error("Input image is empty.");
    }

    const int threshold = otsu_threshold(input_gray);
    const ForegroundChoice fg = choose_foreground(input_gray, threshold);

    if (verbose) {
        std::cout << "Otsu threshold: " << threshold << "\n";
        std::cout << "Chosen mode: " << fg.mode << "\n";
    }

    BoundingBox box = bounding_box_from_binary(fg.bw);

    if (verbose) {
        std::cout << "Bounding box before padding: "
                  << box.min_x << " " << box.max_x << " "
                  << box.min_y << " " << box.max_y << "\n";
    }

    box = add_padding_to_box(box, input_gray.height, input_gray.width, pad);

    if (verbose) {
        std::cout << "Bounding box after padding: "
                  << box.min_x << " " << box.max_x << " "
                  << box.min_y << " " << box.max_y << "\n";
    }

    GrayImage cropped = crop_gray(input_gray, box);
    GrayImage digit_img = emphasize_digit(cropped);
    GrayImage tight = tight_box_from_grayscale(digit_img, tight_eps);

    int new_w = 0;
    int new_h = 0;
    GrayImage resized = resize_to_mnist_box(tight, target_box_size, new_w, new_h);

    if (verbose) {
        std::cout << "Resized digit: " << new_w << "x" << new_h << "\n";
    }

    GrayImage canvas = paste_to_canvas(resized, new_w, new_h, 64);
    canvas = center_of_mass_shift(canvas, 31.5, 31.5);

    if (verbose) {
        uint8_t min_v = 255;
        uint8_t max_v = 0;
        for (uint8_t v : canvas.data) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
        std::cout << "Final canvas stats: "
                  << canvas.height << "x" << canvas.width
                  << " min=" << static_cast<int>(min_v)
                  << " max=" << static_cast<int>(max_v) << "\n";
    }

    return canvas;
}

// int main(int argc, char** argv) {
//     try {
//         if (argc < 2) {
//             std::cerr << "Usage: " << argv[0] << " input.bmp [output.pgm]\n";
//             return 1;
//         }

//         const std::string input_path = argv[1];
//         const std::string output_path = (argc >= 3) ? argv[2] : "final_canvas.pgm";

//         GrayImage gray = load_bmp_as_grayscale(input_path);
//         GrayImage canvas = preprocess_roi_to_canvas_64(gray, 5, 48, 5, true);

//         save_pgm(canvas, output_path);
//         std::cout << "Saved final canvas to " << output_path << "\n";

//         return 0;
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << "\n";
//         return 1;
//     }
// }