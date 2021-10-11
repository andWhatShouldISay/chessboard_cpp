#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include "fastcluster.h"

using namespace std;

typedef cv::Mat Mat;

const int imglen = 256;

auto sobel(Mat &src) {
    Mat dx, dy, grad;
    cv::Sobel(src, dx, CV_16SC1, 1, 0, 3);
    cv::Sobel(src, dy, CV_16SC1, 0, 1, 3);

    cv::convertScaleAbs(dx, dx);
    cv::convertScaleAbs(dy, dy);

    dx.convertTo(dx, CV_32F);
    dy.convertTo(dy, CV_32F);

    cv::pow(dx, 2, dx);
    cv::pow(dy, 2, dy);

    grad = dx + dy;

    cv::pow(grad, 0.5, grad);
    cv::convertScaleAbs(grad, grad);

    grad = grad > 150;

    return grad;
}

auto gen_cross(int size, int line_width, double angle, int imglen, char hv) {
    auto cross = Mat(cv::Size(imglen, imglen), CV_32F);
    cross = 0.0;
    cross(cv::Rect(0, 0, size, size)) = -0.1;

    if (hv == 'v') {
        int mid = size / 2;
        auto sin_ = sin(angle);
        for (int i = 0; i < mid; i++) {
            auto dy = mid - i;
            auto dx = int(sin_ * dy);
            auto sign = dx > 0 ? 1 : -1;

            cross.at<float>(i, mid + dx - sign) = 0;
            cross.at<float>(i, mid + dx) = 1.0;
            cross.at<float>(i, mid + dx + sign) = 1.0;
            cross.at<float>(i, mid + dx + 2 * sign) = 0;

            cross.at<float>(size - 1 - i, size - 1 - mid - dx + sign) = 0;
            cross.at<float>(size - 1 - i, size - 1 - mid - dx) = 1.0;
            cross.at<float>(size - 1 - i, size - 1 - mid - dx - sign) = 1.0;
            cross.at<float>(size - 1 - i, size - 1 - mid - dx - 2 * sign) = 0;
        }
        return cross;
    }
    if (hv == 'h') {
        auto subcross = gen_cross(size, line_width, angle, imglen, 'v')(cv::Rect(0, 0, size, size)).t();
        cross(cv::Rect(0, 0, size, size)) = subcross;
        return cross;
    }
}


class cross {
public:
    Mat mat;
    Mat dft2;
    double norm;

    cross(const Mat &c) : mat(c) {
        cv::dft(mat, dft2, cv::DFT_COMPLEX_OUTPUT);

        norm = sqrt(cv::sum(c.mul(c))[0]);

    }

};

Mat dftE;

vector<cross> crosses;

const int cross_line_width = 2;
const int cross_side_length = 15;
const int cross_size = cross_side_length * 2 + cross_line_width;

void precalc_cross() {

    /*cout << "an example of a vertical line mask:" << endl;
    auto random_cross = gen_cross(cross_size, cross_line_width, acos(-1) * (85 + rand() % 10) / 90, imglen, 'v');
    cout << random_cross(Rect(0, 0, cross_size, cross_size)) << endl;*/

    for (int i = 85; i < 95; i++) {
        auto c = gen_cross(cross_size, cross_line_width, acos(-1) * i / 90, imglen, 'v');
        crosses.emplace_back(c);
    }

    for (int i = 85; i < 95; i++) {
        auto c = gen_cross(cross_size, cross_line_width, acos(-1) * i / 90, imglen, 'h');
        crosses.emplace_back(c);
    }

    Mat E(imglen, imglen, CV_32F);
    E = 0.0;
    E(cv::Rect(0, 0, cross_size, cross_size)) = 1.0;

    dft(E, dftE, cv::DFT_COMPLEX_OUTPUT);

}

Mat fft_ind(Mat &x, vector<cross>::iterator first, vector<cross>::iterator last) {
    Mat dft1, dft1sq;
    dft(x, dft1, cv::DFT_COMPLEX_OUTPUT);
    dft(x.mul(x), dft1sq,
        cv::DFT_COMPLEX_OUTPUT); // dft1sq равен dft1 потому что вход состоит из 0 и 1; оставил для приличия

    Mat xx;
    cv::mulSpectrums(dft1sq, dftE, xx, 0, true);

    cv::idft(xx, xx, cv::DFT_COMPLEX_INPUT | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::pow(xx, 0.5, xx);

    Mat score(x.size(), CV_32F);
    score = 0.0;

    for (auto it = first; it != last; ++it) {
        Mat xy;

        cv::mulSpectrums(dft1, it->dft2, xy, 0, true);
        cv::idft(xy, xy, cv::DFT_COMPLEX_INPUT | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        auto res = Mat(xy / xx / it->norm);

        cv::patchNaNs(res);

        score = max(score, res);

    }

    return score;

}

vector<vector<pair<int, int> > > calc_clusters(const Mat &m, bool show_pics = false) {
    vector<pair<int, int> > points;
    for (int i = 0; i < imglen; i++) {
        for (int j = 0; j < imglen; j++) {
            if (m.at<float>(i, j)) {
                points.emplace_back(i, j);
            }
        }
    }

    int n = points.size();

    auto *distmat = new double[(n * (n - 1)) / 2];
    int k, i, j;
    for (i = k = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            // compute distance between observables i and j
            distmat[k] = hypot(points[i].first - points[j].first, points[i].second - points[j].second);
            k++;
        }
    }

    int *merge = new int[2 * (n - 1)];
    double *height = new double[n - 1];
    hclust_fast(n, distmat, HCLUST_METHOD_SINGLE, merge, height);

    int *labels = new int[n];
    // stop clustering at step with custer distance >= cdist
    cutree_cdist(n, merge, height, 3, labels);

    int sz = *max_element(labels, labels + n) + 1;

    vector<vector<pair<int, int> > > ans(sz);

    if (show_pics) {
        Mat clr_points;

        cv::cvtColor(m, clr_points, cv::COLOR_GRAY2BGR);
        clr_points.convertTo(clr_points, CV_32FC3);

        vector<cv::Vec3f> random_clr(sz);

        for (int i = 0; i < sz; i++) {
            random_clr[i] = {static_cast<float>(rand() % 256), static_cast<float>(rand() % 256),
                             static_cast<float>(rand() % 256)};
            random_clr[i] /= 255.0;
        }
        for (int i = 0; i < n; i++) {
            clr_points.at<cv::Vec3f>(points[i].first, points[i].second) = random_clr[labels[i]];
        }

        cv::imshow("points", clr_points);
    }


    for (int i = 0; i < n; i++) {
        ans[labels[i]].push_back(points[i]);
    }


    return ans;
}

struct line {
    int r;
    int th;

    bool operator<(const line &rhs) const {
        if (r < rhs.r)
            return true;
        if (rhs.r < r)
            return false;
        return th < rhs.th;
    }

    bool operator>(const line &rhs) const {
        return rhs < *this;
    }

    bool operator<=(const line &rhs) const {
        return !(rhs < *this);
    }

    bool operator>=(const line &rhs) const {
        return !(*this < rhs);
    }

};

vector<double> sins(180), coss(180);

float get_y(int r, int th, double x) {
    int th_norm = th >= 0 ? th : 180 + th;
    return (r - x * coss[th_norm]) / sins[th_norm];
}

float get_x(int r, int th, double y) {
    int th_norm = th >= 0 ? th : 180 + th;
    return (r - y * sins[th_norm]) / coss[th_norm];
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
// stack overflow  copypaste
bool intersection(line &lh, line &lv,
                  cv::Point2f &r) {
    cv::Point2f o1, p1, o2, p2;

    o1 = {0.0, get_y(lh.r, lh.th, 0.0)};
    p1 = {256.0, get_y(lh.r, lh.th, 256.0)};

    o2 = {get_x(lv.r, lv.th, 0.0), 0.0};
    p2 = {get_x(lv.r, lv.th, 256.0), 256.0};

    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
    r = o1 + d1 * t1;
    return true;
}


vector<cv::Point2f> find_corners(Mat &source, Mat &borders, Mat &fft_v, Mat &fft_h, Mat &points,
                  vector<vector<pair<int, int> > > &clusters) {

    Mat borders_coloured;

    cv::cvtColor(borders, borders_coloured, cv::COLOR_GRAY2RGB);

    for (int i = 0; i < 180; i++) {
        sins[i] = sin(i / 180.0 * acos(-1));
        coss[i] = cos(i / 180.0 * acos(-1));
    }

    auto verify = [&clusters](line l) {
        int score = 0;
        int r = l.r, th = l.th;
        for (auto &cluster: clusters) {
            for (auto &pt: cluster) {
                int y = pt.first + cross_size / 2, x = pt.second + cross_size / 2;
                int th_norm = th >= 0 ? th : 180 + th;

                if (abs(-r + x * coss[th_norm] + y * sins[th_norm]) <= 5) {
                    score += 1;
                    break;
                }
            }
            if (score >= 3) {
                break;
            }
        }
        return score >= 3;
    };

    int v_beg, v_end, h_beg, h_end;

    auto draw_horizontal = [&](vector<line> &drawn, int hough_threshold, int &beg, int &end) {
        map<line, int> acc;
        for (int i = 0; i < imglen; i++) {
            for (int j = 0; j < imglen; j++) {
                if (fft_h.at<unsigned char>(i, j)) {
                    int y = i + cross_size / 2, x = j + cross_size / 2;
                    if (2 <= y && y <= 251 && 2 <= x && x <= 251) {
                        for (int th = 85; th < 96; th++) {
                            int r = x * coss[th] + y * sins[th];
                            acc[{r, th}] += 1;
                        }
                    }
                }
            }
        }

        vector<double> drawn_mid;

        auto is_drawn = [&drawn_mid](double y) {
            for (auto v: drawn_mid) {
                if (abs(y - v) <= 10) {
                    return true;
                }
            }
            return false;
        };

        vector<line> best;

        for (auto &p: acc) {
            if (p.second >= hough_threshold && verify(p.first)) {
                best.push_back(p.first);
            }
        }

        sort(best.begin(), best.end(), [&](line &a, line &b) { return acc[a] > acc[b]; });


        for (auto &p: best) {
            auto a = coss[p.th];
            auto b = sins[p.th];
            double x0 = a * p.r;
            double y0 = b * p.r;

            auto y_mid = (p.r - a * 128.0) / b;

            if (!is_drawn(y_mid)) {
                drawn_mid.push_back(y_mid);
                drawn.push_back(p);
            }

        }

        sort(drawn.begin(), drawn.end(), [&](line &a, line &b) {
            return abs(a.r) < abs(b.r);
        });

        while (drawn.size() >= 3 && abs(drawn[0].th - drawn[1].th) >= 3) {
            drawn.erase(drawn.begin());
        }

        while (drawn.size() >= 3 && abs(drawn.back().th - drawn[drawn.size() - 2].th) >= 3) {
            drawn.pop_back();
        }

        if (drawn.size() >= 8) {
            beg = 2;
        } else if (drawn.size() >= 6) {
            beg = 1;
        } else {
            beg = 0;
        }

        if (drawn.size() <= 5) {
            end = drawn.size() - 1;
        } else if (drawn.size() <= 7) {
            end = drawn.size() - 2;
        } else {
            end = drawn.size() - 3;
        }

    };


    auto draw_vertical = [&](vector<line> &drawn, int hough_threshold, int &beg, int &end) {
        map<line, int> acc;
        for (int i = 0; i < imglen; i++) {
            for (int j = 0; j < imglen; j++) {
                if (fft_v.at<unsigned char>(i, j)) {
                    int y = i + cross_size / 2, x = j + cross_size / 2;
                    if (2 <= y && y <= 251 && 2 <= x && x <= 251) {
                        for (int th = -10; th < 10; th++) {
                            int th_norm = th >= 0 ? th : 180 + th;
                            int r = x * coss[th_norm] + y * sins[th_norm];
                            acc[{r, th}] += 1;
                        }
                    }
                }
            }
        }

        vector<double> drawn_mid;

        auto is_drawn = [&drawn_mid](double x) {
            for (auto v: drawn_mid) {
                if (abs(x - v) <= 10) {
                    return true;
                }
            }
            return false;
        };

        vector<line> best;

        for (auto &p: acc) {
            if (p.second >= hough_threshold && verify(p.first)) {
                best.push_back(p.first);
            }
        }

        sort(best.begin(), best.end(), [&](line &a, line &b) { return acc[a] > acc[b]; });


        for (auto &p: best) {
            int th_norm = p.th >= 0 ? p.th : 180 + p.th;
            auto a = coss[th_norm];
            auto b = sins[th_norm];
            double x0 = a * p.r;
            double y0 = b * p.r;

            auto x_mid = (p.r - b * 128.0) / a;

            if (!is_drawn(x_mid)) {
                drawn_mid.push_back(x_mid);
                drawn.push_back(p);
            }

        }

        sort(drawn.begin(), drawn.end(), [&](line &a, line &b) {
            return abs(a.r) < abs(b.r);
        });

        while (drawn.size() >= 3 && !(drawn[0].th >= drawn[1].th && drawn[1].th >= drawn[2].th)) {
            drawn.erase(drawn.begin());
        }

        while (drawn.size() >= 3 && !(drawn[drawn.size() - 2].th >= drawn[drawn.size() - 1].th &&
                                      drawn[drawn.size() - 1].th >= drawn[drawn.size() - 0].th)) {
            drawn.pop_back();
        }


        if (drawn.size()) {
            vector<int> dp(drawn.size(), 1);
            vector<int> pt(drawn.size(), -1);
            int mx = 0;

            for (int i = 1; i < drawn.size(); i++) {
                for (int j = 0; j < i; j++) {
                    if (drawn[j].th >= drawn[i].th) {
                        if (dp[j] + 1 >= dp[i]) {
                            dp[i] = dp[j] + 1;
                            pt[i] = j;
                        }
                    }
                }
                if (dp[i] >= dp[mx])
                    mx = i;
            }


            vector<line> lis;
            int ind = mx;

            while (ind != -1) {
                lis.push_back(drawn[ind]);
                ind = pt[ind];
            }

            reverse(lis.begin(), lis.end());
            drawn = lis;
        }

        if (drawn.size() >= 8) {
            end = drawn.size() - 3;
            beg = 2;
        } else {
            end = drawn.size() - 1;
            beg = 1;
        }
    };

    vector<line> lines_hor;
    draw_horizontal(lines_hor, imglen / 4, h_beg, h_end);

    vector<line> lines_vert;
    draw_vertical(lines_vert, imglen / 4, v_beg, v_end);


    auto draw_line = [&](line &p, int blue = 0) {
        int th_norm = p.th >= 0 ? p.th : 180 + p.th;
        auto a = coss[th_norm];
        auto b = sins[th_norm];
        double x0 = a * p.r;
        double y0 = b * p.r;
        int x1 = int(x0 + 300 * (-b));
        int y1 = int(y0 + 300 * (a));
        int x2 = int(x0 - 300 * (-b));
        int y2 = int(y0 - 300 * (a));

        if ((45 <= p.th && p.th <= 135) ^ (!!blue))
            cv::line(borders_coloured, {x1, y1}, {x2, y2}, {static_cast<double>(blue), 0, 255}, 2);
        else
            cv::line(borders_coloured, {x1, y1}, {x2, y2}, {static_cast<double>(blue), 255, 0}, 2);
    };

    /*for (auto &p: lines_hor) {
        cout << "horizontal line " << p.r << ' ' << p.th << endl;
        draw_line(p);
    }

    for (auto &p: lines_vert) {
        cout << "vertical line " << p.r << ' ' << p.th << endl;
        draw_line(p);
    }*/

    //cout << v_beg << ' ' << v_end << " v" << endl;
    //cout << h_beg << ' ' << h_end << " h" << endl;

    vector<cv::Point2f> pts(4);



    if (v_beg >= v_end) {
        cv::imshow("huh?", borders_coloured);
        throw invalid_argument("chessboard not found because of lack of vertical lines");
    }

    if (h_beg >= h_end) {
        cv::imshow("huh?", borders_coloured);
        throw invalid_argument("chessboard not found because of lack of horizontal lines");
    }

    /*draw_line(lines_hor[h_beg], 255);
    draw_line(lines_hor[h_end], 255);

    draw_line(lines_vert[v_beg], 255);
    draw_line(lines_vert[v_end], 255);

    cv::imshow("hough", borders_coloured);*/

    intersection(lines_hor[h_beg], lines_vert[v_beg], pts[0]);
    intersection(lines_hor[h_end], lines_vert[v_beg], pts[1]);
    intersection(lines_hor[h_end], lines_vert[v_end], pts[2]);
    intersection(lines_hor[h_beg], lines_vert[v_end], pts[3]);


    /*cout << "intersection " << pts[0] << endl;
    cout << "intersection " << pts[1] << endl;
    cout << "intersection " << pts[2] << endl;
    cout << "intersection " << pts[3] << endl;*/

    auto pts_shift = pts;
    for (auto &p:pts_shift) {
        p.x -= cross_size / 2.0;
        p.y -= cross_size / 2.0;
    }

    vector<cv::Point2f> target = {{0.0,    0.0},
                                  {0.0,    imglen},
                                  {imglen, imglen},
                                  {imglen, 0.0}};

    auto h = cv::findHomography(pts, target);
    auto h2_v = cv::findHomography(pts_shift, target);
    auto h2_h = cv::findHomography(pts_shift, target);

    Mat warped_lines, warped_points;
    Mat warped_fft_h, warped_fft_v;

    cv::warpPerspective(borders, warped_lines, h, {imglen, imglen});
    //cv::imshow("warped lines", warped_lines);


    cv::warpPerspective(fft_v, warped_fft_v, h2_v, {imglen, imglen});
    cv::warpPerspective(fft_h, warped_fft_h, h2_h, {imglen, imglen});
    //cv::imshow("warped v", warped_fft_v);
    //cv::imshow("warped h", warped_fft_h);

    warped_fft_h.convertTo(warped_fft_h, CV_32F);
    warped_fft_v.convertTo(warped_fft_v, CV_32F);

    vector<double> sum_x(imglen), sum_y(imglen);


    auto peaks = [&](vector<double> &a) {
        vector<int> ans;
        double mx = *max_element(a.begin(), a.end());
        int coef = 3;
        for (int i = 0; i < 234; i++) {
            if (a[i] >= *max_element(a.begin() + i - 20, a.begin() + i + 20) && a[i] > 1) {
                ans.push_back(i);
            }
        }

        if (ans.empty())
            return -1;

        vector<int> probably_difs;
        for (int i = 0; i + 1 < ans.size(); i++) {
            probably_difs.push_back(ans[i + 1] - ans[i]);
        }

        for (int i = 0; i + 1 < probably_difs.size(); i++) {
            for (int j = 0; j < probably_difs.size(); j++) {
                if (i != j && i + 1 != j) {
                    if (abs(probably_difs[j] - probably_difs[i] - probably_difs[i + 1]) <= 10) {
                        probably_difs.push_back(probably_difs[i] + probably_difs[i + 1]);
                        probably_difs.erase(probably_difs.begin() + i, probably_difs.begin() + i + 2);
                        --i;
                        break;
                    }
                }
            }
        }

        vector<int> difs;
        for (auto &x: probably_difs)
            if (x > 15)
                difs.push_back(x);

        sort(difs.rbegin(), difs.rend());


        if (difs.empty())
            return 2;
        return min(8, int(0.001 + round(imglen * 1.0 / difs[difs.size() / 2])));
    };

    int x_cells = -2, y_cells = -2;

    map<int, int> kol_x;
    for (int x = 0; x < imglen; x++) {
        for (int i = 0; i < imglen; i++) {
            sum_y[i] = 256.0 * warped_fft_h.at<float>(i, x);
        }
        ++kol_x[peaks(sum_y)];
    }

    for (auto &p: kol_x) {
        if (p.first > 1 && p.second > kol_x[y_cells])
            y_cells = p.first;
    }

    map<int, int> kol_y;
    for (int y = 0; y < imglen; y++) {
        for (int i = 0; i < imglen; i++) {
            sum_x[i] = 256.0 * warped_fft_v.at<float>(y, i);
        }
        ++kol_y[peaks(sum_x)];
    }

    for (auto &p: kol_y) {
        if (p.first > 1 && p.second > kol_y[x_cells])
            x_cells = p.first;
    }

    if (x_cells < 0 || y_cells < 0){
        throw invalid_argument("i am unsure about cells' sizes");
    }

    /*cv::cvtColor(warped_lines, warped_lines, cv::COLOR_GRAY2RGB);

    for (int i = 1; i < x_cells; i++) {
        int x = imglen * i / x_cells;
        cv::line(warped_lines, {x, 0}, {x, 256}, {0, 0.3, 0.6}, 2);
    }

    for (int i = 1; i < y_cells; i++) {
        int y = imglen * i / y_cells;
        cv::line(warped_lines, {0, y}, {256, y}, {0, 0.3, 0.6}, 2);
    }

    cv::imshow("calculated cells", warped_lines);*/


    auto y8 = float(y_cells / 8.0 * imglen), x8 = float(x_cells / 8.0 * imglen);

    auto h_8_8 = cv::findHomography(target, (vector<cv::Point2f>) {{0.0, 0.0},
                                                                   {0.0, y8},
                                                                   {x8,  y8},
                                                                   {x8,  0.0}});

    h_8_8 = h_8_8 * h;

    float step = imglen / 8.0;
    auto imglen_f = float(imglen);

    auto h_step_x = cv::findHomography((vector<cv::Point2f>) {{0.0,      0.0},
                                                              {0.0,      imglen_f},
                                                              {imglen_f, imglen_f},
                                                              {imglen_f, 0.0}},
                                       (vector<cv::Point2f>) {{step,            0.0},
                                                              {step,            imglen_f},
                                                              {step + imglen_f, imglen_f},
                                                              {step + imglen_f, 0.0}}
    );

    auto h_step_y = cv::findHomography((vector<cv::Point2f>) {{0.0,      0.0},
                                                              {0.0,      imglen_f},
                                                              {imglen_f, imglen_f},
                                                              {imglen_f, 0.0}},
                                       (vector<cv::Point2f>) {{0.0,      step},
                                                              {0.0,      imglen_f + step},
                                                              {imglen_f, imglen_f + step},
                                                              {imglen_f, step}}
    );

    auto activity = [&](const Mat &x) {
        int height = x.size[0];
        int width = x.size[1];

        auto hor = cv::abs(x(cv::Rect(0, 2, width, height - 2)) - x(cv::Rect(0, 0, width, height - 2)));

        int score = 0, threshold = 32;

        for (int a = 1; a < 8; a++) {
            int d1 = -5, d2 = 6;
            int filter_index = int(step * a) + d1;

            score += cv::sum((hor > threshold)(cv::Rect(0, filter_index, width, d2 - d1)))[0];
        }

        return score;
    };

    vector<vector<Mat> > shifts(9, vector<Mat>(9));

    shifts[0][0] = cv::Mat::eye(3, 3, CV_32FC1);
    h_8_8.convertTo(h_8_8, CV_32FC1);
    h_step_x.convertTo(h_step_x, CV_32FC1);
    h_step_y.convertTo(h_step_y, CV_32FC1);

    int best_score = -1;
    Mat best_candidate, best_homo;

    for (int x = 0; x <= 8 - x_cells; x++) {
        for (int y = 0; y <= 8 - y_cells; y++) {
            if (x > 0) {
                shifts[x][y] = h_step_x * shifts[x - 1][y];
            } else if (y > 0) {
                shifts[x][y] = h_step_y * shifts[x][y - 1];
            }

            auto homo = shifts[x][y] * h_8_8;

            Mat candidate;
            cv::warpPerspective(source, candidate, homo, {imglen, imglen});

            //cv::imshow("candidate "+ to_string(x) + " " + to_string(y), candidate);

            auto vertical_stripe1 = activity(candidate(cv::Rect(0, 0, int(step) + 2, imglen)));
            auto vertical_stripe2 = activity(candidate(cv::Rect(imglen - int(step) + 2, 0, int(step) - 2, imglen)));

            auto horizontal_stripe1 = activity(candidate(cv::Rect(0, 0, imglen, int(step) + 2)).t());
            auto horizontal_stripe2 = activity(
                    candidate(cv::Rect(0, imglen - int(step) + 2, imglen, int(step) - 2)).t());

            int score = min({vertical_stripe1, vertical_stripe2, horizontal_stripe1, horizontal_stripe2});

            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
                best_homo = homo;
            }

        }
    }

    //cv::imshow("result", best_candidate);

    auto inv = best_homo.inv();

    cv::Matx34f corners(0.0, 0.0, imglen, imglen,
                        0.0, imglen, imglen, 0.0,
                        1.0, 1.0, 1.0, 1.0);

    auto real_corners = Mat(inv * corners);

    //cout << endl;
    //cout << real_corners << endl;

    real_corners = real_corners.t();

    /*Mat coloured_source;

    cv::cvtColor(source, coloured_source, cv::COLOR_GRAY2RGB);*/

    vector<cv::Point2f> v_corners = {{real_corners.at<float>(0, 0) / real_corners.at<float>(0, 2),
                                                                                                real_corners.at<float>(
                                                                                                        0, 1) /
                                                                                                real_corners.at<float>(
                                                                                                        0, 2)},
                                     {real_corners.at<float>(1, 0) / real_corners.at<float>(1,
                                                                                            2), real_corners.at<float>(
                                             1, 1) / real_corners.at<float>(1, 2)},
                                     {real_corners.at<float>(2, 0) / real_corners.at<float>(2,
                                                                                            2), real_corners.at<float>(
                                             2, 1) / real_corners.at<float>(2, 2)},
                                     {real_corners.at<float>(3, 0) / real_corners.at<float>(3,
                                                                                            2), real_corners.at<float>(
                                             3, 1) / real_corners.at<float>(3, 2)}
    };

    /*cout << endl;
    for (auto &x: v_corners)cout << x << endl;

    cv::line(coloured_source, v_corners[0], v_corners[1], {0, 0, 255}, 2);
    cv::line(coloured_source, v_corners[1], v_corners[2], {0, 0, 255}, 2);
    cv::line(coloured_source, v_corners[2], v_corners[3], {0, 0, 255}, 2);
    cv::line(coloured_source, v_corners[3], v_corners[0], {0, 0, 255}, 2);

    cv::imshow("found corners", coloured_source);*/

    for (auto &p:v_corners)
        p /= 256.0;

    return v_corners;
}

vector<cv::Point2f> find_corners(const string &filename) {
    auto img_grayscale = cv::imread(filename, 0);
    //cv::imshow("source", img_grayscale);
    
    if (img_grayscale.empty()){
        throw invalid_argument("error while reading file");
    }
    
    Mat img_256;
    cv::resize(img_grayscale, img_256, {256, 256});
    //cv::imshow("resized", img_256);

    auto borders = sobel(img_256);
    //cv::imshow("sobel", borders);
    borders.convertTo(borders, CV_32F);
    borders /= 255;

    auto fft_v = fft_ind(borders, crosses.begin() + 0, crosses.begin() + crosses.size() / 2);
    auto fft_h = fft_ind(borders, crosses.begin() + crosses.size() / 2, crosses.end());

    auto threshold1 = 0.1;

    auto points_ = ((fft_v > threshold1) & (fft_h > threshold1) & (fft_v <= 1.0) & (fft_h <= 1.0)); // /255.0;
    fft_v = ((fft_v > threshold1) & (fft_v <= 1.0));
    fft_h = ((fft_h > threshold1) & (fft_h <= 1.0));

    //cv::imshow("fft_v", fft_v);
    //cv::imshow("fft_h", fft_h);

    auto points = Mat(points_);

    points.convertTo(points, CV_32F);
    auto clusters = calc_clusters(points, false);

    fft_v.convertTo(fft_v, CV_8U);
    fft_h.convertTo(fft_h, CV_8U);

    return find_corners(img_256, borders, fft_v, fft_h, points, clusters);

}

int main(int n, char **args) {
    precalc_cross();

    cout << string(args[1]) << endl;

    try {
        cout << find_corners(args[1]) << endl;
    } catch (invalid_argument e) {
        cout << e.what() << endl;
    }

    return 0;
}
