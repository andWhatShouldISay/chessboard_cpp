#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include "fastcluster.h"

using namespace std;

typedef cv::Mat Mat;

const int imglen = 512;
const int imglen0 = 128;
const int coef = imglen / imglen0;

auto sobel(Mat &src, int threshold) {
    Mat dx, dy, grad, dir;
    cv::Sobel(src, dx, CV_16SC1, 1, 0, 3);
    cv::Sobel(src, dy, CV_16SC1, 0, 1, 3);

    cv::convertScaleAbs(dx, dx);
    cv::convertScaleAbs(dy, dy);

    dx.convertTo(dx, CV_32F);
    dy.convertTo(dy, CV_32F);

    cv::phase(dy, dx, dir);

    dir *= float(180 / acos(-1));

    cv::pow(dx, 2, dx);
    cv::pow(dy, 2, dy);

    grad = dx + dy;


    cv::pow(grad, 0.5, grad);
    cv::convertScaleAbs(grad, grad);

    grad = grad >= threshold;


    return make_pair(grad, dir);
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

vector<vector<pair<int, int> > > calc_clusters(const Mat &m, int imglen0, bool show_pics = false) {
    vector<pair<int, int> > points;
    for (int i = 0; i < imglen0; i++) {
        for (int j = 0; j < imglen0; j++) {
            if (m.at<unsigned char>(i, j)) {
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

        cv::imshow("points " + to_string(imglen0), clr_points);
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

double precision(double x) {
    return abs(x - round(x));
}

bool intersects(line &l, vector<line> &v) {
    return any_of(v.begin(), v.end(), [&](line &a) {
        cv::Point2f p;
        bool ok = intersection(l, a, p);
        return ok && 0 <= p.x && p.x <= imglen0 && 0 <= p.y && p.y <= imglen0;
    });
}

Mat get_intersections(Mat &borders, Mat &hor, Mat &vert, int imglen0, int le, bool check_only_8 = false) {
    Mat intersections = cv::Mat::zeros({imglen0, imglen0}, CV_8U);

    vector<int> add(imglen0, 1);
    if (check_only_8)
        for (int i = 13; i < imglen0; i += imglen0 / 8)
            add[i] = imglen0 / 8 - 13 * 2;

    for (int i = le; i + 2 + le < imglen0; i += add[i]) {
        for (int j = le; j + 2 + le < imglen0; j += add[j]) {
            int val1 = cv::sum(borders(cv::Rect(i, j, 2, 2)))[0];

            int val2 = cv::sum(hor(cv::Rect(i - le, j, le, 2)))[0];
            if (!val2)
                continue;

            int val3 = cv::sum(vert(cv::Rect(i, j - le, 2, le)))[0];
            if (!val3)
                continue;


            int val4 = cv::sum(hor(cv::Rect(i + 2, j, le, 2)))[0];
            if (!val4)
                continue;

            int val5 = cv::sum(vert(cv::Rect(i, j + 2, 2, le)))[0];

            int val = (le > 3 || val1) && val2 && val3 && val4 && val5;
            intersections(cv::Rect(i + 1, j + 1, 1, 1)) = (!!val) * 255;
        }
    }

    //cv::imshow("keks " + to_string(imglen0), intersections);

    return intersections;
}

void find_corners(const Mat &source, const Mat &borders, const Mat &fft_v, const Mat &fft_h, const Mat &points,
                  const vector<vector<pair<int, int> > > &clusters) {

    Mat borders_coloured;

    cv::cvtColor(borders, borders_coloured, cv::COLOR_GRAY2RGB);

    auto verify = [&clusters](line l) {
        int score = 0;
        int r = l.r, th = l.th;
        for (auto &cluster: clusters) {
            for (auto &pt: cluster) {
                int y = pt.first, x = pt.second;
                int th_norm = th >= 0 ? th : 180 + th;

                if (abs(-r + x * coss[th_norm] + y * sins[th_norm]) <= 10) {
                    score += 1;
                    break;
                }
            }
            if (score >= 2) {
                break;
            }
        }
        return score >= 2;
    };

    int v_beg, v_end, h_beg, h_end;

    auto draw_horizontal = [&](vector<line> &drawn, int hough_threshold, int &beg, int &end) {
        map<line, int> acc;
        for (int i = 0; i < imglen0; i++) {
            for (int j = 0; j < imglen0; j++) {
                if (fft_h.at<unsigned char>(i, j)) {
                    int y = i, x = j;
                    if (2 <= y && y <= imglen0 - 5 && 2 <= x && x <= imglen0 - 5) {
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
                if (abs(y - v) <= 5) {
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

            auto y_mid = (p.r - a * imglen0 / 2.0) / b;

            if (!is_drawn(y_mid) && !intersects(p, drawn)) {
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
            beg = 1;
        } else if (drawn.size() >= 6) {
            beg = 1;
        } else {
            beg = 0;
        }

        if (drawn.size() <= 5) {
            end = drawn.size() - 1;
        } else if (drawn.size() <= 7) {
            end = drawn.size() - 1;
        } else {
            end = drawn.size() - 2;
        }

    };


    auto draw_vertical = [&](vector<line> &drawn, int hough_threshold, int &beg, int &end) {
        map<line, int> acc;
        for (int i = 0; i < imglen0; i++) {
            for (int j = 0; j < imglen0; j++) {
                if (fft_v.at<unsigned char>(i, j)) {
                    int y = i, x = j;
                    if (2 <= y && y <= imglen - 5 && 2 <= x && x <= imglen - 5) {
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
                if (abs(x - v) <= 5) {
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

            auto x_mid = (p.r - b * (imglen0 / 2.0)) / a;

            if (!is_drawn(x_mid) && !intersects(p, drawn)) {
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
            end = drawn.size() - 2;
            beg = 1;
        } else {
            end = drawn.size() - 1;
            beg = 0;
        }
    };

    vector<line> lines_hor;
    draw_horizontal(lines_hor, imglen0 / 10, h_beg, h_end);

    vector<line> lines_vert;
    draw_vertical(lines_vert, imglen0 / 10, v_beg, v_end);


    auto draw_line = [&](line &p, int blue = 0) {
        int th_norm = p.th >= 0 ? p.th : 180 + p.th;
        auto a = coss[th_norm];
        auto b = sins[th_norm];
        double x0 = a * p.r;
        double y0 = b * p.r;
        int x1 = int(x0 + imglen * (-b));
        int y1 = int(y0 + imglen * (a));
        int x2 = int(x0 - imglen * (-b));
        int y2 = int(y0 - imglen * (a));

        if ((45 <= p.th && p.th <= 135) ^ (!!blue))
            cv::line(borders_coloured, {x1, y1}, {x2, y2}, {static_cast<double>(blue), 0, 255}, 2);
        else
            cv::line(borders_coloured, {x1, y1}, {x2, y2}, {static_cast<double>(blue), 255, 0}, 2);
    };

    for (auto &p: lines_hor) {
        cout << "horizontal line " << p.r << ' ' << p.th << endl;
        draw_line(p);
    }

    for (auto &p: lines_vert) {
        cout << "vertical line " << p.r << ' ' << p.th << endl;
        draw_line(p);
    }

    cout << v_beg << ' ' << v_end << " v" << endl;
    cout << h_beg << ' ' << h_end << " h" << endl;

    vector<cv::Point2f> pts(4);


    if (v_beg >= v_end) {
        //lines_vert.clear();
        //draw_vertical(lines_vert, imglen / 6, v_beg, v_end);

        //if (v_beg >= v_end) {
        cv::imshow("huh?", borders_coloured);
        throw invalid_argument("chessboard not found because of lack of vertical lines");
        //}
    }

    if (h_beg >= h_end) {
        //lines_hor.clear();
        //draw_horizontal(lines_hor, imglen / 6, h_beg, h_end);

        //if (v_beg >= v_end) {
        cv::imshow("huh?", borders_coloured);
        throw invalid_argument("chessboard not found because of lack of horizontal lines");
        //}
    }

    draw_line(lines_hor[h_beg], 255);
    draw_line(lines_hor[h_end], 255);

    draw_line(lines_vert[v_beg], 255);
    draw_line(lines_vert[v_end], 255);

    cv::imshow("hough", borders_coloured);

    intersection(lines_hor[h_beg], lines_vert[v_beg], pts[0]);
    intersection(lines_hor[h_end], lines_vert[v_beg], pts[1]);
    intersection(lines_hor[h_end], lines_vert[v_end], pts[2]);
    intersection(lines_hor[h_beg], lines_vert[v_end], pts[3]);


    cout << "intersection " << pts[0] << endl;
    cout << "intersection " << pts[1] << endl;
    cout << "intersection " << pts[2] << endl;
    cout << "intersection " << pts[3] << endl;

    vector<cv::Point2f> target = {{0.0,     0.0},
                                  {0.0,     imglen0},
                                  {imglen0, imglen0},
                                  {imglen0, 0.0}};

    auto h = cv::findHomography(pts, target);

    Mat warped_lines, warped_points;
    Mat warped_fft_h, warped_fft_v;

    cv::warpPerspective(borders, warped_lines, h, {imglen0, imglen0});

    cv::warpPerspective(fft_v, warped_fft_v, h, {imglen0, imglen0});
    cv::warpPerspective(fft_h, warped_fft_h, h, {imglen0, imglen0});
    cv::imshow("warped v", warped_fft_v);
    cv::imshow("warped h", warped_fft_h);


    warped_fft_h.convertTo(warped_fft_h, CV_32F);
    warped_fft_v.convertTo(warped_fft_v, CV_32F);

    vector<double> sum_x(imglen), sum_y(imglen);

    auto peaks = [&](vector<double> &a) {
        vector<int> ans;
        const int pad = imglen0 / 11;
        for (int i = 0; i < imglen0 - pad; i++) {
            if (a[i] > 1 && a[i] >= *max_element(a.begin() + i - min(i, pad), a.begin() + i + pad)) {
                ans.push_back(i);
                cout << i << ' ' << a[i] << endl;
            }
        }
        cout << endl;

        if (ans.empty())
            return -1;

        vector<int> probably_difs;
        for (int i = 0; i + 1 < ans.size(); i++) {
            probably_difs.push_back(ans[i + 1] - ans[i]);
            cout << ' ' << probably_difs.back() << ' ' << imglen0 * 1.0 / probably_difs.back() << endl;
        }
        cout << endl;

        for (int i = 0; i + 1 < probably_difs.size(); i++) {
            for (int j = 0; j < probably_difs.size(); j++) {
                if (i != j && i + 1 != j) {
                    if (abs(probably_difs[j] - probably_difs[i] - probably_difs[i + 1]) <= 5) {
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
            if (x > pad * 3 / 4)
                difs.push_back(x);


        sort(difs.rbegin(), difs.rend());

        if (difs.empty())
            return 2;

        int res = int(0.001 + round(imglen0 * 1.0 / difs[difs.size() / 3]));

        if (res >= 7)
            return 2 + (int) difs.size();

        return min(8, res);
    };


    for (int i = 0; i < imglen0; i++) {
        sum_y[i] = 256.0 * cv::sum(warped_fft_v(cv::Rect(i, 0, 1, imglen0)))[0];
    }

    for (int i = 0; i < imglen0; i++) {
        sum_x[i] = 256.0 * cv::sum(warped_fft_h(cv::Rect(0, i, imglen0, 1)))[0];
    }

    int x_cells = peaks(sum_y), y_cells = peaks(sum_x);

    if (x_cells < 0 || y_cells < 0) {
        throw invalid_argument("i am unsure about cells' sizes");
    }


    cv::cvtColor(warped_lines, warped_lines, cv::COLOR_GRAY2RGB);

    for (int i = 1; i < x_cells; i++) {
        int x = imglen0 * i / x_cells;
        cv::line(warped_lines, {x, 0}, {x, imglen0}, {0, 75, 150}, 2);
    }

    for (int i = 1; i < y_cells; i++) {
        int y = imglen0 * i / y_cells;
        cv::line(warped_lines, {0, y}, {imglen0, y}, {0, 75, 150}, 2);
    }

    cv::imshow("calculated cells", warped_lines);

    h = cv::Matx33d(coef, 0, 0, 0, coef, 0, 0, 0, 1) * h * cv::Matx33d(1.0 / coef, 0, 0, 0, 1.0 / coef, 0, 0, 0, 1);


    auto y8 = float(y_cells / 8.0 * imglen), x8 = float(x_cells / 8.0 * imglen);

    auto h_8_8 = cv::findHomography(target, (vector<cv::Point2f>) {{0.0, 0.0},
                                                                   {0.0, y8},
                                                                   {x8,  y8},
                                                                   {x8,  0.0}});

    h_8_8 = cv::Matx33d(1.0 / coef, 0, 0, 0, 1.0 / coef, 0, 0, 0, 1) * h_8_8 * h;

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

    auto activity = [&](const Mat &x, bool flag) {
        int height = x.size[0];
        int width = x.size[1];

        int len = 4;

        auto hor = Mat(cv::abs(x(cv::Rect(0, len, width, height - len)) - x(cv::Rect(0, 0, width, height - len))));

        int threshold = 32;
        int times = width / 2;
        int score = 0;


        for (int i = 0; i < height - len; i++) {
            int sum = cv::sum((hor > threshold)(cv::Rect(0, i, width, 1)))[0];

            score += sum >= times * 255.0;

            if (flag && sum >= times * 255.0) {
                hor(cv::Rect(0, i, width, 1)) = 250;
            }
        }


        if (flag) {
            cout << " score " << score << endl;
            cv::imshow("debug hor " + to_string(rand()), hor);
            cv::imshow("debug x " + to_string(rand()), x);
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

            //cv::imshow("candidate " + to_string(x) + " " + to_string(y), candidate);

            int flag = false;

            auto vertical_stripe1 = activity(candidate(cv::Rect(0, 0, int(step) + 2, imglen)), 0);
            auto vertical_stripe2 = activity(candidate(cv::Rect(imglen - int(step) + 2, 0, int(step) - 2, imglen)),
                                             flag);

            auto horizontal_stripe1 = activity(candidate(cv::Rect(0, 0, imglen, int(step) + 2)).t(), 0);
            auto horizontal_stripe2 = activity(
                    candidate(cv::Rect(0, imglen - int(step) + 2, imglen, int(step) - 2)).t(), 0);

            int score = min({vertical_stripe1, vertical_stripe2, horizontal_stripe1, horizontal_stripe2});

            cout << x << ' ' << y << ' ' << score << endl;

            if (score > best_score) {
                best_score = score;
                best_candidate = candidate;
                best_homo = homo;
            }


        }
    }

    cv::imshow("result", best_candidate);

    auto work_with_result_matrix = [&](Mat &best_homo, const string &wname) {

        auto inv = best_homo.inv();

        cv::Matx34f corners(0.0, 0.0, imglen, imglen,
                            0.0, imglen, imglen, 0.0,
                            1.0, 1.0, 1.0, 1.0);

        auto real_corners = Mat(inv * corners);

        cout << endl;
        cout << real_corners << endl;

        real_corners = real_corners.t();

        Mat coloured_source;

        cv::cvtColor(source, coloured_source, cv::COLOR_GRAY2RGB);

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

        cout << endl;
        for (auto &x: v_corners)cout << x << endl;

        cv::line(coloured_source, v_corners[0], v_corners[1], {0, 0, 255}, 2);
        cv::line(coloured_source, v_corners[1], v_corners[2], {0, 0, 255}, 2);
        cv::line(coloured_source, v_corners[2], v_corners[3], {0, 0, 255}, 2);
        cv::line(coloured_source, v_corners[3], v_corners[0], {0, 0, 255}, 2);

        cv::imshow(wname, coloured_source);
        return v_corners;
    };

    auto work_points = [&](const string &name) {
        best_homo.convertTo(best_homo, CV_32F);

        Mat warped_points;

        cv::warpPerspective(source, warped_points, best_homo,
                            {imglen, imglen});

        auto[borders, dir] = sobel(warped_points, 70);

        auto vert = Mat((85 <= dir) & (dir <= 95) & borders);
        auto hor = Mat(((170 <= dir) | (dir <= 10)) & borders);

        auto start = chrono::steady_clock::now();
        Mat intersections = get_intersections(borders, hor, vert, imglen, 10, true);
        auto end = chrono::steady_clock::now();
        cout << "time " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

        if (name == "homo fix") {
            vector<Mat> channels = {intersections, vert, hor};
            Mat coloured;
            merge(channels, coloured);

            cv::imshow("new merge", coloured);
        }

        auto new_clusters = calc_clusters(intersections, imglen, true);

        cv::cvtColor(warped_points, warped_points, cv::COLOR_GRAY2RGB);

        for (int i = imglen / 8; i < imglen; i += imglen / 8) {
            cv::line(warped_points, {i, 0}, {i, imglen}, {255, 0, 0});
            cv::line(warped_points, {0, i}, {imglen, i}, {255, 0, 0});
        }

        vector<cv::Point2f> target, src;

        map<int,int> miss_x = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};
        map<int,int> miss_y = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};

        for (auto &cluster: new_clusters) {
            float y_med = 0, x_med = 0;
            for (auto &pt: cluster) {
                float y = pt.first, x = pt.second;
                y_med += y;
                x_med += x;
            }
            y_med /= cluster.size();
            x_med /= cluster.size();

            double x_cross = x_med / (imglen / 8.0);
            double y_cross = y_med / (imglen / 8.0);

            double tolerance = 0.3;


            if (-tolerance <= x_cross && x_cross <= 8 + tolerance)
                if (-tolerance <= y_cross && y_cross <= 8 + tolerance) {
                    if (precision(x_cross) <= tolerance && precision((y_cross)) <= tolerance) {

                        cv::circle(warped_points, {static_cast<int>(x_med), static_cast<int>(y_med)}, 3,
                                   {0, 0, 255}, -1);
                        src.emplace_back(x_med, y_med);

                        miss_x[round(x_cross)] += 1;
                        miss_y[round(y_cross)] += 1;

                        target.emplace_back(static_cast<float>(imglen / 8.0 * round(x_cross)),
                                            static_cast<float>(imglen / 8.0 * round(y_cross)));
                    }
                }
        }

        int threshold = 1;

        for (int i = 1; i <= 7; i++) {
            if (miss_x[i] <= threshold) {
                for (auto &p: target) {
                    p.x -= step;
                }
            } else {
                break;
            }
        }

        for (int i = 7; i >= 1; i--) {
            if (miss_x[i] <= threshold) {
                for (auto &p: target) {
                    p.x += step;
                }
            } else {
                break;
            }
        }

        for (int i = 1; i <= 7; i++) {
            if (miss_y[i] <= threshold) {
                for (auto &p: target) {
                    p.y -= step;
                }
            } else {
                break;
            }
        }

        for (int i = 7; i >= 1; i--) {
            if (miss_y[i] <= threshold) {
                for (auto &p: target) {
                    p.y += step;
                }
            } else {
                break;
            }
        }

        cv::imshow("warped points " + name, warped_points);

        if (src.size() < 4)
            throw invalid_argument("i agree that what i found isn't accurate");

        Mat homo_fix = cv::findHomography(src, target, cv::RANSAC);
        homo_fix.convertTo(homo_fix, CV_32F);

        return homo_fix * best_homo;
    };

    work_with_result_matrix(best_homo, "found corners");

    best_homo = work_points("homo fix");
    best_homo = work_points("homo fix 2");

    auto v_corners = work_with_result_matrix(best_homo, "fixed corners");
    for (auto &p: v_corners)
        p /= imglen;
}

void work(const string &filename) {
    auto img_grayscale = cv::imread(filename, 0);
    cv::imshow("source", img_grayscale);

    Mat img_128;
    cv::resize(img_grayscale, img_128, {imglen0, imglen0});
    cv::imshow("resized", img_128);

    auto[borders, dir] = sobel(img_128, 200);

    auto vert = Mat((85 <= dir) & (dir <= 95) & borders);
    auto hor = Mat(((170 <= dir) | (dir <= 10)) & borders);

    Mat intersections = get_intersections(borders, hor, vert, imglen0, 2);

    vector<Mat> channels = {intersections, vert, hor};
    Mat coloured;
    merge(channels, coloured);

    cv::resize(coloured, coloured, {imglen, imglen});
    cv::imshow("merge", coloured);

    Mat img_512;
    cv::resize(img_grayscale, img_512, {imglen, imglen});

    find_corners(img_512, borders, Mat(vert), Mat(hor), intersections, calc_clusters(intersections, imglen0, true));
}

int main(int n, char **args) {

    for (int i = 0; i < 180; i++) {
        sins[i] = sin(i / 180.0 * acos(-1));
        coss[i] = cos(i / 180.0 * acos(-1));
    }

    cout << string(args[1]) << endl;

    try {
        work(args[1]);
    } catch (invalid_argument e) {
        cout << e.what() << endl;
    }

    cv::waitKey();

    return 0;
}