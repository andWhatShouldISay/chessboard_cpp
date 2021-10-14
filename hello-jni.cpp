/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <opencv2/opencv.hpp>
#include <string.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <math.h>
#include <chrono>
#include "fastcluster.h"

#define log_print(...) __android_log_print(ANDROID_LOG_DEBUG, __VA_ARGS__)

using namespace std;

std::string test_speed(AAssetManager *mgr);

/* This is a trivial JNI example where we use a native method
 * to return a new VM String. See the corresponding Java source
 * file located at:
 *
 *   hello-jni/app/src/main/java/com/example/hellojni/HelloJni.java
 */
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_hellojni_HelloJni_stringFromJNI( JNIEnv* env, jobject thiz, jobject assetManager)
{
#if defined(__arm__)
    #if defined(__ARM_ARCH_7A__)
    #if defined(__ARM_NEON__)
      #if defined(__ARM_PCS_VFP)
        #define ABI "armeabi-v7a/NEON (hard-float)"
      #else
        #define ABI "armeabi-v7a/NEON"
      #endif
    #else
      #if defined(__ARM_PCS_VFP)
        #define ABI "armeabi-v7a (hard-float)"
      #else
        #define ABI "armeabi-v7a"
      #endif
    #endif
  #else
   #define ABI "armeabi"
  #endif
#elif defined(__i386__)
#define ABI "x86"
#elif defined(__x86_64__)
#define ABI "x86_64"
#elif defined(__mips64)  /* mips64el-* toolchain defines __mips__ too */
#define ABI "mips64"
#elif defined(__mips__)
#define ABI "mips"
#elif defined(__aarch64__)
#define ABI "arm64-v8a"
#else
#define ABI "unknown"
#endif
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    auto test_out = test_speed(mgr);
    std::stringstream ss;
    ss << test_out << "ABI: " << ABI;
    return env->NewStringUTF(ss.str().c_str());
}

auto start = std::chrono::steady_clock::now();

void telling_time(string s) {
    auto end = std::chrono::steady_clock::now();
    //cout << s << ' ' <<  << endl;
    log_print("Dummy", "%s %d", s.c_str(), (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    start = end;
}


typedef cv::Mat Mat;

const int imglen = 256;
const int imglen0 = 128;
const int coef = imglen / imglen0;
const double GLOBAL_TOLERANCE = 0.3;

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

vector<vector<pair<int, int> > > calc_clusters(const Mat &m, int imglen0, bool show_pics = false) {
    vector<cv::Point> points_;
    cv::findNonZero(m, points_);
    vector<pair<int, int> > points;
    for (auto &p: points_)
        points.emplace_back(p.y, p.x);

    //telling_time("      calc_clusters/iterated picture");

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

    //telling_time("      calc_clusters/clusterization itself");

    int sz = *max_element(labels, labels + n) + 1;

    vector<vector<pair<int, int> > > ans(sz);

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

Mat
get_intersections(Mat &borders, Mat &hor, Mat &vert, int imglen0, int le, vector<vector<pair<int, int> > > &clusters,
                  bool check_only_8 = false) {
    Mat hor_sum = hor(cv::Rect(0, 0, imglen0 - le + 1, imglen0 - 1)).clone();
    Mat vert_sum = vert(cv::Rect(0, 0, imglen0 - 1, imglen0 - le + 1)).clone();

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < le; j++) {
            if (i || j) {
                hor_sum = hor_sum | hor(cv::Rect(j, i, imglen0 - le + 1, imglen0 - 1));
                vert_sum = vert_sum | vert(cv::Rect(i, j, imglen0 - 1, imglen0 - le + 1));
            }
        }
    }

    Mat intersections = cv::Mat::zeros({imglen0 + le + le, imglen0 + le + le}, CV_8U);

    intersections(cv::Rect(le + 2, 2, imglen0 - le + 1, imglen0 - 1)) |= hor_sum;
    intersections(cv::Rect(0, 2, imglen0 - le + 1, imglen0 - 1)) &= hor_sum;

    intersections(cv::Rect(2, le + 2, imglen0 - 1, imglen0 - le + 1)) &= vert_sum;
    intersections(cv::Rect(2, 0, imglen0 - 1, imglen0 - le + 1)) &= vert_sum;

    if (le <= 3)
        intersections(cv::Rect(2, 2, imglen0, imglen0)) &= borders;

    intersections = intersections(cv::Rect(2, 2, imglen0, imglen0));

    intersections(cv::Rect(0, 0, le, imglen0)) *= 0;
    intersections(cv::Rect(0, 0, imglen0, le)) *= 0;

    intersections(cv::Rect(imglen0 - 2, 0, 2, imglen0)) *= 0;
    intersections(cv::Rect(0, imglen0 - 2, imglen0, 2)) *= 0;

    //cv::imshow("keks " + to_string(rand()), intersections);

    if (check_only_8) {
        vector<cv::Point> nonzero;
        clusters.resize(81);

        cv::findNonZero(intersections, nonzero);
        for (auto &p: nonzero) {
            double x = p.x / (imglen0 / 8.0);
            double y = p.y / (imglen0 / 8.0);
            if (precision(x) <= GLOBAL_TOLERANCE && precision(y) <= GLOBAL_TOLERANCE) {
                clusters[round(y) * 9 + round(x)].push_back({p.y, p.x});
            }
        }

    };
    return intersections;
}

vector<cv::Point2f>
find_corners(const Mat &source, const Mat &borders, const Mat &fft_v, const Mat &fft_h, const Mat &points,
             const vector<vector<pair<int, int> > > &clusters) {

    //telling_time("function call");
    //Mat borders_coloured;

    //cv::cvtColor(borders, borders_coloured, cv::COLOR_GRAY2RGB);

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
        vector<cv::Point> v;
        cv::findNonZero(fft_h, v);
        for (auto &p: v) {
            int y = p.y, x = p.x;
            if (2 <= y && y <= imglen0 - 5 && 2 <= x && x <= imglen0 - 5) {
                for (int th = 85; th < 96; th++) {
                    int r = x * coss[th] + y * sins[th];
                    acc[{r, th}] += 1;
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
        vector<cv::Point> v;
        cv::findNonZero(fft_v, v);
        for (auto &p: v) {
            int y = p.y, x = p.x;
            if (2 <= y && y <= imglen - 5 && 2 <= x && x <= imglen - 5) {
                for (int th = -10; th < 10; th++) {
                    int th_norm = th >= 0 ? th : 180 + th;
                    int r = x * coss[th_norm] + y * sins[th_norm];
                    acc[{r, th}] += 1;
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

    //telling_time("horizontal hough");

    vector<line> lines_vert;
    draw_vertical(lines_vert, imglen0 / 10, v_beg, v_end);

    //telling_time("vertical hough");


    /*auto draw_line = [&](line &p, int blue = 0) {
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
    cout << h_beg << ' ' << h_end << " h" << endl;*/

    vector<cv::Point2f> pts(4);


    if (v_beg >= v_end) {
        //lines_vert.clear();
        //draw_vertical(lines_vert, imglen / 6, v_beg, v_end);

        //if (v_beg >= v_end) {
        //cv::imshow("huh?", borders_coloured);
        throw invalid_argument("chessboard not found because of lack of vertical lines");
        //}
    }

    if (h_beg >= h_end) {
        //lines_hor.clear();
        //draw_horizontal(lines_hor, imglen / 6, h_beg, h_end);

        //if (v_beg >= v_end) {
        //cv::imshow("huh?", borders_coloured);
        throw invalid_argument("chessboard not found because of lack of horizontal lines");
        //}
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
    //cv::imshow("warped v", warped_fft_v);
    //cv::imshow("warped h", warped_fft_h);


    warped_fft_h.convertTo(warped_fft_h, CV_32F);
    warped_fft_v.convertTo(warped_fft_v, CV_32F);

    //telling_time("warp v h");

    vector<double> sum_x(imglen), sum_y(imglen);

    auto peaks = [&](vector<double> &a) {
        vector<int> ans;
        const int pad = imglen0 / 11;
        for (int i = 0; i < imglen0 - pad; i++) {
            if (a[i] > 1 && a[i] >= *max_element(a.begin() + i - min(i, pad), a.begin() + i + pad)) {
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

    //telling_time("got x_cells y_cells");

    if (x_cells < 0 || y_cells < 0) {
        throw invalid_argument("i am unsure about cells' sizes");
    }


    /*cv::cvtColor(warped_lines, warped_lines, cv::COLOR_GRAY2RGB);

    for (int i = 1; i < x_cells; i++) {
        int x = imglen0 * i / x_cells;
        cv::line(warped_lines, {x, 0}, {x, imglen0}, {0, 75, 150}, 2);
    }

    for (int i = 1; i < y_cells; i++) {
        int y = imglen0 * i / y_cells;
        cv::line(warped_lines, {0, y}, {imglen0, y}, {0, 75, 150}, 2);
    }

    cv::imshow("calculated cells", warped_lines);*/

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

    vector<int> add(imglen, 1);
    for (int i = imglen / 32 - 2; i < imglen; i += imglen / 8 - 1) {
        add[i] = imglen / 16 - 1;
    }

    auto activity = [&](const Mat &x, bool flag) {
        int height = x.size[0];
        int width = x.size[1];

        int len = 4;

        int threshold = 32;

        auto hor = Mat(
                cv::abs(x(cv::Rect(0, len, width, height - len)) - x(cv::Rect(0, 0, width, height - len))) > threshold);

        cv::reduce(hor, hor, 1, cv::REDUCE_SUM, CV_32F);

        int times = width / 2;

/*
 *
[0.273799, 0.0958937]
[0.22171, 0.905925]
[0.767077, 0.900173]
[0.711194, 0.0968123]
 */

        return cv::sum(hor > times * 255.0)[0];
    };

    vector<vector<Mat> > shifts(9, vector<Mat>(9));

    shifts[0][0] = cv::Mat::eye(3, 3, CV_32FC1);
    h_8_8.convertTo(h_8_8, CV_32FC1);
    h_step_x.convertTo(h_step_x, CV_32FC1);
    h_step_y.convertTo(h_step_y, CV_32FC1);

    int best_score = -1;
    Mat best_homo;

    Mat candidates;

    cv::warpPerspective(source, candidates, cv::Matx33f(1, 0, imglen, 0, 1, imglen, 0, 0, 1) * h_8_8,
                        {imglen * 2, imglen * 2});

    set<int> bad_x, bad_y;
    int istep = int(step);

    for (int x = 0; x <= 8 - x_cells; x += 1) {
        for (int y = 0; y <= 8 - y_cells; y += 1) {
            if (x > 0) {
                shifts[x][y] = h_step_x * shifts[x - 1][y];
            } else if (y > 0) {
                shifts[x][y] = h_step_y * shifts[x][y - 1];
            }

            if (bad_x.count(x) || bad_y.count(y)) {
                continue;
            }


            Mat v11, h11, v22, h22;

            v11 = candidates(cv::Rect(imglen - x * istep, imglen - y * istep, istep + 2, imglen));
            v22 = candidates(
                    cv::Rect(imglen - x * istep + (imglen - istep + 2), imglen - y * istep, istep - 2, imglen));

            h11 = candidates(cv::Rect(imglen - x * istep, imglen - y * istep, imglen, istep + 2));
            h22 = candidates(
                    cv::Rect(imglen - x * istep, imglen - y * istep + (imglen - istep + 2), imglen, istep - 2));

            int flag = false; // only for debug purposes

            auto vertical_stripe1 = activity(v11, 0);
            auto vertical_stripe2 = activity(v22, 0);

            auto horizontal_stripe1 = activity(h11.t(), 0);
            auto horizontal_stripe2 = activity(h22.t(), 0);

            int score = min({vertical_stripe1, vertical_stripe2, horizontal_stripe1, horizontal_stripe2});

            if (score > best_score) {
                auto homo = shifts[x][y] * h_8_8;
                best_score = score;
                best_homo = homo;
            }

            if (vertical_stripe1 <= 0 || vertical_stripe2 <= 0) {
                bad_x.insert(x);
            }

            if (horizontal_stripe1 <= 0 || horizontal_stripe2 <= 0) {
                bad_y.insert(y);
            }

        }
    }
    //telling_time("candidates selection");

    //cv::imshow("result", best_candidate);

    auto work_with_result_matrix = [&](Mat &best_homo, const string &wname) {

        auto inv = best_homo.inv();

        cv::Matx34f corners(0.0, 0.0, imglen, imglen,
                            0.0, imglen, imglen, 0.0,
                            1.0, 1.0, 1.0, 1.0);

        auto real_corners = Mat(inv * corners);

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

//        cv::line(coloured_source, v_corners[0], v_corners[1], {0, 0, 255}, 2);
//        cv::line(coloured_source, v_corners[1], v_corners[2], {0, 0, 255}, 2);
//        cv::line(coloured_source, v_corners[2], v_corners[3], {0, 0, 255}, 2);
//        cv::line(coloured_source, v_corners[3], v_corners[0], {0, 0, 255}, 2);
//
//        cv::imshow(wname, coloured_source);
        return v_corners;
    };

    auto work_points = [&](const string &name) {
        best_homo.convertTo(best_homo, CV_32F);

        Mat warped_points;

        cv::warpPerspective(source, warped_points, best_homo,
                            {imglen, imglen});

        auto[borders, dir] = sobel(warped_points, 70);

        /*cv::cvtColor(warped_points, warped_points, cv::COLOR_GRAY2RGB);

        for (int i = imglen / 8; i < imglen; i += imglen / 8) {
            cv::line(warped_points, {i, 0}, {i, imglen}, {255, 0, 0});
            cv::line(warped_points, {0, i}, {imglen, i}, {255, 0, 0});
        }*/

        auto vert = Mat((85 <= dir) & (dir <= 95) & borders);
        auto hor = Mat(((170 <= dir) | (dir <= 10)) & borders);

        vector<vector<pair<int, int> > > new_clusters;
        get_intersections(borders, hor, vert, imglen, 5, new_clusters, true);
        //telling_time("  sobel/v/h/intersections themselves");

        vector<cv::Point2f> target, src;

        map<int, int> miss_x = {{1, 0},
                                {2, 0},
                                {3, 0},
                                {4, 0},
                                {5, 0},
                                {6, 0},
                                {7, 0}};
        map<int, int> miss_y = {{1, 0},
                                {2, 0},
                                {3, 0},
                                {4, 0},
                                {5, 0},
                                {6, 0},
                                {7, 0}};

        for (auto &cluster: new_clusters) {
            if (cluster.empty())
                continue;
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


            if (-GLOBAL_TOLERANCE <= x_cross && x_cross <= 8 + GLOBAL_TOLERANCE)
                if (-GLOBAL_TOLERANCE <= y_cross && y_cross <= 8 + GLOBAL_TOLERANCE) {
                    if (precision(x_cross) <= GLOBAL_TOLERANCE && precision((y_cross)) <= GLOBAL_TOLERANCE) {
                        src.emplace_back(x_med, y_med);

                        //cv::circle(warped_points, {static_cast<int>(x_med), static_cast<int>(y_med)}, 3,
                        //           {0, 0, 255}, -1);

                        miss_x[round(x_cross)] += 1;
                        miss_y[round(y_cross)] += 1;

                        target.emplace_back(static_cast<float>(imglen / 8.0 * round(x_cross)),
                                            static_cast<float>(imglen / 8.0 * round(y_cross)));
                    }
                }
        }

        //telling_time("  clusters iteration");

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

        if (src.size() < 4)
            throw invalid_argument("i agree that what i found isn't accurate");

        //cv::imshow("warped points " + name, warped_points);
        Mat homo_fix = cv::findHomography(src, target, cv::RANSAC);
        homo_fix.convertTo(homo_fix, CV_32F);

        //telling_time("  the rest job");

        return homo_fix * best_homo;
    };

    //work_with_result_matrix(best_homo, "found corners");

    best_homo = work_points("homo fix");
    //telling_time("fix 1");
    best_homo = work_points("homo fix 2");
    //telling_time("fix 2");

    auto v_corners = work_with_result_matrix(best_homo, "fixed corners");

    //telling_time("inverse matrix ");
    for (auto &p: v_corners)
        p /= imglen;
    return v_corners;
}

std::vector<cv::Point2f> find_corners(const Mat& img_grayscale) {
    //cv::imshow("source", img_grayscale);

    Mat img_128;
    cv::resize(img_grayscale, img_128, {imglen0, imglen0});
    //cv::imshow("resized", img_128);

    auto[borders, dir] = sobel(img_128, 200);

    //telling_time("sobel");

    auto vert = Mat((85 <= dir) & (dir <= 95) & borders);
    auto hor = Mat(((170 <= dir) | (dir <= 10)) & borders);

    //telling_time("vert hor");
    vector<vector<pair<int, int> > > fake;
    Mat intersections = get_intersections(borders, hor, vert, imglen0, 2, fake);

    //telling_time("intersections");

    Mat img_512;
    cv::resize(img_grayscale, img_512, {imglen, imglen});


    return find_corners(img_512, borders, Mat(vert), Mat(hor), intersections,
                        calc_clusters(intersections, imglen0, true));
}

std::string test_speed(AAssetManager *mgr)
{
    std::vector<cv::Mat> frames;
    for (auto i = 1; i <= 28; i ++) {

        std::stringstream fname;
        fname << "pic" << i << ".jpg";
        AAsset *asset = AAssetManager_open(mgr, fname.str().c_str(), AASSET_MODE_BUFFER);

        std::stringstream shitstorm;
        shitstorm << asset;
        log_print("Dummy", "%s", shitstorm.str().c_str());

        auto fsize = AAsset_getLength64(asset);

        cv::Mat raw_data(1, fsize, CV_8UC1, (void*)AAsset_getBuffer(asset));
        cv::Mat frame = cv::imdecode(raw_data, cv::IMREAD_COLOR);
        if (frame.data == NULL) {
            log_print("Dummy", "error on reading: %s", fname.str().c_str());
        } else {
            // cv::resize(frame, frame, {0, 0}, 1.0/2.0, 1.0/2.0);
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            frames.push_back(frame);
        }
        AAsset_close(asset);
    }

    std::stringstream result;
    result << "Version: 1" << std::endl;

    for (int d : std::vector<int>({6, 4})) {
        for (int nthreads : std::vector<int>({1, 4})) {
            result << "d: " << d << ", threads: " << nthreads << std::endl;

            cv::setNumThreads(nthreads);

            /*
            here your code for initialization
            */

            for (int i = 0; i < 180; i++) {
                sins[i] = sin(i / 180.0 * acos(-1));
                coss[i] = cos(i / 180.0 * acos(-1));
            }

            auto start = std::chrono::steady_clock::now();
            int f = -1;
            int act_cnt = 0;
            for (auto frame: frames) {
                f ++;

                try{
                    auto v = find_corners(frame);
                    log_print("Dummy", "-----");
                    for (auto &p:v){
                        std::stringstream shitstorm;
                        shitstorm << p;
                        log_print("Dummy", "%s", shitstorm.str().c_str());
                    }
                } catch (std::invalid_argument e) {
                    log_print("Dummy", "-----");
                    log_print("Dummy", "fuckup");
                }

            }
            auto end = std::chrono::steady_clock::now();
            int time = int(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

            result << "Frames: " << frames.size() << ", Time: " << time <<  ", FPS: " << ((double)frames.size()/(double)time*1000.0) << ", act cnt: " << act_cnt;
            result << std::endl;
        }
    }

    return result.str();
}
