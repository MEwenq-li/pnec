/**
 BSD 3-Clause License

 This file is part of the PNEC project.
 https://github.com/tum-vision/pnec

 Copyright (c) 2022, Dominik Muhle.
 All rights reserved.
 */

#include <algorithm>
#include <array>
#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <sophus/se3.hpp>

#include "camera.h"
#include "config.h"
#include "converter.h"
#include "dataset_loader.h"
#include "frame_processing.h"
#include "odometry_output.h"
#include "pnec.h"
#include "timing.h"
#include "tracking_frame.h"
#include "tracking_matcher.h"

namespace {

void log_init() {
  boost::log::add_console_log(std::cout, boost::log::keywords::format =
                                             "[%Severity%] %Message%");
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::info);
}

struct KittiStereoCalibration {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double baseline = 0.0;
};

struct StereoObservation {
  Eigen::Vector2d right_pixel = Eigen::Vector2d::Zero();
  Eigen::Vector3d point_3d = Eigen::Vector3d::Zero();
  double disparity = 0.0;
};

using StereoObservationMap =
    std::unordered_map<pnec::features::KeyPointID, StereoObservation>;

KittiStereoCalibration LoadKittiStereoCalibration(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    std::cerr << "Failed to open KITTI calib file: " << path << std::endl;
    std::exit(-1);
  }

  auto parse_projection = [](const std::string &line) {
    std::stringstream ss(line);
    std::string label;
    ss >> label;
    std::array<double, 12> values{};
    for (size_t i = 0; i < values.size(); ++i) {
      ss >> values[i];
    }
    return values;
  };

  std::string line;
  std::array<double, 12> p0{};
  std::array<double, 12> p1{};
  bool got_p0 = false;
  bool got_p1 = false;
  while (std::getline(in, line)) {
    if (line.rfind("P0:", 0) == 0) {
      p0 = parse_projection(line);
      got_p0 = true;
    } else if (line.rfind("P1:", 0) == 0) {
      p1 = parse_projection(line);
      got_p1 = true;
    }
  }
  if (!got_p0 || !got_p1) {
    std::cerr << "Failed to parse P0/P1 from KITTI calib file: " << path
              << std::endl;
    std::exit(-1);
  }

  KittiStereoCalibration calib;
  calib.fx = p0[0];
  calib.fy = p0[5];
  calib.cx = p0[2];
  calib.cy = p0[6];
  calib.baseline = -p1[3] / p1[0];
  return calib;
}

StereoObservationMap MatchStereo(const cv::Mat &left_image,
                                 const cv::Mat &right_image,
                                 const pnec::features::KeyPoints &left_keypoints,
                                 const KittiStereoCalibration &stereo_calib) {
  StereoObservationMap observations;
  if (left_keypoints.empty()) {
    return observations;
  }

  std::vector<cv::Point2f> left_points;
  std::vector<cv::Point2f> right_points;
  std::vector<pnec::features::KeyPointID> ids;
  left_points.reserve(left_keypoints.size());
  right_points.reserve(left_keypoints.size());
  ids.reserve(left_keypoints.size());
  for (const auto &[id, keypoint] : left_keypoints) {
    left_points.emplace_back(static_cast<float>(keypoint.point_(0)),
                             static_cast<float>(keypoint.point_(1)));
    right_points.emplace_back(static_cast<float>(keypoint.point_(0)),
                              static_cast<float>(keypoint.point_(1)));
    ids.push_back(id);
  }

  std::vector<unsigned char> status;
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(
      left_image, right_image, left_points, right_points, status, error,
      cv::Size(21, 21), 3,
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30,
                       0.01),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  for (size_t i = 0; i < ids.size(); ++i) {
    if (!status[i]) {
      continue;
    }
    const cv::Point2f &left = left_points[i];
    const cv::Point2f &right = right_points[i];
    const double disparity = static_cast<double>(left.x - right.x);
    const double y_error = std::abs(static_cast<double>(left.y - right.y));
    if (disparity <= 0.5 || y_error > 1.5) {
      continue;
    }

    const double depth = stereo_calib.fx * stereo_calib.baseline / disparity;
    if (!std::isfinite(depth) || depth <= 0.0 || depth > 150.0) {
      continue;
    }

    StereoObservation observation;
    observation.right_pixel =
        Eigen::Vector2d(static_cast<double>(right.x), static_cast<double>(right.y));
    observation.disparity = disparity;
    observation.point_3d = Eigen::Vector3d(
        (static_cast<double>(left.x) - stereo_calib.cx) * depth /
            stereo_calib.fx,
        (static_cast<double>(left.y) - stereo_calib.cy) * depth /
            stereo_calib.fy,
        depth);
    observations.emplace(ids[i], observation);
  }

  return observations;
}

Eigen::Vector3d ComponentwiseMedian(
    const std::vector<Eigen::Vector3d> &translations) {
  if (translations.empty()) {
    return Eigen::Vector3d::Zero();
  }

  auto median_of_component = [&translations](int component) {
    std::vector<double> values;
    values.reserve(translations.size());
    for (const auto &translation : translations) {
      values.push_back(translation(component));
    }
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 == 1) {
      return *middle;
    }
    const double upper = *middle;
    std::nth_element(values.begin(), middle - 1, values.end());
    return 0.5 * (upper + *(middle - 1));
  };

  return Eigen::Vector3d(median_of_component(0), median_of_component(1),
                         median_of_component(2));
}

Eigen::Vector3d EstimateMetricTranslation(
    const Sophus::SO3d &rotation,
    const pnec::FeatureMatches &matches, const std::vector<int> &inliers,
    const StereoObservationMap &host_stereo,
    const StereoObservationMap &target_stereo) {
  std::vector<Eigen::Vector3d> translation_candidates;
  translation_candidates.reserve(inliers.size());
  for (const int inlier_idx : inliers) {
    const cv::DMatch &match = matches[inlier_idx];
    const auto host_it = host_stereo.find(match.queryIdx);
    const auto target_it = target_stereo.find(match.trainIdx);
    if (host_it == host_stereo.end() || target_it == target_stereo.end()) {
      continue;
    }
    translation_candidates.push_back(host_it->second.point_3d -
                                     rotation * target_it->second.point_3d);
  }

  if (translation_candidates.empty()) {
    return Eigen::Vector3d::Zero();
  }

  const Eigen::Vector3d median = ComponentwiseMedian(translation_candidates);

  std::vector<Eigen::Vector3d> refined_candidates;
  refined_candidates.reserve(translation_candidates.size());
  for (const auto &candidate : translation_candidates) {
    if ((candidate - median).norm() < 3.0) {
      refined_candidates.push_back(candidate);
    }
  }
  if (refined_candidates.empty()) {
    return median;
  }

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto &candidate : refined_candidates) {
    mean += candidate;
  }
  return mean / static_cast<double>(refined_candidates.size());
}

} // namespace

int main(int argc, const char *argv[]) {
  log_init();

  const cv::String keys =
      "{help h usage ?      |      | print this message                  }"
      "{@camera_config      |<none>| left camera config yaml             }"
      "{@pnec_config        |<none>| pnec/nec config file                }"
      "{@tracking_calib     |<none>| tracking calibration file           }"
      "{@tracking_config    |<none>| tracking config file                }"
      "{@left_sequence_path |<none>| path to left images                 }"
      "{@right_sequence_path|<none>| path to right images                }"
      "{@timestamp_path     |<none>| path to timestamps                  }"
      "{@stereo_calib_path  |<none>| KITTI calib.txt path                }"
      "{@results            |<none>| path to store results               }"
      "{@visualization      |<none>| path to store visualization frames  }"
      "{@no_skip            |<none>| don't skip any frames               }"
      "{image_ext           |.png  | image extension                     }"
      "{gt                  |      | ground truth                        }";

  std::string licence_notice =
      "PNEC Copyright (C) 2022 Dominik Muhle\n"
      "PNEC comes with ABSOLUTELY NO WARRANTY.\n"
      "    This is free software, and you are welcome to redistribute it\n"
      "    under certain conditions; visit\n"
      "    https://github.com/tum-vision/pnec for details.\n"
      "\n";
  std::cout << licence_notice << std::endl;

  cv::CommandLineParser parser(argc, argv, keys);
  size_t parser_counter = 0;
  const std::string camera_config_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string pnec_config_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string tracking_calib_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string tracking_config_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string left_sequence_path(
      parser.get<cv::String>(parser_counter++));
  const std::string right_sequence_path(
      parser.get<cv::String>(parser_counter++));
  const std::string timestamp_path(parser.get<cv::String>(parser_counter++));
  const std::string stereo_calib_path(
      parser.get<cv::String>(parser_counter++));
  const std::string results_path(parser.get<cv::String>(parser_counter++));
  const std::string visualization_path(
      parser.get<cv::String>(parser_counter++));
  const bool no_skip(parser.get<bool>(parser_counter++));
  const std::string image_ext(parser.get<cv::String>("image_ext"));
  const bool gt_provided = parser.has("gt");
  const std::string gt_path =
      gt_provided ? parser.get<cv::String>("gt") : std::string();

  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  if (!boost::filesystem::exists(results_path)) {
    boost::filesystem::create_directories(results_path);
  }
  const boost::filesystem::path poses_dir =
      boost::filesystem::path(results_path) / "rot_avg";
  if (!boost::filesystem::exists(poses_dir)) {
    boost::filesystem::create_directories(poses_dir);
  }
  if (!visualization_path.empty() &&
      !boost::filesystem::exists(visualization_path)) {
    boost::filesystem::create_directories(visualization_path);
  }

  pnec::CameraParameters cam_parameters =
      pnec::input::LoadCameraConfig(camera_config_filename);
  pnec::Camera::instance().init(cam_parameters);
  pnec::rel_pose_estimation::Options nec_options =
      pnec::input::LoadPNECConfig(pnec_config_filename);
  nec_options.use_nec_ = true;
  basalt::Calibration<double> tracking_calib =
      pnec::input::LoadTrackingCalib(tracking_calib_filename);
  basalt::VioConfig tracking_config =
      pnec::input::LoadTrackingConfig(tracking_config_filename);
  const KittiStereoCalibration stereo_calib =
      LoadKittiStereoCalibration(stereo_calib_path);

  pnec::input::DatasetLoader loader(left_sequence_path, image_ext, timestamp_path,
                                    false, 1.0);

  std::vector<Sophus::SE3d> gt_poses;
  std::vector<Sophus::SE3d> rel_gt_poses;
  if (gt_provided) {
    pnec::input::LoadGroundTruth(gt_path, gt_poses, rel_gt_poses);
  }

  basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> tracking(
      tracking_config, tracking_calib, true, false);
  pnec::features::TrackingMatcher matcher(50, 5.0);
  pnec::rel_pose_estimation::PNEC nec_solver(nec_options);

  pnec::common::Timing timing;
  pnec::frames::BaseFrame::Ptr prev_frame;
  StereoObservationMap prev_stereo;
  Sophus::SE3d prev_global_pose = Sophus::SE3d();
  Sophus::SE3d prev_rel_pose = Sophus::SE3d();
  bool have_prev_selected_frame = false;
  std::ios_base::openmode pose_output_mode = std::ios_base::out;

  for (const auto &image : loader) {
    pnec::common::FrameTiming frame_timing(image.id_);
    const std::string left_image_path = image.path_.string();
    const boost::filesystem::path right_image_path =
        boost::filesystem::path(right_sequence_path) / image.path_.filename();

    cv::Mat right_image =
        cv::imread(right_image_path.string(), cv::IMREAD_GRAYSCALE);
    if (right_image.empty()) {
      BOOST_LOG_TRIVIAL(error) << "Failed to read right image "
                               << right_image_path.string();
      continue;
    }

    pnec::frames::BaseFrame::Ptr curr_frame;
    curr_frame.reset(new pnec::frames::TrackingFrame(
        image.id_, image.timestamp_, left_image_path, tracking, frame_timing));
    cv::Mat left_image = curr_frame->getImage();

    auto stereo_tic = std::chrono::high_resolution_clock::now();
    StereoObservationMap curr_stereo =
        MatchStereo(left_image, right_image, curr_frame->keypoints(),
                    stereo_calib);
    auto stereo_toc = std::chrono::high_resolution_clock::now();
    frame_timing.feature_creation_ +=
        std::chrono::duration_cast<std::chrono::milliseconds>(stereo_toc -
                                                              stereo_tic);

    if (!have_prev_selected_frame) {
      pnec::out::SavePose((poses_dir.string() + "/"), "poses",
                          curr_frame->Timestamp(), Sophus::SE3d(),
                          pose_output_mode);
      pose_output_mode = std::ios_base::app;
      prev_frame = curr_frame;
      prev_stereo = std::move(curr_stereo);
      prev_global_pose = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
      have_prev_selected_frame = true;
      timing.push_back(frame_timing);
      continue;
    }

    bool skipping_frame = false;
    pnec::FeatureMatches matches =
        matcher.FindMatches(prev_frame, curr_frame, skipping_frame);
    if (skipping_frame && !no_skip) {
      continue;
    }

    opengv::bearingVectors_t host_bvs;
    opengv::bearingVectors_t target_bvs;
    std::vector<Eigen::Matrix3d> dummy_covariances;
    host_bvs.reserve(matches.size());
    target_bvs.reserve(matches.size());
    dummy_covariances.reserve(matches.size());
    pnec::FeatureMatches stereo_supported_matches;
    stereo_supported_matches.reserve(matches.size());
    for (const auto &match : matches) {
      const auto host_stereo_it = prev_stereo.find(match.queryIdx);
      const auto target_stereo_it = curr_stereo.find(match.trainIdx);
      if (host_stereo_it == prev_stereo.end() ||
          target_stereo_it == curr_stereo.end()) {
        continue;
      }
      const auto &host_keypoint = prev_frame->keypoints()[match.queryIdx];
      const auto &target_keypoint = curr_frame->keypoints()[match.trainIdx];
      host_bvs.push_back(host_keypoint.bearing_vector_);
      target_bvs.push_back(target_keypoint.bearing_vector_);
      dummy_covariances.push_back(Eigen::Matrix3d::Identity());
      stereo_supported_matches.push_back(match);
    }

    if (stereo_supported_matches.size() <
        static_cast<size_t>(nec_options.min_matches_)) {
      continue;
    }

    std::vector<int> inliers;
    Sophus::SE3d nec_pose = nec_solver.Solve(host_bvs, target_bvs,
                                             dummy_covariances, prev_rel_pose,
                                             inliers, frame_timing);
    if (inliers.size() < static_cast<size_t>(nec_options.min_inliers_)) {
      continue;
    }

    const Eigen::Vector3d metric_translation = EstimateMetricTranslation(
        nec_pose.so3(), stereo_supported_matches, inliers, prev_stereo,
        curr_stereo);
    Sophus::SE3d rel_pose(nec_pose.so3(), metric_translation);
    const Sophus::SE3d curr_global_pose = prev_global_pose * rel_pose;

    pnec::out::SavePose((poses_dir.string() + "/"), "poses",
                        curr_frame->Timestamp(), curr_global_pose,
                        pose_output_mode);

    timing.push_back(frame_timing);
    BOOST_LOG_TRIVIAL(info)
        << "Processed stereo NEC frame " << curr_frame->id() << " in "
        << frame_timing.TotalTime() << " milliseconds";

    prev_frame = curr_frame;
    prev_stereo = std::move(curr_stereo);
    prev_global_pose = curr_global_pose;
    prev_rel_pose = rel_pose;
  }

  std::ofstream out(results_path + "timing.txt", std::ios_base::trunc);
  out << timing;
  return 0;
}
