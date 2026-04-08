/**
 BSD 3-Clause License

 This file is part of the PNEC project.
 https://github.com/tum-vision/pnec

 Copyright (c) 2022, Dominik Muhle.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "frame2frame.h"

#include "math.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <opencv2/core/eigen.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>
#include <opengv/types.hpp>

#include "common.h"
#include "essential_matrix_methods.h"
#include "odometry_output.h"
#include "pnec.h"

namespace pnec {
namespace rel_pose_estimation {

Frame2Frame::Frame2Frame(const pnec::rel_pose_estimation::Options &options)
    : options_{options} {}
Frame2Frame::~Frame2Frame() {}

Sophus::SE3d Frame2Frame::Align(pnec::frames::BaseFrame::Ptr frame1,
                                pnec::frames::BaseFrame::Ptr frame2,
                                pnec::FeatureMatches &matches,
                                Sophus::SE3d prev_rel_pose,
                                std::vector<int> &inliers,
                                pnec::common::FrameTiming &frame_timing,
                                bool ablation, std::string ablation_folder,
                                std::string results_folder) {
  if (ablation) {
    if (!boost::filesystem::exists(ablation_folder)) {
      boost::filesystem::create_directory(ablation_folder);
    }
  }

  curr_timestamp_ = frame2->Timestamp();

  FeatureBundle features;
  GetFeatures(frame1, frame2, matches, features);
  if (!options_.use_nec_) {
    ApplyCovarianceExperimentMode(features);
  }
  if (options_.dump_covariance_stats_ && !results_folder.empty()) {
    DumpCovarianceStats(features, results_folder);
  }

  if (ablation) {
    AblationAlign(features.host_bvs, features.target_bvs,
                  features.projected_covs, ablation_folder);
  }

  return PNECAlign(features, prev_rel_pose, inliers, frame_timing,
                   results_folder, ablation_folder);
}

Sophus::SE3d Frame2Frame::AlignFurther(pnec::frames::BaseFrame::Ptr frame1,
                                       pnec::frames::BaseFrame::Ptr frame2,
                                       pnec::FeatureMatches &matches,
                                       Sophus::SE3d prev_rel_pose,
                                       std::vector<int> &inliers,
                                       bool &success) {
  opengv::bearingVectors_t bvs1;
  opengv::bearingVectors_t bvs2;
  std::vector<Eigen::Matrix3d> proj_covs;
  FeatureBundle features;
  GetFeatures(frame1, frame2, matches, features);
  if (!options_.use_nec_) {
    ApplyCovarianceExperimentMode(features);
  }

  pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  Sophus::SE3d rel_pose = PNECAlign(features, prev_rel_pose, inliers,
                                    dummy_timing);

  if (ransac_iterations_ >= options_.max_ransac_iterations_) {
    success = false;
  } else if (inliers.size() < options_.min_inliers_ && inliers.size() != 0) {
    success = false;
  } else {
    success = true;
  }
  return rel_pose;
}

const std::map<std::string, std::vector<std::pair<double, Sophus::SE3d>>> &
Frame2Frame::GetAblationResults() const {
  return ablation_rel_poses_;
}

Sophus::SE3d Frame2Frame::PNECAlign(
    const FeatureBundle &features,
    Sophus::SE3d prev_rel_pose, std::vector<int> &inliers,
    pnec::common::FrameTiming &frame_timing, std::string results_folder,
    std::string ablation_folder) {
  pnec::rel_pose_estimation::PNEC pnec(options_);

  Sophus::SE3d rel_pose;
  if (!options_.use_nec_ && options_.noise_frame_ == pnec::common::Both) {
    rel_pose = pnec.Solve(features.host_bvs, features.target_bvs,
                          features.projected_covs, features.host_covs,
                          features.target_covs, prev_rel_pose, inliers,
                          frame_timing);
  } else {
    rel_pose = pnec.Solve(features.host_bvs, features.target_bvs,
                          features.projected_covs, prev_rel_pose, inliers,
                          frame_timing);
  }

  if (ablation_rel_poses_.count("PNEC") == 0) {
    ablation_rel_poses_["PNEC"] = {
        std::make_pair<double, Sophus::SE3d>(0.0, Sophus::SE3d())};
  }
  ablation_rel_poses_["PNEC"].push_back(
      std::make_pair(curr_timestamp_, rel_pose));
  return rel_pose;
}

void Frame2Frame::AblationAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    std::string ablation_folder) {
  BOOST_LOG_TRIVIAL(debug) << "Using ablation methods";

  auto GetPrevRelPose = [this](std::string method_name) -> Sophus::SE3d {
    Sophus::SE3d prev_rel_pose;
    if (ablation_rel_poses_.count(method_name) == 0) {
      BOOST_LOG_TRIVIAL(debug) << "Didn't previous pose for " << method_name
                               << ", returning identity";
      ablation_rel_poses_[method_name] = {
          std::make_pair<double, Sophus::SE3d>(0.0, Sophus::SE3d())};
      prev_rel_pose = Sophus::SE3d();
    } else {
      BOOST_LOG_TRIVIAL(debug) << "Loading previous pose for " << method_name;
      prev_rel_pose = ablation_rel_poses_[method_name].back().second;
    }
    return prev_rel_pose;
  };

  {
    std::string name = "NEC";

    Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;
    nec_options.use_ceres_ = false;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    ablation_rel_poses_[name].push_back(
        std::make_pair(curr_timestamp_, rel_pose));
  }

  {
    std::string name = "NEC-LS";

    Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    ablation_rel_poses_[name].push_back(
        std::make_pair(curr_timestamp_, rel_pose));
  }

  // {
  //   std::string name = "NEC+PNEC-LS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 1;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNECwoLS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.use_ceres_ = false;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNEConlyLS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 0;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNEConlyLSfull";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 0;
  //   options.ceres_options_.max_num_iterations = 10000;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::vector<size_t> iterations = {5, 15};

  //   for (const auto &max_it : iterations) {
  //     {
  //       std::string name = "PNEC-" + std::to_string(max_it);

  //       Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //       pnec::rel_pose_estimation::Options options = options_;
  //       options.weighted_iterations_ = max_it;

  //       pnec::rel_pose_estimation::PNEC pnec(options);
  //       pnec::common::FrameTiming dummy_timing =
  //       pnec::common::FrameTiming(0); Sophus::SE3d rel_pose =
  //           pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose,
  //           dummy_timing);

  //       ablation_rel_poses_[name].push_back(
  //           std::make_pair(curr_timestamp_, rel_pose));
  //       pnec::out::SavePose(ablation_folder, name, curr_timestamp_,
  //       rel_pose);
  //     }
  //   }
  // }

  // {
  //   std::string name = "8pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           EIGHTPT);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "Nister5pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           NISTER);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "Stewenius5pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           STEWENIUS);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "7pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           SEVENPT);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }
}

void Frame2Frame::GetFeatures(pnec::frames::BaseFrame::Ptr host_frame,
                              pnec::frames::BaseFrame::Ptr target_frame,
                              pnec::FeatureMatches &matches,
                              FeatureBundle &features) {
  std::vector<size_t> host_matches;
  std::vector<size_t> target_matches;
  for (const auto &match : matches) {
    host_matches.push_back(match.queryIdx);
    target_matches.push_back(match.trainIdx);
  }
  pnec::features::KeyPoints host_keypoints =
      host_frame->keypoints(host_matches);
  pnec::features::KeyPoints target_keypoints =
      target_frame->keypoints(target_matches);

  for (auto const &[id, keypoint] : host_keypoints) {
    features.host_bvs.push_back(keypoint.bearing_vector_);
    features.host_covs.push_back(keypoint.bv_covariance_);
  }
  for (auto const &[id, keypoint] : target_keypoints) {
    features.target_bvs.push_back(keypoint.bearing_vector_);
    features.target_covs.push_back(keypoint.bv_covariance_);
  }

  if (options_.noise_frame_ == pnec::common::Host) {
    features.projected_covs = features.host_covs;
  } else {
    features.projected_covs = features.target_covs;
  }
}

void Frame2Frame::ApplyCovarianceExperimentMode(FeatureBundle &features) const {
  auto transform_covariances =
      [this](std::vector<Eigen::Matrix3d> &covariances) {
        for (auto &covariance : covariances) {
          switch (options_.covariance_mode_) {
          case CovarianceExperimentMode::Original:
            break;
          case CovarianceExperimentMode::Isotropic:
            covariance = Eigen::Matrix3d::Identity() *
                         options_.isotropic_covariance_value_;
            break;
          case CovarianceExperimentMode::Diagonal:
            covariance = covariance.diagonal().asDiagonal();
            break;
          case CovarianceExperimentMode::Normalized: {
            const double trace = covariance.trace();
            if (trace > std::numeric_limits<double>::epsilon()) {
              covariance *= options_.normalized_covariance_trace_ / trace;
            }
            break;
          }
          }
        }
      };

  transform_covariances(features.host_covs);
  transform_covariances(features.target_covs);

  if (options_.noise_frame_ == pnec::common::Host) {
    features.projected_covs = features.host_covs;
  } else {
    features.projected_covs = features.target_covs;
  }
}

std::string
Frame2Frame::CovarianceExperimentModeName(CovarianceExperimentMode mode) {
  switch (mode) {
  case CovarianceExperimentMode::Original:
    return "Original";
  case CovarianceExperimentMode::Isotropic:
    return "Isotropic";
  case CovarianceExperimentMode::Diagonal:
    return "Diagonal";
  case CovarianceExperimentMode::Normalized:
    return "Normalized";
  }
  return "Unknown";
}

namespace {
struct CovarianceStatsSummary {
  size_t count = 0;
  double mean_trace = 0.0;
  double mean_min_eig = 0.0;
  double mean_max_eig = 0.0;
  double mean_condition = 0.0;
};

CovarianceStatsSummary SummarizeCovariances(
    const std::vector<Eigen::Matrix3d> &covariances) {
  CovarianceStatsSummary summary;
  summary.count = covariances.size();
  if (covariances.empty()) {
    return summary;
  }

  for (const auto &covariance : covariances) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
    const Eigen::Vector3d eigenvalues = eig.eigenvalues();
    const double min_eig = eigenvalues.minCoeff();
    const double max_eig = eigenvalues.maxCoeff();
    const double safe_min =
        std::max(std::abs(min_eig), std::numeric_limits<double>::epsilon());
    summary.mean_trace += covariance.trace();
    summary.mean_min_eig += min_eig;
    summary.mean_max_eig += max_eig;
    summary.mean_condition += max_eig / safe_min;
  }

  const double inv_count = 1.0 / static_cast<double>(covariances.size());
  summary.mean_trace *= inv_count;
  summary.mean_min_eig *= inv_count;
  summary.mean_max_eig *= inv_count;
  summary.mean_condition *= inv_count;
  return summary;
}
} // namespace

void Frame2Frame::DumpCovarianceStats(const FeatureBundle &features,
                                      const std::string &results_folder) const {
  if (results_folder.empty()) {
    return;
  }

  const boost::filesystem::path stats_path =
      boost::filesystem::path(results_folder) / "covariance_stats.csv";
  const bool write_header = !boost::filesystem::exists(stats_path);

  const CovarianceStatsSummary host_summary =
      SummarizeCovariances(features.host_covs);
  const CovarianceStatsSummary target_summary =
      SummarizeCovariances(features.target_covs);
  const CovarianceStatsSummary projected_summary =
      SummarizeCovariances(features.projected_covs);

  std::ofstream out(stats_path.string(), std::ios_base::app);
  if (write_header) {
    out << "timestamp,matches,covariance_mode,noise_frame,"
        << "host_count,host_mean_trace,host_mean_min_eig,host_mean_max_eig,host_mean_condition,"
        << "target_count,target_mean_trace,target_mean_min_eig,target_mean_max_eig,target_mean_condition,"
        << "projected_count,projected_mean_trace,projected_mean_min_eig,projected_mean_max_eig,projected_mean_condition\n";
  }

  out << std::fixed << std::setprecision(9) << curr_timestamp_ << ","
      << features.projected_covs.size() << ","
      << CovarianceExperimentModeName(options_.covariance_mode_) << ","
      << (options_.noise_frame_ == pnec::common::Host
              ? "Host"
              : (options_.noise_frame_ == pnec::common::Both ? "Both"
                                                             : "Target"))
      << "," << host_summary.count << "," << host_summary.mean_trace << ","
      << host_summary.mean_min_eig << "," << host_summary.mean_max_eig << ","
      << host_summary.mean_condition << "," << target_summary.count << ","
      << target_summary.mean_trace << "," << target_summary.mean_min_eig
      << "," << target_summary.mean_max_eig << ","
      << target_summary.mean_condition << "," << projected_summary.count << ","
      << projected_summary.mean_trace << ","
      << projected_summary.mean_min_eig << ","
      << projected_summary.mean_max_eig << ","
      << projected_summary.mean_condition << "\n";
}

} // namespace rel_pose_estimation
} // namespace pnec
