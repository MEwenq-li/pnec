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

#include "pnec.h"

#include "math.h"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
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
#include "ceres_compat.h"
#include "nec_ceres.h"
#include "pnec_ceres.h"
#include "scf.h"

namespace pnec {
namespace rel_pose_estimation {

namespace {

struct WeightedMinEigenvalueResidual {
  WeightedMinEigenvalueResidual(const opengv::bearingVectors_t &bvs1,
                                const opengv::bearingVectors_t &bvs2,
                                const std::vector<double> &weights)
      : bvs1_{bvs1}, bvs2_{bvs2}, weights_{weights} {}

  bool operator()(const double *const orientation_ptr,
                  double *residual_ptr) const {
    Eigen::Map<const Eigen::Quaterniond> orientation(orientation_ptr);
    const Eigen::Matrix3d rotation = orientation.normalized().toRotationMatrix();

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < bvs1_.size(); ++i) {
      const Eigen::Vector3d n = bvs1_[i].cross(rotation * bvs2_[i]);
      M += weights_[i] * n * n.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(M);
    if (eigensolver.info() != Eigen::Success) {
      residual_ptr[0] = 1.0e12;
      return true;
    }

    const double min_eigenvalue = std::max(eigensolver.eigenvalues()[0], 0.0);
    residual_ptr[0] = std::sqrt(min_eigenvalue + 1.0e-18);
    return true;
  }

private:
  const opengv::bearingVectors_t &bvs1_;
  const opengv::bearingVectors_t &bvs2_;
  const std::vector<double> &weights_;
};

std::vector<double> NormalizeWeights(const std::vector<double> &weights) {
  if (weights.empty()) {
    return weights;
  }

  double weight_sum = 0.0;
  for (const double weight : weights) {
    weight_sum += weight;
  }

  const double mean_weight =
      std::max(weight_sum / static_cast<double>(weights.size()), 1.0e-18);

  std::vector<double> normalized_weights;
  normalized_weights.reserve(weights.size());
  for (const double weight : weights) {
    normalized_weights.push_back(std::max(weight / mean_weight, 1.0e-18));
  }
  return normalized_weights;
}

Eigen::Matrix3d OptimizeWeightedRotationPaperLike(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<double> &weights, const Eigen::Matrix3d &initial_rotation,
    const ceres::Solver::Options &base_options) {
  Eigen::Quaterniond orientation(initial_rotation);

  ceres::Problem problem;
  ceres::CostFunction *cost_function =
      new ceres::NumericDiffCostFunction<WeightedMinEigenvalueResidual,
                                         ceres::CENTRAL, 1, 4>(
          new WeightedMinEigenvalueResidual(bvs1, bvs2, weights));
  problem.AddResidualBlock(cost_function, nullptr, orientation.coeffs().data());
  pnec::optimization::SetEigenQuaternionParameterization(
      problem, orientation.coeffs().data());

  ceres::Solver::Options options = base_options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations =
      std::max(20, static_cast<int>(base_options.max_num_iterations));
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return orientation.normalized().toRotationMatrix();
}

Sophus::SE3d SelectPNECCeresInit(
    const pnec::rel_pose_estimation::Options &options, PNEC *solver,
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covariances,
    const Sophus::SE3d &initial_pose, const Sophus::SE3d &es_solution) {
  switch (options.ceres_init_mode_) {
  case CeresInitMode::NEC:
    return es_solution;
  case CeresInitMode::NECCeres:
    return solver->NECCeresSolver(bvs1, bvs2, es_solution);
  case CeresInitMode::Initial:
    return initial_pose;
  case CeresInitMode::Weighted:
  default:
    if (options.weighted_iterations_ > 1) {
      return solver->WeightedEigensolver(bvs1, bvs2, projected_covariances,
                                         es_solution);
    }
    if (options.weighted_iterations_ == 1) {
      return es_solution;
    }
    return initial_pose;
  }
}

} // namespace

PNEC::PNEC(const pnec::rel_pose_estimation::Options &options)
    : options_{options} {}

PNEC::~PNEC() {}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose) {
  std::vector<int> inliers;
  return Solve(bvs1, bvs2, projected_covs, initial_pose, inliers);
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose,
                         std::vector<int> &inliers) {
  opengv::bearingVectors_t in_bvs1;
  opengv::bearingVectors_t in_bvs2;
  std::vector<Eigen::Matrix3d> in_proj_covs;

  Sophus::SE3d ES_solution = Eigensolver(bvs1, bvs2, initial_pose, inliers);
  if (options_.use_ransac_) {
    InlierExtraction(bvs1, bvs2, projected_covs, in_bvs1, in_bvs2, in_proj_covs,
                     inliers);
  } else {
    in_bvs1 = bvs1;
    in_bvs2 = bvs2;
    in_proj_covs = projected_covs;
  }

  if (options_.use_nec_) {
    if (options_.use_ceres_) {
      BOOST_LOG_TRIVIAL(debug) << "NECCeres" << std::endl;
      return NECCeresSolver(in_bvs1, in_bvs2, ES_solution);
    } else {
      return ES_solution;
    }
  } else {
    Sophus::SE3d ceres_init = SelectPNECCeresInit(
        options_, this, in_bvs1, in_bvs2, in_proj_covs, initial_pose,
        ES_solution);

    Sophus::SE3d solution;
    if (options_.use_ceres_) {
      BOOST_LOG_TRIVIAL(debug) << "Ceres" << std::endl;
      solution = CeresSolver(in_bvs1, in_bvs2, in_proj_covs, ceres_init);
    } else {
      solution = ceres_init;
    }
    return solution;
  }
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const std::vector<Eigen::Matrix3d> &host_covs,
                         const std::vector<Eigen::Matrix3d> &target_covs,
                         const Sophus::SE3d &initial_pose,
                         std::vector<int> &inliers) {
  pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  return Solve(bvs1, bvs2, projected_covs, host_covs, target_covs,
               initial_pose, inliers, dummy_timing);
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose,
                         pnec::common::FrameTiming &timing) {
  std::vector<int> inliers;
  return Solve(bvs1, bvs2, projected_covs, initial_pose, inliers, timing);
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose,
                         std::vector<int> &inliers,
                         pnec::common::FrameTiming &timing) {
  opengv::bearingVectors_t in_bvs1;
  opengv::bearingVectors_t in_bvs2;
  std::vector<Eigen::Matrix3d> in_proj_covs;

  auto tic = std::chrono::high_resolution_clock::now(),
       toc = std::chrono::high_resolution_clock::now();
  Sophus::SE3d ES_solution = Eigensolver(bvs1, bvs2, initial_pose, inliers);
  if (options_.use_ransac_) {
    InlierExtraction(bvs1, bvs2, projected_covs, in_bvs1, in_bvs2, in_proj_covs,
                     inliers);
  } else {
    in_bvs1 = bvs1;
    in_bvs2 = bvs2;
    in_proj_covs = projected_covs;
  }
  toc = std::chrono::high_resolution_clock::now();
  timing.nec_es_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

  if (options_.use_nec_) {
    Sophus::SE3d solution;
    if (options_.use_ceres_) {
      BOOST_LOG_TRIVIAL(debug) << "NECCeres" << std::endl;
      tic = std::chrono::high_resolution_clock::now();
      solution = NECCeresSolver(in_bvs1, in_bvs2, ES_solution);
      toc = std::chrono::high_resolution_clock::now();
      timing.ceres_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    } else {
      solution = ES_solution;
      timing.ceres_ = std::chrono::milliseconds(0);
    }
    return solution;
  } else {
    Sophus::SE3d ceres_init;
    if (options_.ceres_init_mode_ == CeresInitMode::Weighted &&
        options_.weighted_iterations_ > 1) {
      BOOST_LOG_TRIVIAL(debug) << "Weighted" << std::endl;
      tic = std::chrono::high_resolution_clock::now();
      ceres_init =
          WeightedEigensolver(in_bvs1, in_bvs2, in_proj_covs, ES_solution);
      toc = std::chrono::high_resolution_clock::now();
      timing.it_es_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
      timing.avg_it_es_ = std::chrono::duration_cast<std::chrono::milliseconds>(
          (toc - tic) / options_.weighted_iterations_);
    } else {
      ceres_init = SelectPNECCeresInit(options_, this, in_bvs1, in_bvs2,
                                       in_proj_covs, initial_pose,
                                       ES_solution);
      timing.it_es_ = std::chrono::milliseconds(0);
      timing.avg_it_es_ = std::chrono::milliseconds(0);
    }
    Sophus::SE3d solution;
    if (options_.use_ceres_) {
      BOOST_LOG_TRIVIAL(debug) << "Ceres" << std::endl;
      tic = std::chrono::high_resolution_clock::now();
      solution = CeresSolver(in_bvs1, in_bvs2, in_proj_covs, ceres_init);
      toc = std::chrono::high_resolution_clock::now();
      timing.ceres_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    } else {
      solution = ceres_init;
      timing.ceres_ = std::chrono::milliseconds(0);
    }
    return solution;
  }
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const std::vector<Eigen::Matrix3d> &host_covs,
                         const std::vector<Eigen::Matrix3d> &target_covs,
                         const Sophus::SE3d &initial_pose,
                         std::vector<int> &inliers,
                         pnec::common::FrameTiming &timing) {
  opengv::bearingVectors_t in_bvs1;
  opengv::bearingVectors_t in_bvs2;
  std::vector<Eigen::Matrix3d> in_proj_covs;
  std::vector<Eigen::Matrix3d> in_host_covs;
  std::vector<Eigen::Matrix3d> in_target_covs;

  auto tic = std::chrono::high_resolution_clock::now(),
       toc = std::chrono::high_resolution_clock::now();
  Sophus::SE3d ES_solution = Eigensolver(bvs1, bvs2, initial_pose, inliers);
  if (options_.use_ransac_) {
    InlierExtraction(bvs1, bvs2, projected_covs, host_covs, target_covs,
                     in_bvs1, in_bvs2, in_proj_covs, in_host_covs,
                     in_target_covs, inliers);
  } else {
    in_bvs1 = bvs1;
    in_bvs2 = bvs2;
    in_proj_covs = projected_covs;
    in_host_covs = host_covs;
    in_target_covs = target_covs;
  }
  toc = std::chrono::high_resolution_clock::now();
  timing.nec_es_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

  if (options_.use_nec_) {
    Sophus::SE3d solution;
    if (options_.use_ceres_) {
      BOOST_LOG_TRIVIAL(debug) << "NECCeres" << std::endl;
      tic = std::chrono::high_resolution_clock::now();
      solution = NECCeresSolver(in_bvs1, in_bvs2, ES_solution);
      toc = std::chrono::high_resolution_clock::now();
      timing.ceres_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    } else {
      solution = ES_solution;
      timing.ceres_ = std::chrono::milliseconds(0);
    }
    return solution;
  }

  Sophus::SE3d ceres_init;
  if (options_.ceres_init_mode_ == CeresInitMode::Weighted &&
      options_.weighted_iterations_ > 1) {
    BOOST_LOG_TRIVIAL(debug) << "Weighted" << std::endl;
    tic = std::chrono::high_resolution_clock::now();
    ceres_init =
        WeightedEigensolver(in_bvs1, in_bvs2, in_proj_covs, ES_solution);
    toc = std::chrono::high_resolution_clock::now();
    timing.it_es_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    timing.avg_it_es_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        (toc - tic) / options_.weighted_iterations_);
  } else {
    ceres_init = SelectPNECCeresInit(options_, this, in_bvs1, in_bvs2,
                                     in_proj_covs, initial_pose, ES_solution);
    timing.it_es_ = std::chrono::milliseconds(0);
    timing.avg_it_es_ = std::chrono::milliseconds(0);
  }

  Sophus::SE3d solution;
  if (options_.use_ceres_) {
    BOOST_LOG_TRIVIAL(debug) << "CeresSymmetric" << std::endl;
    tic = std::chrono::high_resolution_clock::now();
    solution =
        CeresSolver(in_bvs1, in_bvs2, in_host_covs, in_target_covs, ceres_init);
    toc = std::chrono::high_resolution_clock::now();
    timing.ceres_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
  } else {
    solution = ceres_init;
    timing.ceres_ = std::chrono::milliseconds(0);
  }
  return solution;
}

void PNEC::InlierExtraction(const opengv::bearingVectors_t &bvs1,
                            const opengv::bearingVectors_t &bvs2,
                            const std::vector<Eigen::Matrix3d> &proj_covs,
                            opengv::bearingVectors_t &in_bvs1,
                            opengv::bearingVectors_t &in_bvs2,
                            std::vector<Eigen::Matrix3d> &in_proj_covs,
                            const std::vector<int> &inliers) {
  in_bvs1.clear();
  in_bvs1.reserve(inliers.size());
  in_bvs2.clear();
  in_bvs2.reserve(inliers.size());
  in_proj_covs.clear();
  in_proj_covs.reserve(inliers.size());
  for (const auto inlier : inliers) {
    // for (int inlier = 0; inlier < n_matches; inlier++) {
    in_bvs1.push_back(bvs1[inlier]);
    in_bvs2.push_back(bvs2[inlier]);
    in_proj_covs.push_back(proj_covs[inlier]);
  }
}

void PNEC::InlierExtraction(const opengv::bearingVectors_t &bvs1,
                            const opengv::bearingVectors_t &bvs2,
                            const std::vector<Eigen::Matrix3d> &proj_covs,
                            const std::vector<Eigen::Matrix3d> &host_covs,
                            const std::vector<Eigen::Matrix3d> &target_covs,
                            opengv::bearingVectors_t &in_bvs1,
                            opengv::bearingVectors_t &in_bvs2,
                            std::vector<Eigen::Matrix3d> &in_proj_covs,
                            std::vector<Eigen::Matrix3d> &in_host_covs,
                            std::vector<Eigen::Matrix3d> &in_target_covs,
                            const std::vector<int> &inliers) {
  InlierExtraction(bvs1, bvs2, proj_covs, in_bvs1, in_bvs2, in_proj_covs,
                   inliers);
  in_host_covs.clear();
  in_host_covs.reserve(inliers.size());
  in_target_covs.clear();
  in_target_covs.reserve(inliers.size());
  for (const auto inlier : inliers) {
    in_host_covs.push_back(host_covs[inlier]);
    in_target_covs.push_back(target_covs[inlier]);
  }
}

Sophus::SE3d PNEC::Eigensolver(const opengv::bearingVectors_t &bvs1,
                               const opengv::bearingVectors_t &bvs2,
                               const Sophus::SE3d &initial_pose,
                               std::vector<int> &inliers) {
  opengv::rotation_t init_rotation = initial_pose.rotationMatrix();
  opengv::relative_pose::CentralRelativeAdapter adapter(bvs1, bvs2,
                                                        init_rotation);
  Sophus::SE3d solution;
  if (options_.use_ransac_) {
    opengv::sac::Ransac<
        opengv::sac_problems::relative_pose::EigensolverSacProblem>
        ransac;
    std::shared_ptr<opengv::sac_problems::relative_pose::EigensolverSacProblem>
        eigenproblem_ptr(
            new opengv::sac_problems::relative_pose::EigensolverSacProblem(
                adapter, options_.ransac_sample_size_));
    ransac.sac_model_ = eigenproblem_ptr;
    ransac.threshold_ = options_.ransac_threshold_;
    ransac.max_iterations_ = options_.max_ransac_iterations_;

    ransac.computeModel();

    opengv::sac_problems::relative_pose::EigensolverSacProblem::model_t
        optimizedModel;
    eigenproblem_ptr->optimizeModelCoefficients(
        ransac.inliers_, ransac.model_coefficients_, optimizedModel);

    inliers = ransac.inliers_;
    // TODO: use translation from M
    opengv::bearingVectors_t in_bvs1;
    opengv::bearingVectors_t in_bvs2;
    in_bvs1.reserve(inliers.size());
    in_bvs2.reserve(inliers.size());
    for (const auto inlier : inliers) {
      // for (int inlier = 0; inlier < n_matches; inlier++) {
      in_bvs1.push_back(bvs1[inlier]);
      in_bvs2.push_back(bvs2[inlier]);
    }

    opengv::translation_t translation = pnec::common::TranslationFromM(
        pnec::common::ComposeM(in_bvs1, in_bvs2, optimizedModel.rotation));
    solution = Sophus::SE3d(optimizedModel.rotation, translation);
  } else {
    opengv::rotation_t rotation = opengv::relative_pose::eigensolver(adapter);
    opengv::translation_t translation = pnec::common::TranslationFromM(
        pnec::common::ComposeM(bvs1, bvs2, rotation));
    inliers.clear();
    solution = Sophus::SE3d(rotation, translation);
  }
  return solution;
}

Sophus::SE3d PNEC::WeightedEigensolver(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covariances,
    const Sophus::SE3d &initial_pose) {
  Sophus::SE3d rel_pose = initial_pose;
  for (size_t iteration = 0; iteration < options_.weighted_iterations_ - 1;
       iteration++) {
    std::vector<double> weights;
    weights.reserve(projected_covariances.size());
    for (size_t i = 0; i < projected_covariances.size(); i++) {
      double weight = pnec::common::Weight(
          bvs1[i], bvs2[i], rel_pose.translation(), rel_pose.rotationMatrix(),
          projected_covariances[i], options_.regularization_, false);
      weights.push_back(weight);
    }

    opengv::rotation_t rotation = rel_pose.rotationMatrix();
    if (options_.weighted_rotation_update_mode_ ==
        WeightedRotationUpdateMode::ScaledBearing) {
      const std::vector<double> normalized_weights = NormalizeWeights(weights);
      opengv::bearingVectors_t w_bvs2;
      w_bvs2.reserve(bvs2.size());
      for (size_t i = 0; i < bvs2.size(); i++) {
        w_bvs2.push_back(bvs2[i] * std::sqrt(normalized_weights[i]));
      }

      opengv::relative_pose::CentralRelativeAdapter adapter(
          bvs1, w_bvs2, rel_pose.rotationMatrix());

      rotation = opengv::relative_pose::eigensolver(adapter);
    } else if (options_.weighted_rotation_update_mode_ ==
               WeightedRotationUpdateMode::PaperLike) {
      const std::vector<double> normalized_weights = NormalizeWeights(weights);
      rotation = OptimizeWeightedRotationPaperLike(
          bvs1, bvs2, normalized_weights, rel_pose.rotationMatrix(),
          options_.ceres_options_);
    }

    std::vector<Eigen::Matrix3d> Ai, Bi;
    // TODO: Host and Target frame
    Ai.reserve(projected_covariances.size());
    Bi.reserve(projected_covariances.size());
    for (size_t i = 0; i < projected_covariances.size(); i++) {
      Eigen::Matrix3d bv1_skew = pnec::common::SkewFromVector(bvs1[i]);
      Ai.push_back((bv1_skew * rotation * bvs2[i]) *
                   (bv1_skew * rotation * bvs2[i]).transpose());
      Bi.push_back(bv1_skew * rotation * projected_covariances[i] *
                       rotation.transpose() * bv1_skew.transpose() +
                   Eigen::Matrix3d::Identity() * options_.regularization_);
    }

    std::vector<Eigen::Vector3d> points =
        pnec::optimization::fibonacci_sphere(500);
    Eigen::Vector3d best_point = rel_pose.translation();
    double best_cost = pnec::optimization::obj_fun(best_point, Ai, Bi);
    for (const auto &point : points) {
      double cost = pnec::optimization::obj_fun(point, Ai, Bi);
      if (cost < best_cost) {
        best_cost = cost;
        best_point = point;
      }
    }

    opengv::translation_t translation;
    if (options_.use_scf_) {
      translation = pnec::optimization::scf(Ai, Bi, best_point, 10);
    } else {
      translation = pnec::common::TranslationFromM(pnec::common::ComposeMPNEC(
          bvs1, bvs2, projected_covariances,
          Sophus::SE3d(rotation, rel_pose.translation()),
          options_.regularization_));
    }

    rel_pose = Sophus::SE3d(rotation, translation);
  }
  return rel_pose;
}

Sophus::SE3d
PNEC::CeresSolver(const opengv::bearingVectors_t &bvs1,
                  const opengv::bearingVectors_t &bvs2,
                  const std::vector<Eigen::Matrix3d> &projected_covariances,
                  const Sophus::SE3d &initial_pose) {
  pnec::optimization::PNECCeres optimizer;
  optimizer.InitValues(Eigen::Quaterniond(initial_pose.rotationMatrix()),
                       initial_pose.translation());
  std::vector<Eigen::Vector3d> bvs_1;
  std::vector<Eigen::Vector3d> bvs_2;
  for (const auto &bv : bvs1) {
    bvs_1.push_back(bv);
  }
  for (const auto &bv : bvs2) {
    bvs_2.push_back(bv);
  }
  optimizer.Optimize(bvs_1, bvs_2, projected_covariances,
                     options_.regularization_, options_.noise_frame_);

  return optimizer.Result();
}

Sophus::SE3d
PNEC::CeresSolver(const opengv::bearingVectors_t &bvs1,
                  const opengv::bearingVectors_t &bvs2,
                  const std::vector<Eigen::Matrix3d> &host_covariances,
                  const std::vector<Eigen::Matrix3d> &target_covariances,
                  const Sophus::SE3d &initial_pose) {
  pnec::optimization::PNECCeres optimizer;
  optimizer.InitValues(Eigen::Quaterniond(initial_pose.rotationMatrix()),
                       initial_pose.translation());
  std::vector<Eigen::Vector3d> bvs_1;
  std::vector<Eigen::Vector3d> bvs_2;
  for (const auto &bv : bvs1) {
    bvs_1.push_back(bv);
  }
  for (const auto &bv : bvs2) {
    bvs_2.push_back(bv);
  }
  optimizer.Optimize(bvs_1, bvs_2, host_covariances, target_covariances,
                     options_.regularization_);

  return optimizer.Result();
}

Sophus::SE3d
PNEC::CeresSolverFull(const opengv::bearingVectors_t &bvs1,
                      const opengv::bearingVectors_t &bvs2,
                      const std::vector<Eigen::Matrix3d> &projected_covariances,
                      double regularization, const Sophus::SE3d &initial_pose) {
  pnec::optimization::PNECCeres optimizer;
  optimizer.InitValues(Eigen::Quaterniond(initial_pose.rotationMatrix()),
                       initial_pose.translation());
  std::vector<Eigen::Vector3d> bvs_1;
  std::vector<Eigen::Vector3d> bvs_2;
  for (const auto &bv : bvs1) {
    bvs_1.push_back(bv);
  }
  for (const auto &bv : bvs2) {
    bvs_2.push_back(bv);
  }

  optimizer.Optimize(bvs_1, bvs_2, projected_covariances, regularization,
                     options_.noise_frame_);

  return optimizer.Result();
}

Sophus::SE3d PNEC::NECCeresSolver(const opengv::bearingVectors_t &bvs1,
                                  const opengv::bearingVectors_t &bvs2,
                                  const Sophus::SE3d &initial_pose) {
  pnec::optimization::NECCeres optimizer;
  optimizer.InitValues(Eigen::Quaterniond(initial_pose.rotationMatrix()),
                       initial_pose.translation());
  std::vector<Eigen::Vector3d> bvs_1;
  std::vector<Eigen::Vector3d> bvs_2;
  for (const auto &bv : bvs1) {
    bvs_1.push_back(bv);
  }
  for (const auto &bv : bvs2) {
    bvs_2.push_back(bv);
  }
  optimizer.Optimize(bvs_1, bvs_2);

  return optimizer.Result();
}

} // namespace rel_pose_estimation
} // namespace pnec
