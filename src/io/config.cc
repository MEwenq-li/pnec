#include "config.h"

#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <iostream>

#include <basalt/calibration/calibration.hpp>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/vio_config.h>
#include <opencv2/core/core.hpp>

#include <camera.h>

namespace pnec {
namespace input {

// TODO: ceres options
basalt::Calibration<double> LoadTrackingCalib(const std::string &calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  basalt::Calibration<double> tracking_calib;
  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(tracking_calib);
    BOOST_LOG_TRIVIAL(info) << "Loaded camera with "
                            << tracking_calib.intrinsics.size() << " cameras";
  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
  return tracking_calib;
}

basalt::VioConfig LoadTrackingConfig(const std::string &config_path) {
  basalt::VioConfig tracking_config;
  tracking_config.load(config_path);
  return tracking_config;
}

pnec::rel_pose_estimation::Options
LoadPNECConfig(const std::string &config_path) {
  // check settings file
  cv::FileStorage settings(config_path.c_str(), cv::FileStorage::READ);
  if (!settings.isOpened()) {
    std::cerr << "Failed to open settings file: " << config_path << std::endl;
    std::exit(-1);
  }

  std::string noise_frame_str = settings["PNEC.noiseFrame"];
  pnec::common::NoiseFrame noise_frame = pnec::common::Target;
  if (noise_frame_str == "Host") {
    noise_frame = pnec::common::Host;
  }
  if (noise_frame_str == "Target") {
    noise_frame = pnec::common::Target;
  }
  if (noise_frame_str == "Both") {
    noise_frame = pnec::common::Both;
  }
  std::string covariance_mode_str = "Original";
  cv::FileNode covariance_mode_node = settings["PNEC.covarianceMode"];
  if (!covariance_mode_node.empty()) {
    covariance_mode_node >> covariance_mode_str;
  }
  pnec::rel_pose_estimation::CovarianceExperimentMode covariance_mode =
      pnec::rel_pose_estimation::CovarianceExperimentMode::Original;
  if (covariance_mode_str == "Isotropic") {
    covariance_mode =
        pnec::rel_pose_estimation::CovarianceExperimentMode::Isotropic;
  } else if (covariance_mode_str == "Diagonal") {
    covariance_mode =
        pnec::rel_pose_estimation::CovarianceExperimentMode::Diagonal;
  } else if (covariance_mode_str == "Normalized") {
    covariance_mode =
        pnec::rel_pose_estimation::CovarianceExperimentMode::Normalized;
  }
  std::string weighted_rotation_update_mode_str = "ScaledBearing";
  cv::FileNode weighted_rotation_update_mode_node =
      settings["PNEC.weightedRotationUpdateMode"];
  if (!weighted_rotation_update_mode_node.empty()) {
    weighted_rotation_update_mode_node >>
        weighted_rotation_update_mode_str;
  }
  pnec::rel_pose_estimation::WeightedRotationUpdateMode
      weighted_rotation_update_mode =
          pnec::rel_pose_estimation::WeightedRotationUpdateMode::
              ScaledBearing;
  if (weighted_rotation_update_mode_str == "FreezeRotation") {
    weighted_rotation_update_mode =
        pnec::rel_pose_estimation::WeightedRotationUpdateMode::
            FreezeRotation;
  } else if (weighted_rotation_update_mode_str == "PaperLike") {
    weighted_rotation_update_mode =
        pnec::rel_pose_estimation::WeightedRotationUpdateMode::PaperLike;
  }
  std::string ceres_init_mode_str = "Weighted";
  cv::FileNode ceres_init_mode_node = settings["PNEC.ceresInitMode"];
  if (!ceres_init_mode_node.empty()) {
    ceres_init_mode_node >> ceres_init_mode_str;
  }
  pnec::rel_pose_estimation::CeresInitMode ceres_init_mode =
      pnec::rel_pose_estimation::CeresInitMode::Weighted;
  if (ceres_init_mode_str == "NEC") {
    ceres_init_mode = pnec::rel_pose_estimation::CeresInitMode::NEC;
  } else if (ceres_init_mode_str == "NECCeres") {
    ceres_init_mode = pnec::rel_pose_estimation::CeresInitMode::NECCeres;
  } else if (ceres_init_mode_str == "Initial") {
    ceres_init_mode = pnec::rel_pose_estimation::CeresInitMode::Initial;
  }
  const int nec_int = settings["PNEC.NEC"];
  const int scf_int = settings["PNEC.SCF"];
  const int ceres_int = settings["PNEC.ceres"];
  const int ransac_int = settings["PNEC.ransac"];

  pnec::rel_pose_estimation::Options options;
  options.use_nec_ = (nec_int == 0) ? false : true;

  options.noise_frame_ = noise_frame;
  options.regularization_ = settings["PNEC.regularization"];
  options.covariance_mode_ = covariance_mode;
  cv::FileNode isotropic_covariance_value =
      settings["PNEC.isotropicCovarianceValue"];
  if (!isotropic_covariance_value.empty()) {
    options.isotropic_covariance_value_ =
        static_cast<double>(isotropic_covariance_value);
  }
  cv::FileNode normalized_covariance_trace =
      settings["PNEC.normalizedCovarianceTrace"];
  if (!normalized_covariance_trace.empty()) {
    options.normalized_covariance_trace_ =
        static_cast<double>(normalized_covariance_trace);
  }
  cv::FileNode dump_covariance_stats =
      settings["PNEC.dumpCovarianceStats"];
  if (!dump_covariance_stats.empty()) {
    options.dump_covariance_stats_ =
        static_cast<int>(dump_covariance_stats) != 0;
  }

  options.weighted_iterations_ =
      static_cast<int>(settings["PNEC.weightedIterations"]);
  options.use_scf_ = (scf_int == 0) ? false : true;
  options.weighted_rotation_update_mode_ = weighted_rotation_update_mode;
  options.ceres_init_mode_ = ceres_init_mode;

  options.use_ceres_ = (ceres_int == 0) ? false : true;
  options.ceres_options_ = ceres::Solver::Options();

  options.use_ransac_ = (ransac_int == 0) ? false : true;
  cv::FileNode ransac_threshold = settings["PNEC.RANSACThreshold"];
  if (!ransac_threshold.empty()) {
    options.ransac_threshold_ = static_cast<double>(ransac_threshold);
  }
  options.max_ransac_iterations_ = settings["PNEC.maxRANSACIterations"];
  options.ransac_sample_size_ = settings["PNEC.RANSACSampleSize"];

  return options;
}

pnec::CameraParameters LoadCameraConfig(const std::string &config_path) {
  // check settings file
  cv::FileStorage settings(config_path.c_str(), cv::FileStorage::READ);
  if (!settings.isOpened()) {
    std::cerr << "Failed to open settings file: " << config_path << std::endl;
    std::exit(-1);
  }

  cv::Matx33d K = cv::Matx33d::eye();

  const double fx = settings["Camera.fx"];
  const double fy = settings["Camera.fy"];
  const double cx = settings["Camera.cx"];
  const double cy = settings["Camera.cy"];

  K(0, 0) = fx;
  K(1, 1) = fy;
  K(0, 2) = cx;
  K(1, 2) = cy;

  cv::Vec4d dist_coef;

  dist_coef(0) = settings["Camera.k1"];
  dist_coef(1) = settings["Camera.k2"];
  dist_coef(2) = settings["Camera.p1"];
  dist_coef(3) = settings["Camera.p2"];

  return pnec::CameraParameters(K, dist_coef);
}
} // namespace input
} // namespace pnec
