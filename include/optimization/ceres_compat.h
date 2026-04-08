#ifndef OPTIMIZATION_CERES_COMPAT_H_
#define OPTIMIZATION_CERES_COMPAT_H_

#include <ceres/ceres.h>
#include <ceres/version.h>

namespace pnec {
namespace optimization {

inline void SetEigenQuaternionParameterization(ceres::Problem& problem,
                                               double* quaternion_coeffs) {
#if CERES_VERSION_MAJOR >= 2
  ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;
  problem.SetManifold(quaternion_coeffs, quaternion_manifold);
#else
  ceres::LocalParameterization* quaternion_parameterization =
      new ceres::EigenQuaternionParameterization;
  problem.SetParameterization(quaternion_coeffs, quaternion_parameterization);
#endif
}

} // namespace optimization
} // namespace pnec

#endif // OPTIMIZATION_CERES_COMPAT_H_
