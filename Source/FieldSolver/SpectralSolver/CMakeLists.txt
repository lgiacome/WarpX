target_sources(WarpX
  PRIVATE
    SpectralFieldData.cpp
    SpectralKSpace.cpp
    SpectralSolver.cpp
)

if(WarpX_COMPUTE STREQUAL CUDA)
    target_sources(WarpX PRIVATE WrapCuFFT.cpp)
elseif(WarpX_COMPUTE STREQUAL HIP)
    target_sources(WarpX PRIVATE WrapRocFFT.cpp)
else()
    target_sources(WarpX PRIVATE WrapFFTW.cpp)
endif()

if(WarpX_DIMS STREQUAL RZ)
    target_sources(WarpX
      PRIVATE
        SpectralSolverRZ.cpp
        SpectralFieldDataRZ.cpp
        SpectralKSpaceRZ.cpp
        SpectralBinomialFilter.cpp
    )
    add_subdirectory(SpectralHankelTransform)
endif()

add_subdirectory(SpectralAlgorithms)
