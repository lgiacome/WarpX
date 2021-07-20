macro(find_amrex)
    if(WarpX_amrex_src)
        message(STATUS "Compiling local AMReX ...")
        message(STATUS "AMReX source path: ${WarpX_amrex_src}")
    elseif(WarpX_amrex_internal)
        message(STATUS "Downloading AMReX ...")
        message(STATUS "AMReX repository: ${WarpX_amrex_repo} (${WarpX_amrex_branch})")
        include(FetchContent)
    endif()
    if(WarpX_amrex_internal OR WarpX_amrex_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # see https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#customization-options
        if(WarpX_ASCENT)
            set(AMReX_ASCENT ON CACHE INTERNAL "")
            set(AMReX_CONDUIT ON CACHE INTERNAL "")
        endif()

        if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(AMReX_ASSERTIONS ON CACHE BOOL "")
            # note: floating-point exceptions can slow down debug runs a lot
            set(AMReX_FPE ON CACHE BOOL "")
        else()
            set(AMReX_ASSERTIONS OFF CACHE BOOL "")
            set(AMReX_FPE OFF CACHE BOOL "")
        endif()

        if(WarpX_COMPUTE STREQUAL OMP)
            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
            set(AMReX_OMP          ON     CACHE INTERNAL "")
        elseif(WarpX_COMPUTE STREQUAL NOACC)
            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
            set(AMReX_OMP          OFF    CACHE INTERNAL "")
        else()
            set(AMReX_GPU_BACKEND  "${WarpX_COMPUTE}" CACHE INTERNAL "")
            set(AMReX_OMP          OFF    CACHE INTERNAL "")
        endif()

        if(WarpX_EB)
            set(AMReX_EB ON CACHE INTERNAL "")
        else()
            set(AMReX_EB OFF CACHE INTERNAL "")
        endif()

        if(WarpX_MPI)
            set(AMReX_MPI ON CACHE INTERNAL "")
            if(WarpX_MPI_THREAD_MULTIPLE)
                set(AMReX_MPI_THREAD_MULTIPLE ON CACHE INTERNAL "")
            else()
                set(AMReX_MPI_THREAD_MULTIPLE OFF CACHE INTERNAL "")
            endif()
        else()
            set(AMReX_MPI OFF CACHE INTERNAL "")
        endif()

        if(WarpX_PRECISION STREQUAL "DOUBLE")
            set(AMReX_PRECISION "DOUBLE" CACHE INTERNAL "")
            set(AMReX_PARTICLES_PRECISION "DOUBLE" CACHE INTERNAL "")
        else()
            set(AMReX_PRECISION "SINGLE" CACHE INTERNAL "")
            set(AMReX_PARTICLES_PRECISION "SINGLE" CACHE INTERNAL "")
        endif()

        set(AMReX_INSTALL ${BUILD_SHARED_LIBS} CACHE INTERNAL "")
        set(AMReX_AMRLEVEL OFF CACHE INTERNAL "")
        set(AMReX_ENABLE_TESTS OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN OFF CACHE INTERNAL "")
        set(AMReX_FORTRAN_INTERFACES OFF CACHE INTERNAL "")
        set(AMReX_BUILD_TUTORIALS OFF CACHE INTERNAL "")
        set(AMReX_PARTICLES ON CACHE INTERNAL "")
        set(AMReX_TINY_PROFILE ON CACHE BOOL "")

        # AMReX_SENSEI
        # shared libs, i.e. for Python bindings, need relocatable code
        if(WarpX_LIB)
            set(AMReX_PIC ON CACHE INTERNAL "")
        endif()

        # IPO/LTO
        if(WarpX_IPO)
            set(AMReX_IPO ON CACHE INTERNAL "")
        endif()

        if(WarpX_DIMS STREQUAL RZ)
            set(AMReX_SPACEDIM 2 CACHE INTERNAL "")
        else()
            set(AMReX_SPACEDIM ${WarpX_DIMS} CACHE INTERNAL "")
        endif()

        if(WarpX_amrex_src)
            list(APPEND CMAKE_MODULE_PATH "${WarpX_amrex_src}/Tools/CMake")
            if(WarpX_COMPUTE STREQUAL CUDA)
                enable_language(CUDA)
                include(AMReX_SetupCUDA)
            endif()
            add_subdirectory(${WarpX_amrex_src} _deps/localamrex-build/)
        else()
            FetchContent_Declare(fetchedamrex
                GIT_REPOSITORY ${WarpX_amrex_repo}
                GIT_TAG        ${WarpX_amrex_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_GetProperties(fetchedamrex)

            if(NOT fetchedamrex_POPULATED)
                FetchContent_Populate(fetchedamrex)
                list(APPEND CMAKE_MODULE_PATH "${fetchedamrex_SOURCE_DIR}/Tools/CMake")
                if(WarpX_COMPUTE STREQUAL CUDA)
                    enable_language(CUDA)
                    include(AMReX_SetupCUDA)
                endif()
                add_subdirectory(${fetchedamrex_SOURCE_DIR} ${fetchedamrex_BINARY_DIR})
            endif()

            # advanced fetch options
            mark_as_advanced(FETCHCONTENT_BASE_DIR)
            mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_QUIET)
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDAMREX)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDAMREX)
        endif()

        # AMReX options not relevant to WarpX users
        mark_as_advanced(AMREX_BUILD_DATETIME)
        mark_as_advanced(AMReX_ENABLE_TESTS)
        mark_as_advanced(AMReX_SPACEDIM)
        mark_as_advanced(AMReX_ASSERTIONS)
        mark_as_advanced(AMReX_AMRDATA)
        mark_as_advanced(AMReX_BASE_PROFILE) # mutually exclusive to tiny profile
        mark_as_advanced(AMReX_CONDUIT)
        mark_as_advanced(AMReX_CUDA)
        mark_as_advanced(AMReX_PARTICLES)
        mark_as_advanced(AMReX_PARTICLES_PRECISION)
        mark_as_advanced(AMReX_DPCPP)
        mark_as_advanced(AMReX_EB)
        mark_as_advanced(AMReX_FPE)
        mark_as_advanced(AMReX_FORTRAN)
        mark_as_advanced(AMReX_FORTRAN_INTERFACES)
        mark_as_advanced(AMReX_HDF5)  # we do HDF5 I/O (and more) via openPMD-api
        mark_as_advanced(AMReX_HIP)
        mark_as_advanced(AMReX_HYPRE)
        mark_as_advanced(AMReX_LINEAR_SOLVERS)
        mark_as_advanced(AMReX_MEM_PROFILE)
        mark_as_advanced(AMReX_MPI)
        mark_as_advanced(AMReX_MPI_THREAD_MULTIPLE)
        mark_as_advanced(AMReX_OMP)
        mark_as_advanced(AMReX_PETSC)
        mark_as_advanced(AMReX_PIC)
        mark_as_advanced(AMReX_SENSEI)
        mark_as_advanced(AMReX_TINY_PROFILE)
        mark_as_advanced(AMReX_TP_PROFILE)
        mark_as_advanced(USE_XSDK_DEFAULTS)

        message(STATUS "AMReX: Using version '${AMREX_PKG_VERSION}' (${AMREX_GIT_VERSION})")
    else()
        message(STATUS "Searching for pre-installed AMReX ...")
        # https://amrex-codes.github.io/amrex/docs_html/BuildingAMReX.html#importing-amrex-into-your-cmake-project
        if(WarpX_ASCENT)
            set(COMPONENT_ASCENT AMReX_ASCENT AMReX_CONDUIT)
        else()
            set(COMPONENT_ASCENT)
        endif()
        if(WarpX_DIMS STREQUAL RZ)
            set(COMPONENT_DIM 2D)
        else()
            set(COMPONENT_DIM ${WarpX_DIMS}D)
        endif()
        if(WarpX_EB)
            set(COMPONENT_EB EB)
        else()
            set(COMPONENT_EB)
        endif()
        if(WarpX_LIB)
            set(COMPONENT_PIC PIC)
        else()
            set(COMPONENT_PIC)
        endif()
        set(COMPONENT_PRECISION ${WarpX_PRECISION} P${WarpX_PRECISION})

        find_package(AMReX 21.05 CONFIG REQUIRED COMPONENTS ${COMPONENT_ASCENT} ${COMPONENT_DIM} ${COMPONENT_EB} PARTICLES ${COMPONENT_PIC} ${COMPONENT_PRECISION} TINYP LSOLVERS)
        message(STATUS "AMReX: Found version '${AMReX_VERSION}'")
    endif()
endmacro()

# local source-tree
set(WarpX_amrex_src ""
    CACHE PATH
    "Local path to AMReX source directory (preferred if set)")

# Git fetcher
set(WarpX_amrex_repo "https://github.com/AMReX-Codes/amrex.git"
    CACHE STRING
    "Repository URI to pull and build AMReX from if(WarpX_amrex_internal)")
set(WarpX_amrex_branch "646d5f63445ff46e4df23cbdb1fbe662ca1708fb"
    CACHE STRING
    "Repository branch for WarpX_amrex_repo if(WarpX_amrex_internal)")

find_amrex()
