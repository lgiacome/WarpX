#include "WarpX.H"

#include <AMReX_Config.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EB2.H>
#   include <AMReX_ParmParse.H>
#endif

#ifdef AMREX_USE_EB
namespace {
    class ParserIF
        : public amrex::GPUable
    {
    public:
        ParserIF (const ParserWrapper<3>& a_parser)
            : m_parser(a_parser.getParser())
            {}

        ParserIF (const ParserIF& rhs) noexcept = default;
        ParserIF (ParserIF&& rhs) noexcept = default;
        ParserIF& operator= (const ParserIF& rhs) = delete;
        ParserIF& operator= (ParserIF&& rhs) = delete;

        AMREX_GPU_HOST_DEVICE inline
        amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                             amrex::Real z)) const noexcept {
#if (AMREX_SPACEDIM == 2)
            return m_parser(x,amrex::Real(0.0),y);
#else
            return m_parser(x,y,z);
#endif
        }

        inline amrex::Real operator() (const amrex::RealArray& p) const noexcept {
            return this->operator()(AMREX_D_DECL(p[0],p[1],p[2]));
        }

    private:
        HostDeviceParser<3> m_parser;
    };
}
#endif

void
WarpX::InitEB (int lev)
{
#ifdef AMREX_USE_EB
    BL_PROFILE("InitEB");

    amrex::ParmParse pp_warpx("warpx");
    std::string impf;
    pp_warpx.query("eb_implicit_function", impf);
    if (! impf.empty()) {
        WarpXParser wp = makeParser(impf, {"x", "y", "z"});
        m_eb_if_parser = std::make_unique<ParserWrapper<3> >(wp);
        ParserIF pif(*m_eb_if_parser);
        auto gshop = amrex::EB2::makeShop(pif);
        amrex::EB2::Build(gshop, Geom(lev), maxLevel(), lev);
    } else {
        amrex::ParmParse pp_eb2("eb2");
        if (!pp_eb2.contains("geom_type")) {
            std::string geom_type = "all_regular";
            pp_eb2.add("geom_type", geom_type); // use all_regular by default
        }
        amrex::EB2::Build(Geom(lev), maxLevel(), lev);
    }

#endif
}

/**
 * \brief Compute the length of the mesh edges. Here the length is a value in [0, 1].
 *        An edge of length 0 is fully covered.
 */
void
WarpX::ComputeEdgeLengths (std::array< std::unique_ptr<amrex::MultiFab>, 3 >& edge_lengths,
                           int lev, bool flag_cp) {
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeEdgeLengths");

    //This variable is equal to lev if this is a fine patch or
    // equal to lev -1 if this is a coarse patch
    int lev_loc = lev;
    if(flag_cp and lev > 0){
        lev_loc = lev -1;
    }
    auto const eb_fact = fieldEBFactory(lev_loc);

    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &edge_centroid = eb_fact.getEdgeCent();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
        for (amrex::MFIter mfi(*edge_lengths[idim]); mfi.isValid(); ++mfi){
            amrex::Box box = mfi.validbox();
            amrex::FabType fab_type = flags[mfi].getType(box);
            auto const &edge_lengths_dim = edge_lengths[idim]->array(mfi);
            if (fab_type == amrex::FabType::regular) {
                amrex::ParallelFor(box, [=](int i, int j, int k) {
                    edge_lengths_dim(i, j, k) = 1.;
                });
            } else if (fab_type == amrex::FabType::covered) {
                amrex::ParallelFor(box, [=](int i, int j, int k) {
                    edge_lengths_dim(i, j, k) = 0.;
                });
            } else {
                auto const &edge_cent = edge_centroid[idim]->const_array(mfi);
                amrex::ParallelFor(box, [=](int i, int j, int k) {
                    if (edge_cent(i, j, k) == amrex::Real(-1.0)) {
                        // This edge is all covered
                        edge_lengths_dim(i, j, k) = 0.;
                    } else if (edge_cent(i, j, k) == amrex::Real(1.0)) {
                        // This edge is all open
                        edge_lengths_dim(i, j, k) = 1.;
                    } else {
                        // This edge is cut.
                        edge_lengths_dim(i, j, k) = 1 - amrex::Math::abs(amrex::Real(2.0)
                                                                        * edge_cent(i, j, k));
                    }
                });
            }
        }
    }
#endif
}

/**
 * \brief Compute the ara of the mesh faces. Here the area is a value in [0, 1].
 *        An edge of area 0 is fully covered.
 */
void
WarpX::ComputeFaceAreas (std::array< std::unique_ptr<amrex::MultiFab>, 3 >& face_areas,
                         int lev, bool flag_cp) {
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeFaceAreas");

    //This variable is equal to lev if this is a fine patch or
    // equal to lev -1 if this is a coarse patch
    int lev_loc = lev;
    if(flag_cp and lev > 0){
        lev_loc = lev -1;
    }

    auto const eb_fact = fieldEBFactory(lev_loc);
    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &area_frac = eb_fact.getAreaFrac();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (amrex::MFIter mfi(*face_areas[idim]); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.validbox();
            amrex::FabType fab_type = flags[mfi].getType(box);
            auto const &face_areas_dim = face_areas[idim]->array(mfi);
            if (fab_type == amrex::FabType::regular) {
                // every cell in box is all regular
                amrex::ParallelFor(box, [=](int i, int j, int k) {
                    face_areas_dim(i, j, k) = amrex::Real(1.);
                });
            } else if (fab_type == amrex::FabType::covered) {
                // every cell in box is all covered
                amrex::ParallelFor(box, [=](int i, int j, int k) {
                    face_areas_dim(i, j, k) = amrex::Real(0.);
                });
            } else {
                auto const &face = area_frac[idim]->const_array(mfi);
                amrex::ParallelFor(box, [=](int i, int j, int k) {
                    face_areas_dim(i, j, k) = face(i, j, k);
                });
            }
        }
    }
#endif
}

/**
 * \brief Scale the edges lengths by the mesh width to obtain the real lengths.
 */
void
WarpX::ScaleEdges (std::array< std::unique_ptr<amrex::MultiFab>, 3 >& edge_lengths,
                   int lev, bool flag_cp) {
#ifdef AMREX_USE_EB
    BL_PROFILE("ScaleEdges");

    //This variable is equal to lev if this is a fine patch or
    // equal to lev -1 if this is a coarse patch
    int lev_loc = lev;
    if(flag_cp and lev > 0){
        lev_loc = lev -1;
    }

    auto const &cell_size = CellSize(lev_loc);
    auto const eb_fact = fieldEBFactory(lev_loc);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (amrex::MFIter mfi(*edge_lengths[idim]); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.validbox();
            auto const &edge_lengths_dim = edge_lengths[idim]->array(mfi);
            amrex::ParallelFor(box, [=](int i, int j, int k) {
                edge_lengths_dim(i, j, k) *= cell_size[idim];
            });
        }
    }
#endif
}

/**
 * \brief Scale the edges areas by the mesh width to obtain the real areas.
 */
void
WarpX::ScaleAreas(std::array< std::unique_ptr<amrex::MultiFab>, 3 >& face_areas,
                  int lev, bool flag_cp) {
#ifdef AMREX_USE_EB
    BL_PROFILE("ScaleAreas");

    //This variable is equal to lev if this is a fine patch or
    // equal to lev -1 if this is a coarse patch
    int lev_loc = lev;
    if(flag_cp and lev > 0){
        lev_loc = lev -1;
    }

    auto const& cell_size = CellSize(lev_loc);
    amrex::Real full_area;

    auto const eb_fact = fieldEBFactory(lev_loc);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (amrex::MFIter mfi(*face_areas[idim]); mfi.isValid(); ++mfi) {
        amrex::Box box = mfi.validbox();
            if (idim == 0) {
                full_area = cell_size[1]*cell_size[2];
            } else if (idim == 1) {
                full_area = cell_size[0]*cell_size[2];
            } else {
                full_area = cell_size[0]*cell_size[1];
            }
            auto const &face_areas_dim = face_areas[idim]->array(mfi);
            amrex::ParallelFor(box, [=](int i, int j, int k) {
                face_areas_dim(i, j, k) *= full_area;
            });
        }
    }
#endif
}
