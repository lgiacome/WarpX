#include "WarpX.H"
#include <math.h>
#include <AMReX_Config.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EB2.H>
#   include <AMReX_ParmParse.H>
#endif


void
WarpX::InitEB ()
{
#ifdef AMREX_USE_EB
    BL_PROFILE("InitEB");

    amrex::ParmParse pp_eb2("eb2");
    if (!pp_eb2.contains("geom_type")) {
        pp_eb2.add("geom_type", "all_regular"); // use all_regular by default
    }
    amrex::EB2::Build(Geom(maxLevel()), maxLevel(), maxLevel());

#endif
}

void
WarpX::ComputeEdgeLengths () {
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeEdgeLengths");

    auto const eb_fact = fieldEBFactory(maxLevel());

    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &edge_centroid = eb_fact.getEdgeCent();
    for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi){
        amrex::Box const &box = mfi.validbox();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            auto const &edge = edge_centroid[idim]->const_array(mfi);
            auto const &edge_lengths_dim = m_edge_lengths[maxLevel()][idim]->array(mfi);
            amrex::LoopOnCpu(amrex::convert(box, amrex::Box(edge).ixType()),
                            [=](int i, int j, int k) {
                if (edge(i, j, k) == amrex::Real(-1.0)) {
                    // This edge is all covered
                    edge_lengths_dim(i, j, k) = 0.;
                } else if (edge(i, j, k) == amrex::Real(1.0)) {
                    // This edge is all open
                    edge_lengths_dim(i, j, k) = 1.;
                } else {
                    // This edge is cut.
                    amrex::Real edge_cent = edge(i, j, k); // edge centroid: (-0.5,0.5)
                    /*
                    if (edge_cent < amrex::Real(0.0)) {
                        // The right side is covered.
                        edge_lengths_dim(i, j, k) = amrex::Real(2.0) * edge_cent + amrex::Real(1); // (0, 1)
                    } else {
                        // The left side is covered
                        edge_lengths_dim(i, j, k) = (-amrex::Real(2.0) * edge_cent + amrex::Real(1)); // (0, 1)
                        //edge_lengths_dim(i, j, k) = 0.5 - amrex::Real(2.0) * edge_cent - amrex::Real(0.5); // (0, 1)
                    }
                    */
                    edge_lengths_dim(i, j, k) = -abs(amrex::Real(2.0)*edge_cent) + amrex::Real(1); // - std::copysign(amrex::Real(0.5), edge_cent);
                    //edge_lengths_dim(i, j, k) = amrex::Real(2.0)*edge_cent - std::copysign(amrex::Real(0.5), edge_cent);
                }
            });
        }
    }
#endif
}


void
WarpX::ComputeFaceAreas () {
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeFaceAreas");

    auto const eb_fact = fieldEBFactory(maxLevel());
    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &area_frac = eb_fact.getAreaFrac();

    for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi) {
        amrex::Box const &box = mfi.validbox();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
              auto const &face = area_frac[idim]->const_array(mfi);
              auto const &face_areas_dim = m_face_areas[maxLevel()][idim]->array(mfi);
              amrex::LoopOnCpu(amrex::convert(box, amrex::Box(face).ixType()),
                               [=](int i, int j, int k) {
                  face_areas_dim(i, j, k) = face(i, j, k);
              });
        }
    }
#endif
}

void
WarpX::ScaleEdges () {
#ifdef AMREX_USE_EB
    auto const& cell_size = CellSize(maxLevel());
    auto const eb_fact = fieldEBFactory(maxLevel());
    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &edge_centroid = eb_fact.getEdgeCent();

    for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi) {
        amrex::Box const &box = mfi.validbox();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            auto const &edge = edge_centroid[idim]->const_array(mfi);
            auto const &edge_lengths_dim = m_edge_lengths[maxLevel()][idim]->array(mfi);
            amrex::LoopOnCpu(amrex::convert(box, amrex::Box(edge).ixType()),
                             [=](int i, int j, int k) {
                                 edge_lengths_dim(i, j, k) *= cell_size[idim];
              });
        }
    }
#endif
}


void
WarpX::ScaleAreas() {
#ifdef AMREX_USE_EB
    auto const& cell_size = CellSize(maxLevel());
    amrex::Real full_area;

    auto const eb_fact = fieldEBFactory(maxLevel());
    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &area_frac = eb_fact.getAreaFrac();

    for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi) {
        amrex::Box const &box = mfi.validbox();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (idim == 0) {
                full_area = cell_size[1]*cell_size[2];
            } else if (idim == 1) {
                full_area = cell_size[0]*cell_size[2];
            } else {
                full_area = cell_size[0]*cell_size[1];
            }
            auto const &face = area_frac[idim]->const_array(mfi);
            auto const &face_areas_dim = m_face_areas[maxLevel()][idim]->array(mfi);
            auto const &areas_red_dim = m_area_red[maxLevel()][idim]->array(mfi);
            amrex::LoopOnCpu(amrex::convert(box, amrex::Box(face).ixType()),
                             [=](int i, int j, int k) {
                                face_areas_dim(i, j, k) *= full_area;
                                areas_red_dim(i, j, k) = face_areas_dim(i, j, k);
            });

        }
    }
#endif
}

void
WarpX::MarkCells(){

    ComputeAreaStab();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (amrex::MFIter mfi(*Bfield_fp[maxLevel()][idim]); mfi.isValid(); ++mfi) {
            amrex::Box const &box = mfi.validbox();

            auto const &S = m_face_areas[maxLevel()][idim]->array(mfi);
            auto const &flag_int_face = m_flag_int_face[maxLevel()][idim]->array(mfi);
            auto const &flag_unst_face = m_flag_unst_face[maxLevel()][idim]->array(mfi);
            auto const &flag_ext_face = m_flag_ext_face[maxLevel()][idim]->array(mfi);
            auto const &flag_avail_face = m_flag_avail_face[maxLevel()][idim]->array(mfi);
            auto const &area_stab = m_area_stab[maxLevel()][idim]->array(mfi);
            amrex::LoopOnCpu(box,
                             [=](int i, int j, int k) {
                flag_int_face(i, j, k) = int(S(i, j, k) > 0);
                // This face is unstable if it has less area than area_stab
                flag_unst_face(i, j, k) = int((S(i, j, k) < area_stab(i, j, k))
                                            and !isnan(S(i, j, k)) and S(i, j, k) > 0);
                // Does this face need to be extended? This is the same as flag_unst_face here,
                // but it is modified later to keep track o which faces still need to be extended
                flag_ext_face(i, j, k) = flag_unst_face(i, j, k);
                // Is this face available to lend area to other faces?
                // The criterion is that the face has to be interior and not already unstable itself
                flag_avail_face(i, j, k) = int(flag_int_face(i, j, k) and
                                               !flag_unst_face(i, j, k));
            });
        }
    }
}

void
WarpX::ComputeAreaStab() {
    auto const eb_fact = fieldEBFactory(maxLevel());
    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &cell_size = CellSize(maxLevel());
    auto const &area_frac = eb_fact.getAreaFrac();
    amrex::Real dx = cell_size[0];
    amrex::Real dy = cell_size[1];
    amrex::Real dz = cell_size[2];
    for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi) {
        amrex::Box const &box = mfi.validbox();
        auto const &lx = m_edge_lengths[maxLevel()][0]->array(mfi);
        auto const &ly = m_edge_lengths[maxLevel()][1]->array(mfi);
        auto const &lz = m_edge_lengths[maxLevel()][2]->array(mfi);
        auto const &area_stab_x = m_area_stab[maxLevel()][0]->array(mfi);
        auto const &area_stab_y = m_area_stab[maxLevel()][1]->array(mfi);
        auto const &area_stab_z = m_area_stab[maxLevel()][2]->array(mfi);
        auto const &face = area_frac[0]->const_array(mfi);
        amrex::LoopOnCpu(amrex::convert(box, amrex::Box(face).ixType()),
                         [=](int i, int j, int k) {
            area_stab_z(i, j, k) = 0.5 * std::max({lx(i, j, k) * dy, lx(i, j + 1, k) * dy,
                                                     ly(i, j, k) * dx, ly(i + 1, j, k) * dx});
        });
        amrex::LoopOnCpu(amrex::convert(box, amrex::Box(face).ixType()),
                         [=](int i, int j, int k) {
            area_stab_y(i, j, k) = 0.5 * std::max({lx(i, j, k) * dz, lx(i, j, k + 1) * dz,
                                                     lz(i, j, k) * dx, lz(i + 1, j, k) * dx});
        });
        amrex::LoopOnCpu(amrex::convert(box, amrex::Box(face).ixType()),
                         [=](int i, int j, int k) {
            area_stab_x(i, j, k) = 0.5 * std::max({ly(i, j, k) * dz, ly(i, j, k + 1) * dz,
                                                     lz(i, j, k) * dy, lz(i, j + 1, k) * dy});
        });
    }
}