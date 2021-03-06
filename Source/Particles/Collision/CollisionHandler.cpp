/* Copyright 2020 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CollisionHandler.H"

#include "BinaryCollision.H"
#include "PairWiseCoulombCollisionFunc.H"
#include "BackgroundMCCCollision.H"

#include <AMReX_ParmParse.H>

#include <vector>

CollisionHandler::CollisionHandler()
{

    // Read in collision input
    amrex::ParmParse pp_collisions("collisions");
    pp_collisions.queryarr("collision_names", collision_names);

    // Create instances based on the collision type
    auto const ncollisions = collision_names.size();
    collision_types.resize(ncollisions);
    allcollisions.resize(ncollisions);
    for (int i = 0; i < static_cast<int>(ncollisions); ++i) {
        amrex::ParmParse pp_collision_name(collision_names[i]);

        // For legacy, pairwisecoulomb is the default
        std::string type = "pairwisecoulomb";

        pp_collision_name.query("type", type);
        collision_types[i] = type;

        if (type == "pairwisecoulomb") {
            allcollisions[i] =
               std::make_unique<BinaryCollision<PairWiseCoulombCollisionFunc>>(collision_names[i]);
        }
        else if (type == "background_mcc") {
            allcollisions[i] = std::make_unique<BackgroundMCCCollision>(collision_names[i]);
        }
        else{
            amrex::Abort("Unknown collision type.");
        }

    }

}

/** Perform all collisions
 *
 * @param cur_time Current time
 * @param mypc MultiParticleContainer calling this method
 *
 */
void CollisionHandler::doCollisions ( amrex::Real cur_time, MultiParticleContainer* mypc)
{

    for (auto& collision : allcollisions) {
        collision->doCollisions(cur_time, mypc);
    }

}
