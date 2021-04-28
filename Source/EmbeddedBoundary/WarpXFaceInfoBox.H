//
// Created by Lorenzo Giacomel on 14/04/2021.
//

#ifndef WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXFACEINFOBOX_H
#define WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXFACEINFOBOX_H

struct FaceInfoBox {
    amrex::Gpu::DeviceVector<int> i_face, j_face, k_face;
    amrex::Gpu::DeviceVector<uint8_t> masks;
    amrex::Gpu::DeviceVector<amrex::Real> area, rho_face;
    amrex::BaseFab<std::vector<int>> inds;

    /*
     *    Numbering convention:
     *    -------------------
     *    |  2  |  4  |  8  |
     *    -------------------
     *    |  16 |     |  32 |
     *    -------------------
     *    |  64 | 128 | 256 |
     *    -------------------
     *
     */
    uint8_t top_left = 2;
    uint8_t top = 4;
    uint8_t top_right = 8;
    uint8_t left = 16;
    uint8_t right = 32;
    uint8_t bottom_left = 64;
    uint8_t bottom = 128;
    uint8_t bottom_right = 256;

    uint8_t
    AddIndicesToList(int i, int j, int pos){

        if(pos > masks.size()){
            masks.push_back(0);
        }
        if(i == -1){
            if(j == -1){
                masks[pos] =  top_left | masks[pos];
            }else if(j == 0){
                masks[pos] =  top | masks[pos];
            }else if(j == 1){
                masks[pos] =  top_right | masks[pos];
            }
        }else if(i == 0){
            if(j == -1){
                masks[pos] =  left | masks[pos];
            }else if(j == 1){
                masks[pos] =  right | masks[pos];
            }
        }else if(i == 1){
            if(j == -1){
                masks[pos] =  bottom_left | masks[pos];
            }else if(j == 0){
                masks[pos] =  bottom | masks[pos];
            }else if(j == 1){
                masks[pos] =  bottom_right | masks[pos];
            }
        }
        amrex::Abort("Wrong indices in FaceInfoBox");
    }

    std::tuple<std::vector<int> , std::vector<int>>
    GetIndicesFromPos(int pos){
        uint8_t mask = masks[pos];
        std::vector<int> i_res, j_res;
        if((mask & top_left) == top_left){
            i_res.push_back(-1);
            j_res.push_back(-1);
        }
        if((mask & top) == top){
            i_res.push_back(-1);
            j_res.push_back(0);
        }
        if( (mask & top_right) == top_left){
            i_res.push_back(-1);
            j_res.push_back(1);
        }
        if( (mask & left) == left){
            i_res.push_back(0);
            j_res.push_back(-1);
        }
        if( (mask & right) == right){
            i_res.push_back(0);
            j_res.push_back(1);
        }
        if((mask & bottom_left) == bottom_left){
            i_res.push_back(1);
            j_res.push_back(-1);
        }
        if((mask & bottom) == bottom){
            i_res.push_back(1);
            j_res.push_back(0);
        }
        if( (mask & bottom_right) == bottom_left){
            i_res.push_back(1);
            j_res.push_back(1);
        }

        return std::make_tuple(i_res, j_res);
    }
};

#endif //WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXFACEINFOBOX_H
