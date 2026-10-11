# Metal GPU back end for the FV1, FV2, DG2 and ACC solvers (Apple Silicon)

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    # The kernels are compiled to a metallib next to the executable, which loads
    # it at run time. -fno-fast-math keeps the GPU arithmetic identical to the
    # CPU solver's, so the verification cases are bit-for-bit reproducible.
    set(METALLIB ${CMAKE_BINARY_DIR}/metal_fv1_kernels.metallib)
    add_custom_command(
        OUTPUT ${METALLIB}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/fv1/metal_fv1_kernels.metal
                -o ${CMAKE_BINARY_DIR}/metal_fv1_kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_fv1_kernels.air
                -o ${METALLIB}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fv1/metal_fv1_kernels.metal
        COMMENT "Compiling Metal FV1 kernels")
    # ACC kernels: -fno-fast-math as well (correctly rounded sqrt for the CFL
    # step; pow still differs from the CPU's powf in the last bit)
    set(METALLIB_ACC ${CMAKE_BINARY_DIR}/metal_acc_kernels.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_ACC}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/acc/metal_acc_kernels.metal
                -o ${CMAKE_BINARY_DIR}/metal_acc_kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_acc_kernels.air
                -o ${METALLIB_ACC}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/acc/metal_acc_kernels.metal
        COMMENT "Compiling Metal ACC kernels")
    set(METALLIB_FV2 ${CMAKE_BINARY_DIR}/metal_fv2_kernels.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_FV2}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/fv2/metal_fv2_kernels.metal
                -o ${CMAKE_BINARY_DIR}/metal_fv2_kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_fv2_kernels.air
                -o ${METALLIB_FV2}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fv2/metal_fv2_kernels.metal
        COMMENT "Compiling Metal FV2 kernels")
    set(METALLIB_DG2 ${CMAKE_BINARY_DIR}/metal_dg2_kernels.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_DG2}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/dg2/metal_dg2_kernels.metal
                -o ${CMAKE_BINARY_DIR}/metal_dg2_kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_dg2_kernels.air
                -o ${METALLIB_DG2}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dg2/metal_dg2_kernels.metal
        COMMENT "Compiling Metal DG2 kernels")
    # infiltration and evaporation, shared by the four solvers
    set(METALLIB_LOSSES ${CMAKE_BINARY_DIR}/metal_losses_kernels.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_LOSSES}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/losses/metal_losses_kernels.metal
                -o ${CMAKE_BINARY_DIR}/metal_losses_kernels.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_losses_kernels.air
                -o ${METALLIB_LOSSES}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/losses/metal_losses_kernels.metal
        COMMENT "Compiling Metal loss kernels")
    set(METALLIB_TREE ${CMAKE_BINARY_DIR}/metal_tree_primitives.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_TREE}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_tree_primitives.metal
                -o ${CMAKE_BINARY_DIR}/metal_tree_primitives.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_tree_primitives.air
                -o ${METALLIB_TREE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_tree_primitives.metal
        COMMENT "Compiling native Metal tree primitives")
    set(METALLIB_MWDG2 ${CMAKE_BINARY_DIR}/metal_mwdg2_kernels.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_MWDG2}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -I ${CMAKE_CURRENT_SOURCE_DIR}/adaptive
                -c ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_flow.metal
                -o ${CMAKE_BINARY_DIR}/metal_mwdg2_flow.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_mwdg2_flow.air
                -o ${METALLIB_MWDG2}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_flow.metal
                ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_mra.metal
                ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/mwdg2_transform.h
                ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/mwdg2_filters.h
        COMMENT "Compiling native Metal MWDG2 hierarchy and flow kernels")
    set(METALLIB_MWDG2_SELECTION ${CMAKE_BINARY_DIR}/metal_mwdg2_selection.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_MWDG2_SELECTION}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_selection.metal
                -o ${CMAKE_BINARY_DIR}/metal_mwdg2_selection.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_mwdg2_selection.air
                -o ${METALLIB_MWDG2_SELECTION}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_selection.metal
        COMMENT "Compiling CUDA-equivalent Metal MWDG2 mesh selection kernels")
    set(METALLIB_MWDG2_INPUT ${CMAKE_BINARY_DIR}/metal_mwdg2_input.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_MWDG2_INPUT}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_input.metal
                -o ${CMAKE_BINARY_DIR}/metal_mwdg2_input.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_mwdg2_input.air
                -o ${METALLIB_MWDG2_INPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_input.metal
        COMMENT "Compiling Metal MWDG2 raster projection kernel")
    set(METALLIB_MWDG2_RECON ${CMAKE_BINARY_DIR}/metal_mwdg2_reconstruct.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_MWDG2_RECON}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_minicase.metal
                -o ${CMAKE_BINARY_DIR}/metal_mwdg2_reconstruct.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_mwdg2_reconstruct.air
                -o ${METALLIB_MWDG2_RECON}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_minicase.metal
        COMMENT "Compiling Metal MWDG2 adaptive reconstruction kernels")
    set(METALLIB_MWDG2_NATIVE ${CMAKE_BINARY_DIR}/metal_mwdg2_native.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_MWDG2_NATIVE}
        COMMAND xcrun -sdk macosx metallib
                ${CMAKE_BINARY_DIR}/metal_mwdg2_flow.air
                ${CMAKE_BINARY_DIR}/metal_mwdg2_selection.air
                ${CMAKE_BINARY_DIR}/metal_tree_primitives.air
                ${CMAKE_BINARY_DIR}/metal_mwdg2_input.air
                ${CMAKE_BINARY_DIR}/metal_mwdg2_reconstruct.air
                -o ${METALLIB_MWDG2_NATIVE}
        DEPENDS ${METALLIB_MWDG2} ${METALLIB_MWDG2_SELECTION}
                ${METALLIB_TREE} ${METALLIB_MWDG2_INPUT} ${METALLIB_MWDG2_RECON}
        COMMENT "Linking native adaptive Metal MWDG2 kernels")
    set(METALLIB_ACC_NUGRID ${CMAKE_BINARY_DIR}/metal_acc_nugrid.metallib)
    add_custom_command(
        OUTPUT ${METALLIB_ACC_NUGRID}
        COMMAND xcrun -sdk macosx metal -fno-fast-math
                -c ${CMAKE_CURRENT_SOURCE_DIR}/acc_nugrid/metal_acc_nugrid.metal
                -o ${CMAKE_BINARY_DIR}/metal_acc_nugrid.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_acc_nugrid.air
                -o ${METALLIB_ACC_NUGRID}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/acc_nugrid/metal_acc_nugrid.metal
        COMMENT "Compiling Metal ACC non-uniform grid kernels")
    add_custom_target(metal_shaders DEPENDS ${METALLIB} ${METALLIB_ACC} ${METALLIB_FV2} ${METALLIB_DG2} ${METALLIB_LOSSES} ${METALLIB_TREE} ${METALLIB_MWDG2} ${METALLIB_MWDG2_SELECTION} ${METALLIB_MWDG2_INPUT} ${METALLIB_MWDG2_RECON} ${METALLIB_MWDG2_NATIVE} ${METALLIB_ACC_NUGRID})

    add_library(lisflood-metal STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/metal_integrate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fv1/metal_fv1_simulate.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/acc/metal_acc_simulate.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/fv2/metal_fv2_simulate.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/dg2/metal_dg2_simulate.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/losses/metal_losses.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_scan.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_mesh.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/adaptive/metal_mwdg2_simulate.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/acc_nugrid/metal_acc_nugrid_simulate.mm)
    add_dependencies(lisflood-metal metal_shaders)
    target_include_directories(lisflood-metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
    target_link_libraries(lisflood-metal PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} lisflood-base)
    target_compile_options(lisflood-metal PRIVATE -fobjc-arc)
    target_compile_definitions(lisflood-metal PUBLIC METAL_ENABLED)

    set(METAL_ENABLED TRUE PARENT_SCOPE)
    set(METAL_LIBRARY lisflood-metal PARENT_SCOPE)
else()
    message(WARNING "Metal framework not found: building without the Metal back end")
endif()
