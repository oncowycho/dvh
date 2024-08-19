import streamlit as st
import pydicom
import numpy as np
import plotly.graph_objs as go

# Streamlit UI
st.title("Dose Volume Histogram (DVH) Viewer")

# File upload widgets
rtstruct_file = st.file_uploader("Upload RT Structure File (.dcm)", type=["dcm"])
rtdose_file = st.file_uploader("Upload RT Dose File (.dcm)", type=["dcm"])

if rtstruct_file and rtdose_file:
    # Load DICOM files
    rtstruct = pydicom.dcmread(rtstruct_file)
    rtdose = pydicom.dcmread(rtdose_file)

    # Extract structure names
    structures = {roi.ROIName: roi.ROINumber for roi in rtstruct.StructureSetROISequence}
    
    # Dose grid information
    dose_grid = rtdose.pixel_array
    dose_grid = dose_grid * rtdose.DoseGridScaling  # Apply scaling factor

    # Extract pixel spacing and slice thickness
    pixel_spacing = rtdose.PixelSpacing  # [x_spacing, y_spacing]
    slice_thickness = rtdose.GridFrameOffsetVector[1] - rtdose.GridFrameOffsetVector[0]  # z_spacing

    fig = go.Figure()

    oars = ['PTV-OAR',
            'OAR-RA1-100[%]','OAR-RA1-090[%]','OAR-RA1-080[%]','OAR-RA1-070[%]','OAR-RA1-060[%]',]
            # 'OAR-RA1-050[%]','OAR-RA1-040[%]','OAR-RA1-030[%]','OAR-RA1-020[%]','OAR-RA1-010[%]']
    
    for structure_name, roi_number in structures.items():
        if structure_name not in oars: continue
        # Find contours for each structure
        contours = None
        for structure in rtstruct.ROIContourSequence:
            if structure.ReferencedROINumber == roi_number and hasattr(structure, "ContourSequence"):
                contours = structure.ContourSequence
                break

        if contours:
            # Calculate DVH for the selected structure
            doses = []
            for contour in contours:
                contour_data = np.array(contour.ContourData).reshape(-1, 3)
                for point in contour_data:
                    # Convert contour point to voxel coordinates
                    voxel_index = np.round([
                        (point[0] - rtdose.ImagePositionPatient[0]) / pixel_spacing[0],
                        (point[1] - rtdose.ImagePositionPatient[1]) / pixel_spacing[1],
                        (point[2] - rtdose.ImagePositionPatient[2]) / slice_thickness
                    ]).astype(int)

                    # Ensure voxel indices are within the valid range
                    if all(0 <= idx < dim for idx, dim in zip(voxel_index, dose_grid.shape)):
                        doses.append(dose_grid[tuple(voxel_index)])

            # Create DVH
            if doses:
                hist, bin_edges = np.histogram(doses, bins=100)
                cumulative_volume = np.cumsum(hist[::-1])[::-1]

                # Add DVH trace for this structure
                fig.add_trace(go.Scatter(
                    x=bin_edges[:-1], 
                    y=cumulative_volume, 
                    mode='lines',
                    name=f"DVH for {structure_name}"
                ))

    # Customize and show the plot
    fig.update_layout(
        title="Dose Volume Histograms (DVH) for All Structures",
        xaxis_title="Dose (Gy)",
        yaxis_title="Volume (cc)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template="plotly_white",
        legend_title="Structures"
    )

    st.plotly_chart(fig)
