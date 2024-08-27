import streamlit as st
import pydicom
import numpy as np
import plotly.graph_objs as go

st.markdown("""
    <style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st.session_state.theme = "light"

# Streamlit UI
st.title("Dose Volume Histogram (DVH)")

# File upload widgets
rtstruct_file = st.file_uploader("Upload RT Structure File (.dcm)", type=["dcm"])
rtdose_file = st.file_uploader("Upload RT Dose File (.dcm)", type=["dcm"])

# Use default files if no input is given
if rtstruct_file is None: rtstruct_file = 'rts.dcm'
if rtdose_file is None: rtdose_file = 'dose.dcm'

if rtstruct_file and rtdose_file:
    # Load DICOM files
    rtstruct = pydicom.dcmread(rtstruct_file)
    rtdose = pydicom.dcmread(rtdose_file)
    
    # Extract structure names
    structures = {roi.ROIName: roi.ROINumber for roi in rtstruct.StructureSetROISequence}

    all_options = ["All"] + list(structures.keys())

    selected_structures = st.multiselect(
        "Select Structures to Display",
        all_options,
        key='option'
    )

    if 'All' in selected_structures:
        if selected_structures == ['All']:
            selected_structures = list(structures.keys())
        else:
            selected_structures.remove('All')
        
    # Dose grid information
    dose_grid = rtdose.pixel_array
    dose_grid = dose_grid * rtdose.DoseGridScaling  # Apply scaling factor
    
    # Extract pixel spacing and slice thickness
    pixel_spacing = rtdose.PixelSpacing  # [x_spacing, y_spacing]
    slice_thickness = rtdose.GridFrameOffsetVector[1] - rtdose.GridFrameOffsetVector[0]  # z_spacing
    
    fig1 = go.Figure()
    fig2 = go.Figure()
    
    for structure_name, roi_number in structures.items():
        if structure_name not in selected_structures:
            continue

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
                if max(doses) == 0: continue
                hist, bin_edges = np.histogram(doses, bins=100, range=(min(doses), max(doses)))
                cumulative_volume = np.cumsum(hist[::-1])[::-1]
                
                # Normalize cumulative volume to get the ratio of the total structure volume
                total_volume = np.sum(hist)
                cumulative_volume_ratio = (cumulative_volume / total_volume)*100

                # Add DVH trace for this structure
                fig1.add_trace(go.Scatter(
                    x=bin_edges[:-1], 
                    y=cumulative_volume_ratio, 
                    mode='lines',
                    name=f"DVH for {structure_name}"
                ))

                 # Add DVH trace for this structure
                fig2.add_trace(go.Scatter(
                    x=bin_edges[:-1], 
                    y=cumulative_volume, 
                    mode='lines',
                    name=f"DVH for {structure_name}"
                ))
                
    # Customize and show the plot
    fig1.update_layout(
        title="Dose Volume Histograms (DVH) for Selected Structures",
        xaxis_title="Dose (Gy)",
        yaxis_title="Ratio of Total Structure Volume (%)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template="plotly_white",
        legend_title="Structures"
    )

    st.plotly_chart(fig1)
    
    # Customize and show the plot
    fig2.update_layout(
        title="Dose Volume Histograms (DVH) for Selected Structures",
        xaxis_title="Dose (Gy)",
        yaxis_title="Structure Volume (cc)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template="plotly_white",
        legend_title="Structures"
    )

    st.plotly_chart(fig2)
