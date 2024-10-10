import streamlit as st
from dicompylercore import dicomparser, dvh
import numpy as np
import matplotlib.path
import pydicom
import plotly.graph_objects as go
from six import iteritems


st.set_page_config(layout="wide")

def get_dvh(structure, dose, roi, limit=None, callback=None):
    """Calculate a cumulative DVH in Gy from a DICOM RT Structure Set & Dose."""
    rtss = dicomparser.DicomParser(structure)
    rtdose = dicomparser.DicomParser(dose)
    structures = rtss.GetStructures()

    s = structures[roi]
    s['planes'] = rtss.GetStructureCoordinates(roi)
    print(rtss.GetStructureCoordinates(roi))
    s['thickness'] = rtss.CalculatePlaneThickness(s['planes'])
    hist = calculate_dvh(s, rtdose, limit, callback)
    return dvh.DVH(counts=hist,
                   bins=(np.arange(0, 2) if (hist.size == 1) else
                         np.arange(0, hist.size + 1) / 100),
                   dvh_type='differential',
                   dose_units='gy',
                   name=s['name']
                   ).cumulative


def calculate_dvh(structure, dose, limit=None, callback=None):
    """Calculate the differential DVH for the given structure and dose grid."""
    planes = structure['planes']

    if ((len(planes)) and ("PixelData" in dose.ds)):
        dd = dose.GetDoseData()
        id = dose.GetImageData()

        x, y = np.meshgrid(np.array(dd['lut'][0]), np.array(dd['lut'][1]))
        x, y = x.flatten(), y.flatten()
        dosegridpoints = np.vstack((x, y)).T

        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
        if isinstance(limit, int):
            if (limit < maxdose):
                maxdose = limit
        hist = np.zeros(maxdose)
    else:
        return np.array([0])

    n = 0
    planedata = {}
    for z, plane in iteritems(planes):
        doseplane = dose.GetDoseGrid(z)
        planedata[z] = calculate_plane_histogram(
            plane, doseplane, dosegridpoints,
            maxdose, dd, id, structure, hist)
        n += 1
        if callback:
            callback(n, len(planes))
    volume = sum([p[1] for p in planedata.values()]) / 1000
    hist = sum([p[0] for p in planedata.values()])
    hist = hist * volume / sum(hist)
    hist = np.trim_zeros(hist, trim='b')

    return hist


def calculate_plane_histogram(plane, doseplane, dosegridpoints,
                              maxdose, dd, id, structure, hist):
    contours = [[x[0:2] for x in c['data']] for c in plane]

    if not len(doseplane):
        return (np.arange(0, maxdose), 0)

    grid = np.zeros((dd['rows'], dd['columns']), dtype=np.uint8)

    for i, contour in enumerate(contours):
        m = get_contour_mask(dd, id, dosegridpoints, contour)
        grid = np.logical_xor(m.astype(np.uint8), grid).astype(bool)

    hist, vol = calculate_contour_dvh(
        grid, doseplane, maxdose, dd, id, structure)
    return (hist, vol)


def get_contour_mask(dd, id, dosegridpoints, contour):
    doselut = dd['lut']

    c = matplotlib.path.Path(list(contour))
    grid = c.contains_points(dosegridpoints)
    grid = grid.reshape((len(doselut[1]), len(doselut[0])))

    return grid


def calculate_contour_dvh(mask, doseplane, maxdose, dd, id, structure):
    mask = np.ma.array(doseplane * dd['dosegridscaling'] * 100, mask=~mask)
    hist, edges = np.histogram(mask.compressed(),
                               bins=maxdose,
                               range=(0, maxdose))

    vol = sum(hist) * ((id['pixelspacing'][0]) *
                       (id['pixelspacing'][1]) *
                       (structure['thickness']))
    return hist, vol


def main():
    st.title("DVH Calculation from DICOM RT Structure Set & Dose")

    st.sidebar.header("Upload DICOM Files")
    rtss_file = st.sidebar.file_uploader("Upload RT Structure Set (rts.dcm)", type=["dcm"])
    rtdose_file = st.sidebar.file_uploader("Upload RT Dose (dose.dcm)", type=["dcm"])

    if rtss_file is None: rtss_file = 'rts.dcm'
    if rtdose_file is None: rtdose_file = 'dose.dcm'
    
    if rtss_file is not None and rtdose_file is not None:
        # Read the DICOM files using pydicom
        rtss_file = pydicom.dcmread(rtss_file, force=True)
        rtdose_file = pydicom.dcmread(rtdose_file, force=True)

        rtss_parser = dicomparser.DicomParser(rtss_file)
       
        RTstructures = rtss_parser.GetStructures()

        # Create a dictionary mapping structure names to structure IDs
        structure_name_to_id = {structure['name']: key for key, structure in RTstructures.items()}

        # Using multiselect for selecting multiple structures by name
        selected_structure_names = st.sidebar.multiselect(
            "Select Structures for DVH Calculation", list(structure_name_to_id.keys())
        )

        if st.sidebar.button("Calculate DVH"):
            calcdvhs = {}
            fig = go.Figure()

            # Loop through the selected structure names and get the corresponding structure IDs
            for structure_name in selected_structure_names:
                structure_id = structure_name_to_id[structure_name]
                structure = RTstructures[structure_id]
                calcdvhs[structure_id] = get_dvh(rtss_file, rtdose_file, structure_id)
                
                if calcdvhs[structure_id].counts.any():
                    # Add the DVH plot for each selected structure
                    fig.add_trace(go.Scatter(
                        x=np.arange(0,len(calcdvhs[structure_id].counts))/100,
                        y=calcdvhs[structure_id].counts * 100 / calcdvhs[structure_id].counts[0],
                        mode='lines',
                        name=structure['name'],
                        line=dict(color=f'rgb({structure["color"][0]}, {structure["color"][1]}, {structure["color"][2]})', dash='dash')
                    ))

            # Customize the plot
            fig.update_layout(
                title="Dose Volume Histogram (DVH)",
                xaxis_title="Dose (Gy)",
                yaxis_title="Ratio of Total Structure Volume (%)",
                xaxis = dict(
                    tickmode = 'linear',
                    tick0 = 0,
                    dtick = 2,
                    showgrid=True
                    ),
                font=dict(size=18, color="black"),
                height=600
            )

            # Display the plot
            st.plotly_chart(fig)

        else:
            st.write("Please select at least one structure to calculate the DVH.")
    else:
        st.write("Please upload both RT Structure Set and RT Dose files to proceed.")

if __name__ == "__main__":
    main()
