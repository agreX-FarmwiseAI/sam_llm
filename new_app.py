# --- START OF FILE new_app.py ---

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium # Using streamlit_folium for drawing
import geopandas as gpd
import os
import time
import json
from datetime import datetime

# --- Import the backend logic ---
# Ensure current_sam.py is in the same directory
try:
    import current_sam # Import the logic file
except ModuleNotFoundError:
    st.error("❌ **Error:** `current_sam.py` not found. Please ensure it's in the same directory as `new_app.py`.")
    st.stop()
except Exception as e:
    st.error(f"❌ **Error importing `current_sam.py`:** {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="SAM Land Classifier", layout="wide", initial_sidebar_state="collapsed")

# --- App Title and Introduction ---
st.title("🌍 SAM Land Classifier with LLM")
st.markdown("""
Welcome! This tool uses the Segment Anything Model (SAM) and a Large Language Model (LLM)
to segment and classify land cover within a selected satellite image region.
""")
st.divider() # Visual separator

# --- Constants and Configuration ---
OUTPUT_DIR = "sam_output_streamlit"
SEGMENT_IMG_DIR = "sam_segment_images_streamlit"
DEFAULT_CENTER = [10.4397, 79.3302] # Center near the example ROI from new_sam.py
DEFAULT_ZOOM = 18
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENT_IMG_DIR, exist_ok=True)
GEOJSON_RESULTS_FILE = os.path.join(OUTPUT_DIR, "segmented_parcels.geojson")

# --- Check for prerequisites ---
if not os.path.exists("sam_prompt.txt"):
    st.warning("⚠️ **Warning:** `sam_prompt.txt` not found. LLM classification might be disabled or use default settings.")
    # Decide if you want to stop execution
    # st.stop()

# --- Session State Initialization ---
if 'roi_bounds' not in st.session_state:
    st.session_state.roi_bounds = None # Will store [min_lon, min_lat, max_lon, max_lat]
if 'geojson_path' not in st.session_state:
    st.session_state.geojson_path = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'map_key' not in st.session_state:
    st.session_state.map_key = 0 # To force map redraw when needed
if 'last_drawn_roi' not in st.session_state:
    st.session_state.last_drawn_roi = None # Store the raw drawn geometry


# --- Step 1: Select ROI ---
with st.container():
    st.subheader("📍 Step 1: Select Region of Interest (ROI)")
    st.markdown("""
    Navigate the map below to find your area. Use the **Rectangle tool** ( <i class="fa fa-square-o" style="color:black;"></i> icon on the left)
    to draw a box around your target region. The coordinates will appear below the map once drawn.
    """) # Corrected icon hint

    # Create a Folium map for ROI selection
    m = folium.Map(
            location=DEFAULT_CENTER,
            zoom_start=DEFAULT_ZOOM,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            control_scale=True
        )

    # Add drawing tools
    draw_options = {
        'polyline': False, 'polygon': False, 'circle': False, 'marker': False,
        'circlemarker': False, 'rectangle': True
    }
    # Allow deleting drawings, disable editing after creation for simplicity here
    Draw(export=False, draw_options=draw_options, edit_options={'edit': False, 'remove': True}).add_to(m)

    # Render the map using st_folium
    map_output = st_folium(m, key=f"draw_map_{st.session_state.map_key}", width='100%', height=500)

    # --- Process Drawn ROI ---
    drawn_roi = None
    if map_output and map_output.get("last_active_drawing"):
        # Use last_active_drawing which reflects the final state after drawing/deleting
        last_drawing = map_output["last_active_drawing"]

        # Check if the geometry exists and is a Polygon (rectangles are polygons)
        if last_drawing and last_drawing.get("geometry") and last_drawing["geometry"]["type"] == "Polygon":
            st.session_state.last_drawn_roi = last_drawing # Store the raw drawing
            coords = last_drawing["geometry"]["coordinates"][0]
            # Check for valid polygon (at least 4 points for a rectangle, last=first)
            if len(coords) >= 4:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                min_lon, max_lon = min(lons), max(lons)
                min_lat, max_lat = min(lats), max(lats)

                # Validate coordinates (simple check for non-zero area)
                if min_lon != max_lon and min_lat != max_lat:
                    drawn_roi = [min_lon, min_lat, max_lon, max_lat]
                    # Update session state only if it's a different, valid drawing
                    if st.session_state.roi_bounds != drawn_roi:
                        st.session_state.roi_bounds = drawn_roi
                        st.rerun() # Rerun to update the displayed info and button state
                else:
                    # Handle case where rectangle drawn has zero area (line)
                    if st.session_state.roi_bounds is not None:
                        st.warning("Invalid rectangle drawn (zero area). Please redraw or use previously confirmed ROI.")
                        st.session_state.roi_bounds = None # Clear invalid ROI if it was just drawn
                        st.session_state.last_drawn_roi = None # Clear invalid drawing
                        st.rerun()
            else:
                 # Handle case where geometry is not a valid polygon
                 if st.session_state.roi_bounds is not None:
                     st.warning("Invalid shape drawn. Please draw a rectangle.")
                     st.session_state.roi_bounds = None
                     st.session_state.last_drawn_roi = None
                     st.rerun()

        # Handle case where the user deleted the drawing
        elif map_output.get("last_active_drawing") is None and st.session_state.last_drawn_roi is not None:
             st.session_state.roi_bounds = None
             st.session_state.last_drawn_roi = None
             st.rerun() # Rerun to clear displayed ROI


    # Display the currently selected/confirmed ROI
    if st.session_state.roi_bounds:
        st.success(f"✅ ROI Ready: `{st.session_state.roi_bounds}`")
    elif st.session_state.last_drawn_roi and st.session_state.roi_bounds is None:
         # This case means an invalid shape was drawn (e.g., zero area)
         st.warning("⚠️ Please draw a valid rectangle with non-zero area.")
    else:
        st.info("ℹ️ Draw a rectangle on the map to define your ROI.")


st.divider() # Visual separator

# --- Step 2: Confirmation and Processing ---
with st.container():
    st.subheader("⚙️ Step 2: Confirm and Process")
    confirm_button = st.button(
        "Confirm ROI and Start Processing",
        disabled=(st.session_state.roi_bounds is None),
        type="primary" # Make the button more prominent
    )

    if confirm_button and st.session_state.roi_bounds:
        st.session_state.processing_complete = False # Reset flag
        st.session_state.geojson_path = None # Reset path

        # --- Loader for SAM Segmentation ---
        status_placeholder = st.empty() # Placeholder for status updates
        with st.spinner("Initializing processing..."): # Brief initial spinner
             time.sleep(1) # Simulate initialization


        with st.status("🚀 Running Analysis...", expanded=True) as status:
            try:
                start_time = time.time()
                st.write("Fetching satellite imagery for the ROI...")
                # (Assuming the backend function downloads the image first)
                time.sleep(1.5) # Simulate download time

                st.write("🧠 Running SAM segmentation...")
                # Call the main function from current_sam.py
                # Pass the specific output file path for clarity
                gdf_results = current_sam.segment_image_with_sam(
                    output_folder=OUTPUT_DIR,
                    roi_coordinates=st.session_state.roi_bounds,
                    zoom=DEFAULT_ZOOM, # Consider making this an advanced option
                    segment_image_output=SEGMENT_IMG_DIR,
                    perform_llm_classification=True, # Explicitly request classification
                )
                st.write("🤖 Performing LLM classification...")
                # (The backend function handles this internally now, but good to show progress)
                time.sleep(1) # Simulate classification time if separate


                end_time = time.time()
                processing_time = end_time - start_time
                status.update(label=f"✅ Analysis Complete! ({processing_time:.2f}s)", state="complete", expanded=False)

                # Check if the expected output file exists
                if os.path.exists(GEOJSON_RESULTS_FILE):
                    st.session_state.geojson_path = GEOJSON_RESULTS_FILE
                    st.session_state.processing_complete = True
                    st.session_state.map_key += 1 # Force results map redraw
                    st.balloons() # Fun success indicator
                else:
                    st.error(f"❌ Processing finished, but the output file was not found: {GEOJSON_RESULTS_FILE}")
                    st.session_state.processing_complete = False

            except ImportError as ie:
                 st.error(f"❌ Import Error during processing: {ie}. Is `current_sam.py` accessible and correct?")
                 status.update(label="Processing Failed (Import Error)", state="error")
                 st.session_state.processing_complete = False
            except FileNotFoundError as fnf:
                 st.error(f"❌ File Not Found Error during processing: {fnf}. Check paths and prerequisite files like `sam_prompt.txt`.")
                 status.update(label="Processing Failed (File Not Found)", state="error")
                 st.session_state.processing_complete = False
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {type(e).__name__} - {e}")
                status.update(label="Processing Failed", state="error")
                # import traceback
                # st.error(traceback.format_exc()) # Uncomment for detailed debugging
                st.session_state.processing_complete = False

        # Rerun to display results section correctly after processing
        if st.session_state.processing_complete:
            st.rerun()


st.divider() # Visual separator

# --- Step 3: Display Results ---
with st.container():
    st.subheader("📊 Step 3: View Results")

    if st.session_state.processing_complete and st.session_state.geojson_path:
        st.success(f"Results loaded successfully from: `{st.session_state.geojson_path}`")
        try:
            gdf = gpd.read_file(st.session_state.geojson_path)
            # Ensure data is in web-friendly projection
            if gdf.crs != "EPSG:4326":
                 gdf = gdf.to_crs("EPSG:4326")


            if not gdf.empty:
                st.info(f"Displaying {len(gdf)} segmented parcels.")

                # Create a new map centered on the results
                results_map = folium.Map(control_scale=True, tiles=None) # No base tiles initially

                # Add Esri World Imagery base layer
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri",
                    name="Esri Satellite",
                    overlay=False,
                    control=True
                ).add_to(results_map)


                # --- Prepare Tooltip with Fixed Width ---
                tooltip_fields = []
                if 'llm_class' in gdf.columns: tooltip_fields.append('llm_class')
                if 'llm_conf' in gdf.columns: tooltip_fields.append('llm_conf')
                if 'llm_reason' in gdf.columns: tooltip_fields.append('llm_reason')

                if not tooltip_fields:
                     st.warning("LLM classification columns (llm_class, llm_conf, llm_reason) not found. Tooltip will be basic.")
                     tooltip = None
                else:
                    aliases = [f"{field.replace('llm_', '').capitalize()}:" for field in tooltip_fields]

                    # --- *** MODIFIED TOOLTIP CONFIGURATION *** ---
                    # Use Folium's max_width parameter directly for better control
                    # Keep CSS for styling only (background, border, font etc.)
                    tooltip_style_css = """
                        background-color: #F0EFEF;
                        border: 1px solid black;
                        border-radius: 3px;
                        box-shadow: 3px;
                        /* width and max-width are now controlled by the parameter below */
                        padding: 5px;
                        font-size: 11px; /* Slightly smaller font if needed */
                        word-wrap: break-word; /* Crucial for wrapping long text */
                        white-space: normal; /* Ensure wrapping works */
                    """
                    tooltip = folium.features.GeoJsonTooltip(
                        fields=tooltip_fields,
                        aliases=aliases,
                        localize=True,
                        sticky=False, # Tooltip follows mouse, disappears on mouseout
                        labels=True,
                        style=tooltip_style_css, # Pass the CSS string for styling
                        max_width=40, # *** KEY CHANGE: Use parameter for width control (pixels) ***
                    )
                    # --- *** END OF MODIFIED TOOLTIP CONFIGURATION *** ---


                # --- Add GeoJSON layer with Adjusted Transparency ---
                folium.GeoJson(
                    gdf,
                    # --- MODIFIED STYLE FUNCTION ---
                    style_function=lambda x: {
                        'fillColor': '#ffff00', # Yellow fill
                        'color': '#000000',     # Black border
                        'weight': 1,             # Border weight
                        'fillOpacity': 0.3       # Adjusted transparency (was 0.5)
                    },
                    tooltip=tooltip, # Use the configured tooltip object
                    name='Segmented Parcels'
                ).add_to(results_map)

                # Add Layer Control
                folium.LayerControl().add_to(results_map)

                # Zoom to the bounds of the GeoJSON
                try:
                     bounds = gdf.total_bounds # [minx, miny, maxx, maxy]
                     results_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                except Exception as fit_e:
                     st.warning(f"Could not automatically zoom to layer bounds: {fit_e}. Using ROI center.")
                     if st.session_state.roi_bounds:
                         center_lon = (st.session_state.roi_bounds[0] + st.session_state.roi_bounds[2]) / 2
                         center_lat = (st.session_state.roi_bounds[1] + st.session_state.roi_bounds[3]) / 2
                         results_map.location = [center_lat, center_lon]
                         results_map.zoom_start = DEFAULT_ZOOM # Use configured zoom


                # Display the results map
                st_folium(results_map, key=f"results_map_{st.session_state.map_key}", width='100%', height=600, returned_objects=[])

                # --- Optional: Display GeoDataFrame Table ---
                with st.expander("📂 Show Segment Data Table"):
                     display_cols = [col for col in gdf.columns if col != 'geometry'] # Exclude geometry
                     if not display_cols:
                         st.info("No attribute data found to display in the table.")
                     else:
                        try:
                            # Select only non-geometry columns for display
                            st.dataframe(gdf[display_cols])
                        except Exception as df_e:
                            st.error(f"Could not display dataframe: {df_e}")
                            st.write("Attempting to display basic info:")
                            st.write(gdf[display_cols].head()) # Fallback


                # --- Optional: Download Button ---
                try:
                    # Ensure geometry is preserved for download
                    geojson_data = gdf.to_json()
                    st.download_button(
                        label="💾 Download Results GeoJSON",
                        data=geojson_data,
                        file_name=f"sam_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                        mime="application/geo+json"
                    )
                except Exception as download_e:
                    st.error(f"Failed to prepare GeoJSON for download: {download_e}")

            else:
                 st.warning("Processing completed, but the resulting GeoJSON file is empty.")


        except FileNotFoundError:
            st.error(f"❌ Error: Could not find the results file: {st.session_state.geojson_path}")
            st.session_state.processing_complete = False # Reset state
        except ImportError:
            st.error("❌ Error: Geopandas might not be installed correctly or is missing.")
            st.session_state.processing_complete = False # Reset state
        except Exception as e:
            st.error(f"❌ An error occurred displaying the results: {type(e).__name__} - {e}")
            # import traceback
            # st.error(traceback.format_exc()) # Uncomment for detailed debugging
            st.session_state.processing_complete = False # Reset state

    elif st.session_state.roi_bounds and not st.session_state.processing_complete and not confirm_button:
         # Case where ROI is selected, but processing hasn't run or failed previously
         st.info("Click the 'Confirm ROI and Start Processing' button above to generate results.")
    elif not st.session_state.roi_bounds:
        # Case where no ROI has been selected yet
        st.info("Draw a rectangle on the first map and confirm it to see results here.")


# --- END OF FILE new_app.py ---