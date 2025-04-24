# Combined file: SAM Land Classifier Streamlit App with Backend Logic

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium # Using streamlit_folium for drawing
import geopandas as gpd
import os
import time
import json
from datetime import datetime, timedelta
import google.generativeai as genai
import pandas as pd # Added for potential future use, though not strictly needed for current logic
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS # Import CRS object
from PIL import Image
import numpy as np
from samgeo import SamGeo, tms_to_geotiff

# --- LLM Configuration (From current_sam.py) ---
# WARNING: Hardcoding API keys is insecure. Consider using environment variables or a secrets manager.
API_KEYS = [
    "AIzaSyCNb64e_usdhwZd8GLzvngHizyK1wERJ2Q", # Replace with your actual keys
    "AIzaSyA9sRX75ZhHWdPG1UAPD6nCPIN7K6g9Vm4", # Replace with your actual keys
    "AIzaSyC9c3jdozjLBUIrVK0UPPuo9UD7Y3IB2aU", # Replace with your actual keys
    "AIzaSyDxVY2daczzAbkQOZVMcB5HiRY_2xTl02w", # Replace with your actual keys
    "AIzaSyAc2MJMkPQGoP9P26oFw3wxtpU-CrFMdqA", # Replace with your actual keys
    "AIzaSyAeRLHOyj_fwpaLA7fEc2KsvrII9TfcWKE", # Replace with your actual keys
    "AIzaSyBU3k3MFcWr9Xd2eyrAKZK-aT2OIX9PstU", # Replace with your actual keys
]
MODEL_NAME = "gemini-2.0-flash" # Use a known valid model like 1.5 flash - Adjusted model name slightly as 2.0 isn't standard
MIN_KEY_COOLDOWN = 8  # Minimum seconds between uses of the same API key

# Track when each key was last used
last_used = {key: datetime.now() - timedelta(seconds=MIN_KEY_COOLDOWN * 2) for key in API_KEYS} # Initialize to allow immediate use

# Load the LLM Prompt
try:
    with open("sam_prompt.txt", "r") as f:
        llm_prompt = f.read()
except FileNotFoundError:
    # Streamlit apps should handle missing files gracefully
    if 'st' in globals(): # Check if streamlit is running
         st.warning("⚠️ **Warning:** `sam_prompt.txt` not found. LLM classification might be disabled or fail.")
    else: # Running as script?
         print("Warning: sam_prompt.txt not found. LLM classification will likely fail.")
    llm_prompt = None # Set to None to handle the error later

generation_config = {
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k": 64,
    "max_output_tokens": 8182, # Adjust if needed for the prompt/model
    "response_mime_type": "application/json", # Request JSON directly
}

# --- LLM Helper Functions (From current_sam.py) ---

def get_next_available_key():
    """Gets the next available API key respecting the cooldown period."""
    if not API_KEYS:
        raise ValueError("No API keys provided.")

    while True: # Loop until a key is found or wait time is calculated
        now = datetime.now()
        # Sort keys by when they were last used, oldest first
        sorted_keys = sorted(last_used.items(), key=lambda x: x[1])

        for key, last_use_time in sorted_keys:
            time_since_last_use = (now - last_use_time).total_seconds()
            if time_since_last_use >= MIN_KEY_COOLDOWN:
                last_used[key] = now  # Update the last used time
                # print(f"Using API key ending with ...{key[-5:]}") # Optional: for debugging
                return key

        # If all keys are on cooldown, wait for the one that will be available soonest
        soonest_key, soonest_time = sorted_keys[0]
        wait_time = MIN_KEY_COOLDOWN - (now - soonest_time).total_seconds()
        if wait_time > 0:
            # Use st.write for status in Streamlit context if available
            status_msg = f"All API keys on cooldown, waiting {wait_time:.2f} seconds..."
            if 'st' in globals():
                st.write(status_msg) # Show waiting message in Streamlit status
            else:
                print(status_msg)
            time.sleep(wait_time + 0.1) # Add a small buffer
        # Loop again to re-check availability after waiting

def call_llm(image_path):
    """Calls the Gemini LLM to classify the image segment using the next available API key."""
    if not llm_prompt:
        print("Error: LLM prompt is not loaded. Cannot perform classification.")
        # Optionally use st.warning if in Streamlit context
        if 'st' in globals():
             st.warning("LLM Prompt not loaded. Classification skipped for this segment.")
        return None
    if not API_KEYS:
        print("Error: No API keys configured. Cannot perform classification.")
        if 'st' in globals():
            st.warning("No LLM API keys configured. Classification skipped for this segment.")
        return None

    api_key = get_next_available_key()

    try:
        # Configure genai with the selected API key
        genai.configure(api_key=api_key)
        # Ensure model name is correct and available
        model = genai.GenerativeModel(MODEL_NAME) # Removed generation_config here, applied in generate_content

        status_msg = f"Classifying {os.path.basename(image_path)} using API key ...{api_key[-5:]}"
        # Use st.write for status in Streamlit context if available
        if 'st' in globals():
            st.write(f"🧠 {status_msg}") # Add icon for visual cue
        else:
             print(status_msg)


        # Prepare image part
        image_part = {
            "mime_type": "image/png", # We are using PNG files
            "data": open(image_path, "rb").read()
        }

        # Prepare prompt parts
        prompt_parts = [llm_prompt, "INPUT IMAGE:", image_part]

        # Make the API call
        response = model.generate_content(prompt_parts, generation_config=generation_config)

        # Debug: Print raw response text
        # print("Raw LLM Response:", response.text)

        if response.text:
            try:
                # The response should be directly JSON if response_mime_type="application/json" worked
                result_json = json.loads(response.text)
                # Validate expected keys
                if all(k in result_json for k in ["classification", "confidence", "reasoning"]):
                     return result_json
                else:
                    print(f"Error: LLM JSON response missing expected keys for {image_path}. Response: {result_json}")
                    return None
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON response from LLM for {image_path}: {e}. Response text: {response.text}")
                # Attempt to clean common markdown issues if JSON parsing fails initially
                cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                try:
                    result_json = json.loads(cleaned_text)
                    if all(k in result_json for k in ["classification", "confidence", "reasoning"]):
                        print("Successfully parsed JSON after cleaning markdown.")
                        return result_json
                    else:
                         print(f"Error: Cleaned LLM JSON response missing expected keys for {image_path}. Cleaned text: {cleaned_text}")
                         return None
                except json.JSONDecodeError:
                     print(f"Error: Still unable to parse JSON after cleaning for {image_path}.")
                     return None

            except Exception as e: # Catch other potential errors during processing
                 print(f"Error processing LLM response for {image_path}: {type(e).__name__} - {e}")
                 return None
        else:
            # Handle cases where the response might be empty but not necessarily an error object
            # Check for safety ratings or other indicators of blocked content
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"Error: Prompt blocked for {image_path}. Reason: {response.prompt_feedback.block_reason}")
                 last_used[api_key] = datetime.now() # Mark key as used
                 return None
            else:
                print(f"Error: Empty or unexpected response from LLM for {image_path}. Response: {response}")
                return None


    except genai.types.generation_types.BlockedPromptException as bpe:
         print(f"Error: Prompt blocked for {image_path}. Reason: {bpe}")
         # Mark key as used even if blocked to potentially cycle
         last_used[api_key] = datetime.now()
         return None
    except Exception as e:
        # Catch potential API errors like quota exceeded, invalid API key etc.
        print(f"Error calling LLM for {image_path}: {type(e).__name__} - {e}")
        # If we hit a rate limit or other API error, mark key as recently used
        last_used[api_key] = datetime.now()
        return None

# --- Modified SAM Segmentation Function (From current_sam.py) ---

def segment_image_with_sam(
    output_folder,
    roi_coordinates=None,
    zoom=18,
    batch=True,
    erosion_kernel=(3, 3),
    mask_multiplier=255,
    segment_image_output=None,
    perform_llm_classification=True # New parameter to control LLM classification
):
    """
    Segment a satellite image using SAM, save segments as PNGs, optionally classify
    them using an LLM, and add results to the GeoJSON. Ensures outputs use EPSG:4326.

    Parameters:
    - output_folder (str): Directory to save outputs (GeoTIFF, mask, and shapefile).
    - roi_coordinates (list, optional): ROI bounds as [min_lon, min_lat, max_lon, max_lat] in WGS84 (EPSG:4326).
    - zoom (int, optional): Tile zoom level. Default is 17.
    - batch (bool, optional): Enable batch segmentation for large images. Default is True.
    - erosion_kernel (tuple, optional): Kernel size for mask erosion. Default is (3, 3).
    - mask_multiplier (int, optional): Multiplier for mask values. Default is 255.
    - segment_image_output (str, optional): Directory to save individual segment PNG images.
                                            LLM classification requires this directory.
    - perform_llm_classification (bool, optional): Whether to classify each segment PNG using the LLM.
                                                  Requires `segment_image_output` to be set. Default is True.
    """

    # Use st.write for status messages if available
    def log_message(msg, level="info"):
        if 'st' in globals():
            if level == "warning":
                st.warning(msg)
            elif level == "error":
                st.error(msg)
            else:
                st.write(msg) # Default to st.write for info
        else:
            print(f"[{level.upper()}] {msg}")


    # Validate parameters for LLM classification
    if perform_llm_classification and not segment_image_output:
        log_message("Warning: LLM classification requested but `segment_image_output` directory is not provided. Disabling LLM classification.", "warning")
        perform_llm_classification = False
    if perform_llm_classification and not llm_prompt:
         log_message("Warning: LLM classification requested but prompt file 'sam_prompt.txt' was not loaded. Disabling LLM classification.", "warning")
         perform_llm_classification = False
    if perform_llm_classification and not API_KEYS:
         log_message("Warning: LLM classification requested but no API keys are configured. Disabling LLM classification.", "warning")
         perform_llm_classification = False


    # Ensure output directories exist
    os.makedirs(output_folder, exist_ok=True)
    if segment_image_output:
        os.makedirs(segment_image_output, exist_ok=True)

    # Define output file paths
    image_file = os.path.join(output_folder, "satellite.tif")
    mask_file = os.path.join(output_folder, "segment_mask.tif") # Renamed for clarity
    geojson_file = os.path.join(output_folder, "segmented_parcels.geojson") # Renamed for clarity

    # Define target CRS
    target_crs_epsg4326 = CRS.from_epsg(4326) # WGS 84


    # Download map tiles and create a GeoTIFF
    # tms_to_geotiff assumes WGS84 bbox and typically outputs in the tile source's projection (often Web Mercator)
    # or attempts to match the bbox. We will ensure it's WGS84 afterwards if needed, but samgeo
    # usually handles this well when vectorizing based on the source image's CRS derived from the bbox.
    log_message(f"🛰️ Downloading map tiles for ROI: {roi_coordinates} at zoom {zoom}...")
    tms_to_geotiff(output=image_file, bbox=roi_coordinates, zoom=zoom, source="Satellite", overwrite=True)
    log_message(f"✅ Satellite image saved to {image_file}")

    # --- Verify and potentially enforce CRS for the downloaded image ---
    try:
         with rasterio.open(image_file) as src:
             if src.crs is None:
                log_message(f"Warning: Downloaded image {image_file} has no CRS defined. Assuming EPSG:3857 (Web Mercator) based on tiling scheme.", "warning")
                # We'll handle vector CRS later
             elif src.crs.is_epsg_code and src.crs.to_epsg() == 3857:
                 log_message(f"ℹ️ Downloaded image CRS is EPSG:3857 (Web Mercator).")
             elif src.crs.is_epsg_code and src.crs.to_epsg() == 4326:
                 log_message(f"ℹ️ Downloaded image CRS is already EPSG:4326 (WGS84).")
             else:
                log_message(f"Warning: Downloaded image CRS is {src.crs}. Proceeding, but vector conversion might need reprojection.", "warning")
    except rasterio.RasterioIOError as e:
         log_message(f"Error opening downloaded GeoTIFF {image_file}: {e}", "error")
         return None # Cannot proceed without the image

    # Initialize SAM and segment the image
    log_message("🧠 Initializing SAM model...")
    # Add progress indication in Streamlit if possible
    progress_bar = None
    if 'st' in globals():
        progress_bar = st.progress(0, text="Initializing SAM...")

    sam = SamGeo(
        model_type="vit_h", # Consider vit_l or vit_b for faster processing if acceptable
        # checkpoint="/path/to/your/sam_vit_h_4b8939.pth", # Optional: Specify local checkpoint if needed
        sam_kwargs=None,
    )
    if progress_bar: progress_bar.progress(10, text="Generating segmentation mask...")

    log_message(f"🔪 Generating segmentation mask for {image_file}...")
    # sam.generate creates a mask based on the input image. It should preserve spatial reference.
    sam.generate(image_file, mask_file, batch=batch, foreground=True, erosion_kernel=erosion_kernel, mask_multiplier=mask_multiplier)
    log_message(f"✅ Segmentation mask saved to {mask_file}")
    if progress_bar: progress_bar.progress(50, text="Vectorizing segmentation mask...")


    # Polygonize the raster data
    log_message(f"🖋️ Converting mask {mask_file} to vector {geojson_file}...")
    # sam.tiff_to_vector should read the CRS from the mask_file (which comes from image_file)
    # and create a GeoJSON with that CRS.
    sam.tiff_to_vector(mask_file, geojson_file) # This saves the initial GeoJSON
    log_message(f"✅ Initial vector data saved to {geojson_file}")
    if progress_bar: progress_bar.progress(60, text="Loading vector data...")

    # Read the generated GeoJSON and ensure CRS is correct
    try:
        geojson = gpd.read_file(geojson_file)
        log_message(f"📊 Loaded {len(geojson)} segments from {geojson_file}")

        # --- Explicitly check and set CRS to EPSG:4326 ---
        if geojson.crs is None:
             log_message("Warning: GeoJSON loaded without CRS. Assuming original raster CRS and attempting to set to EPSG:4326.", "warning")
             # Try to get CRS from mask file as fallback
             try:
                 with rasterio.open(mask_file) as mask_src:
                     source_crs = mask_src.crs
                     if source_crs:
                          geojson.crs = source_crs
                          log_message(f"Inferred CRS from mask: {source_crs}. Reprojecting to EPSG:4326...")
                          geojson = geojson.to_crs(epsg=4326)
                     else:
                          log_message("Warning: Mask file also lacks CRS. Cannot reliably set CRS for GeoJSON. Output might be misplaced.", "warning")
                          # As a last resort, maybe assume WGS84 if the original bbox was? Risky.
                          # geojson.crs = target_crs_epsg4326 # Use with caution
             except Exception as crs_err:
                 log_message(f"Error reading CRS from mask file: {crs_err}", "error")
                 log_message("Cannot reliably set CRS for GeoJSON.", "warning")

        elif geojson.crs != target_crs_epsg4326:
            log_message(f"GeoJSON CRS is {geojson.crs}. Reprojecting to EPSG:4326...")
            geojson = geojson.to_crs(epsg=4326)
        else:
            log_message("GeoJSON CRS is already EPSG:4326.")

    except Exception as e:
        log_message(f"Error reading or setting CRS for GeoJSON file {geojson_file}: {e}", "error")
        if progress_bar: progress_bar.empty()
        return None # Cannot proceed without the GeoJSON

    # --- Process and Classify Each Segment ---
    llm_classifications = []
    llm_confidences = []
    llm_reasonings = []

    if segment_image_output:
        log_message(f"🖼️ Processing {len(geojson)} segments and saving PNGs to {segment_image_output}...")
        # Use tqdm for console, update progress bar for Streamlit
        segment_iterator = range(len(geojson))
        if 'st' not in globals(): # Use tqdm only if not in Streamlit
             from tqdm import tqdm
             segment_iterator = tqdm(geojson.iterrows(), total=len(geojson), desc="Processing Segments")
        else:
             segment_iterator = geojson.iterrows() # Simple iterator for Streamlit loop

        # Open the source image *once* outside the loop for efficiency
        try:
            with rasterio.open(image_file) as src:
                # Verify source CRS again just before cropping
                if src.crs is None:
                    log_message("Warning: Source image CRS is None during segment cropping.", "warning")
                # Ensure geojson is in the *same* CRS as the source image for masking
                geojson_for_masking = geojson.to_crs(src.crs)

                for idx, row in segment_iterator:
                    current_progress = 60 + int(40 * (idx + 1) / len(geojson))
                    if progress_bar: progress_bar.progress(current_progress, text=f"Processing segment {idx+1}/{len(geojson)}...")

                    segment_png_path = None
                    temp_tif = None # Define temp_tif here to ensure it's accessible in finally block
                    try:
                        # Get the geometry in the source image's CRS
                        geom_for_masking = geojson_for_masking.iloc[idx].geometry

                        # Ensure the geometry is valid
                        if not geom_for_masking.is_valid:
                            log_message(f"Warning: Segment {idx} has invalid geometry. Attempting to buffer.", "warning")
                            # Attempt to fix with a zero buffer
                            geom_buffered = geom_for_masking.buffer(0)
                            if not geom_buffered.is_valid or geom_buffered.is_empty:
                                 log_message(f"Error: Segment {idx} geometry could not be fixed. Skipping.", "error")
                                 if perform_llm_classification:
                                     llm_classifications.append("Skipped - Invalid Geometry")
                                     llm_confidences.append(0)
                                     llm_reasonings.append("Original geometry was invalid and could not be fixed.")
                                 continue # Skip to next segment
                            else:
                                geom_for_masking = geom_buffered


                        # Crop the raster using the segment's geometry
                        # Ensure geometry is mapped correctly if CRS mismatch occurred (though we try to enforce 4326)
                        out_image, out_transform = mask(src, [geom_for_masking], crop=True, nodata=0, filled=True) # Ensure background is consistent

                        # Check if the cropped image is valid (non-zero dimensions)
                        if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                            log_message(f"Warning: Segment {idx} resulted in an empty image after cropping (geometry might be too small or outside raster bounds). Skipping.", "warning")
                            if perform_llm_classification:
                                llm_classifications.append("Skipped - Empty Crop")
                                llm_confidences.append(0)
                                llm_reasonings.append("Segment geometry resulted in an empty image after mask/crop.")
                            continue # Skip to the next segment

                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "nodata": 0, # Explicitly set nodata
                            "crs": src.crs # Preserve the CRS of the source image for the temp TIF
                        })

                        # Save the cropped raster as a temporary GeoTIFF
                        temp_tif = os.path.join(segment_image_output, f"temp_segment_{idx}.tif")
                        with rasterio.open(temp_tif, "w", **out_meta) as dest:
                            dest.write(out_image)

                        # Convert the GeoTIFF to PNG
                        with rasterio.open(temp_tif) as temp_src:
                            # Read only the first 3 bands (assuming RGB)
                            # Handle cases where source might have < 3 bands (e.g., grayscale)
                            num_bands = min(temp_src.count, 3)
                            segment_image_data = temp_src.read(list(range(1, num_bands + 1)))
                            if num_bands < 3:
                                # If less than 3 bands, duplicate the last band to make it RGB for PIL
                                log_message(f"Warning: Segment {idx} source has only {num_bands} bands. Creating RGB PNG.", "warning")
                                while segment_image_data.shape[0] < 3:
                                     segment_image_data = np.vstack([segment_image_data, segment_image_data[-1:, :, :]])


                        # Transpose to (height, width, channels) for PIL
                        segment_image_data = np.transpose(segment_image_data, (1, 2, 0))

                        # Ensure data is in uint8 format for PIL
                        if segment_image_data.dtype != np.uint8:
                             # Basic scaling if not uint8 (might need adjustment based on actual data range)
                             if np.max(segment_image_data) > 0: # Avoid division by zero if image is all black
                                 max_val = np.max(segment_image_data)
                                 if max_val > 255:
                                      segment_image_data = (segment_image_data / max_val * 255).astype(np.uint8)
                                 else:
                                      segment_image_data = segment_image_data.astype(np.uint8)
                             else:
                                segment_image_data = segment_image_data.astype(np.uint8)


                        img = Image.fromarray(segment_image_data, 'RGB')
                        segment_png_path = os.path.join(segment_image_output, f"segment_{idx}.png")
                        img.save(segment_png_path)


                        # --- Perform LLM Classification ---
                        if perform_llm_classification and segment_png_path:
                            llm_result = call_llm(segment_png_path)
                            if llm_result:
                                llm_classifications.append(llm_result.get('classification', 'LLM Error - Key Missing'))
                                llm_confidences.append(llm_result.get('confidence', 0))
                                llm_reasonings.append(llm_result.get('reasoning', 'LLM Error - Key Missing'))
                            else:
                                llm_classifications.append("LLM Call Failed")
                                llm_confidences.append(0)
                                llm_reasonings.append("LLM API call or processing failed.")
                        elif perform_llm_classification: # Case where PNG path wasn't set (should not happen if logic is correct)
                             llm_classifications.append("Skipped - No PNG")
                             llm_confidences.append(0)
                             llm_reasonings.append("PNG file was not generated for this segment.")


                    except ValueError as ve:
                         # Catch specific rasterio mask errors like "Input shapes do not overlap raster."
                         log_message(f"Warning: Skipping segment {idx} due to geometry/mask error: {ve}", "warning")
                         if perform_llm_classification:
                             llm_classifications.append("Skipped - Geometry/Mask Error")
                             llm_confidences.append(0)
                             llm_reasonings.append(f"Rasterio mask error: {ve}")
                    except rasterio.errors.RasterioIOError as rio_err:
                         log_message(f"Warning: Skipping segment {idx} due to Rasterio IO error: {rio_err}", "warning")
                         if perform_llm_classification:
                             llm_classifications.append("Skipped - Rasterio Error")
                             llm_confidences.append(0)
                             llm_reasonings.append(f"Rasterio IO error: {rio_err}")
                    except Exception as e:
                        log_message(f"Error processing segment {idx}: {type(e).__name__} - {e}", "error")
                        # Ensure lists stay synchronized even if an error occurs mid-segment processing
                        if perform_llm_classification and len(llm_classifications) < (idx + 1):
                             llm_classifications.append("Processing Error")
                             llm_confidences.append(0)
                             llm_reasonings.append(f"Error during segment processing: {e}")
                    finally:
                        # Remove the temporary GeoTIFF file if it exists
                        if temp_tif and os.path.exists(temp_tif):
                            try:
                                os.remove(temp_tif)
                            except OSError as ose:
                                log_message(f"Warning: Could not remove temporary file {temp_tif}: {ose}", "warning")

        except rasterio.RasterioIOError as e:
             log_message(f"FATAL ERROR: Could not open source image {image_file}. Cannot process segments. Error: {e}", "error")
             if progress_bar: progress_bar.empty()
             return None # Cannot proceed if source image isn't readable
        except Exception as e:
             log_message(f"FATAL ERROR: An unexpected error occurred before segment processing loop: {e}", "error")
             if progress_bar: progress_bar.empty()
             return None


        log_message("✅ Finished processing segments.")
        if progress_bar: progress_bar.progress(100, text="Finalizing results...")


        # --- Add LLM results to GeoDataFrame ---
        if perform_llm_classification:
            log_message("✍️ Adding LLM classification results to GeoDataFrame...")
            # Ensure the lengths match before assigning columns
            if len(llm_classifications) == len(geojson):
                # Ensure the target GeoDataFrame ('geojson') still has the correct index
                # if rows were skipped. A simple assignment should work if index wasn't altered.
                geojson['llm_class'] = llm_classifications
                geojson['llm_conf'] = llm_confidences
                geojson['llm_reason'] = llm_reasonings
            else:
                log_message(f"Error: Mismatch between number of segments ({len(geojson)}) and LLM results ({len(llm_classifications)}). LLM results not added correctly.", "error")
                # Pad with errors to avoid crashing, but flag it clearly
                pad_len = len(geojson) - len(llm_classifications)
                if pad_len > 0:
                    llm_classifications.extend(["Error - Mismatch"] * pad_len)
                    llm_confidences.extend([0] * pad_len)
                    llm_reasonings.extend(["Result list length mismatch"] * pad_len)
                # Try assigning again after padding (might still fail if index is weird)
                try:
                    geojson['llm_class'] = llm_classifications[:len(geojson)] # Truncate just in case
                    geojson['llm_conf'] = llm_confidences[:len(geojson)]
                    geojson['llm_reason'] = llm_reasonings[:len(geojson)]
                    log_message("Padded LLM results due to mismatch. Review output carefully.", "warning")
                except Exception as assign_err:
                     log_message(f"Error assigning padded LLM results: {assign_err}", "error")


    elif perform_llm_classification:
         log_message("LLM classification requested, but segment PNGs were not generated (`segment_image_output` not set).")


    # --- Save the final GeoJSON (with or without LLM results) ---
    # Ensure the CRS is correctly set before saving (already done above, but double check)
    if geojson.crs != target_crs_epsg4326:
        log_message(f"Warning: Final GeoDataFrame CRS is {geojson.crs}. Attempting to re-project to EPSG:4326 before saving.", "warning")
        try:
            geojson = geojson.to_crs(epsg=4326)
        except Exception as reproject_err:
             log_message(f"Error re-projecting final GeoJSON: {reproject_err}. Saving with original CRS.", "error")


    try:
        geojson.to_file(geojson_file, driver='GeoJSON')
        # # Overwrite CRS field in GeoJSON file (optional, geopandas usually handles this)
        # try:
        #     with open(geojson_file, 'r') as f:
        #         gj_content = json.load(f)
        #     if 'crs' not in gj_content or gj_content.get('crs', {}).get('properties', {}).get('name') != 'urn:ogc:def:crs:EPSG::4326':
        #          gj_content['crs'] = { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::4326" } }
        #          with open(geojson_file, 'w') as f:
        #              json.dump(gj_content, f, indent=2) # Add indent for readability
        # except Exception as crs_write_err:
        #      log_message(f"Warning: Could not explicitly write EPSG:4326 CRS to GeoJSON file: {crs_write_err}", "warning")

        log_message(f"💾 Final GeoJSON (CRS: {geojson.crs}) saved to {geojson_file}")
    except Exception as e:
        log_message(f"Error saving final GeoJSON to {geojson_file}: {e}", "error")


    log_message(f"\n✅ Script finished. Main outputs are in {output_folder}.")
    if segment_image_output:
        log_message(f"Segment PNGs are in {segment_image_output}.")
    log_message(f"Final GeoJSON: {geojson_file}")
    if perform_llm_classification and 'llm_class' in geojson.columns:
        log_message("GeoJSON includes LLM classification columns.")
    elif perform_llm_classification:
        log_message("GeoJSON does *not* include LLM columns, likely due to processing errors or length mismatch.", "warning")

    if progress_bar: progress_bar.empty() # Remove progress bar on completion

    return geojson


# --- Streamlit Application Code (From new_app.py) ---

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
DEFAULT_CENTER = [8.62477, 78.03046] # Center near the example ROI from new_sam.py [Lat, Lon]
DEFAULT_ZOOM = 18
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENT_IMG_DIR, exist_ok=True)
GEOJSON_RESULTS_FILE = os.path.join(OUTPUT_DIR, "segmented_parcels.geojson")

# --- Check for prerequisites ---
if llm_prompt is None: # Check the loaded prompt status from above
    st.warning("⚠️ **Warning:** `sam_prompt.txt` not found or failed to load. LLM classification might be disabled or use default settings.")
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
    # Use a consistent key to avoid map resets unless forced by map_key change
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
                status.write("Fetching satellite imagery for the ROI...") # Use status.write
                # (Assuming the backend function downloads the image first)
                # Call the main function from current_sam.py (now defined in this file)
                # Pass the specific output file path for clarity
                gdf_results = segment_image_with_sam( # No current_sam prefix needed
                    output_folder=OUTPUT_DIR,
                    roi_coordinates=st.session_state.roi_bounds,
                    zoom=DEFAULT_ZOOM, # Consider making this an advanced option
                    segment_image_output=SEGMENT_IMG_DIR,
                    perform_llm_classification=True, # Explicitly request classification
                )
                # Status updates are now handled inside segment_image_with_sam using st.write
                # This section is simplified as the function handles logging/status internally

                end_time = time.time()
                processing_time = end_time - start_time

                # Check if the function returned successfully (even if GDF is empty)
                # and if the expected output file exists
                if gdf_results is not None and os.path.exists(GEOJSON_RESULTS_FILE):
                    status.update(label=f"✅ Analysis Complete! ({processing_time:.2f}s)", state="complete", expanded=False)
                    st.session_state.geojson_path = GEOJSON_RESULTS_FILE
                    st.session_state.processing_complete = True
                    st.session_state.map_key += 1 # Force results map redraw
                    st.balloons() # Fun success indicator
                elif gdf_results is None:
                     # gdf_results is None indicates a fatal error within the function
                     st.error(f"❌ Processing failed internally. Check logs/console for details.")
                     status.update(label="Processing Failed (Internal Error)", state="error", expanded=True)
                     st.session_state.processing_complete = False
                else:
                    # gdf_results might be an empty GeoDataFrame, but the file wasn't found
                    st.error(f"❌ Processing finished, but the output file was not found: {GEOJSON_RESULTS_FILE}")
                    status.update(label="Processing Finished (Output File Missing)", state="error", expanded=True)
                    st.session_state.processing_complete = False

            except ImportError as ie:
                 st.error(f"❌ Import Error during processing: {ie}. Ensure all required libraries (geopandas, rasterio, samgeo, google-generativeai, etc.) are installed.")
                 status.update(label="Processing Failed (Import Error)", state="error")
                 st.session_state.processing_complete = False
            except FileNotFoundError as fnf:
                 st.error(f"❌ File Not Found Error during processing: {fnf}. Check paths and prerequisite files like `sam_prompt.txt`.")
                 status.update(label="Processing Failed (File Not Found)", state="error")
                 st.session_state.processing_complete = False
            except Exception as e:
                st.error(f"❌ An unexpected error occurred during processing: {type(e).__name__} - {e}")
                status.update(label="Processing Failed", state="error")
                import traceback
                st.error(traceback.format_exc()) # Show detailed traceback in Streamlit for debugging
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
            # Ensure data is in web-friendly projection (EPSG:4326)
            if gdf.crs is None:
                 st.warning("⚠️ Results GeoJSON has no CRS defined. Assuming EPSG:4326 for display, but review the data source.")
                 gdf.crs = "EPSG:4326" # Assume WGS84 for display
            elif str(gdf.crs).upper() != "EPSG:4326":
                 st.info(f"Results CRS is {gdf.crs}. Reprojecting to EPSG:4326 for map display.")
                 try:
                     gdf = gdf.to_crs("EPSG:4326")
                 except Exception as reproj_err:
                      st.error(f"❌ Failed to reproject results to EPSG:4326 for display: {reproj_err}. Map might be incorrect.")


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
                # Check for column existence case-insensitively, but use original case if found
                gdf_cols_lower = {col.lower(): col for col in gdf.columns}
                llm_class_col = gdf_cols_lower.get('llm_class')
                llm_conf_col = gdf_cols_lower.get('llm_conf')
                llm_reason_col = gdf_cols_lower.get('llm_reason')

                if llm_class_col: tooltip_fields.append(llm_class_col)
                if llm_conf_col: tooltip_fields.append(llm_conf_col)
                if llm_reason_col: tooltip_fields.append(llm_reason_col)


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
                        max_width=250, # *** Adjusted parameter for width control (pixels) *** was 40 before
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
                     # Add a small buffer to bounds if they are valid
                     if len(bounds) == 4 and None not in bounds and bounds[0] < bounds[2] and bounds[1] < bounds[3]:
                         results_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                     else:
                          raise ValueError("Invalid bounds calculated")
                except Exception as fit_e:
                     st.warning(f"Could not automatically zoom to layer bounds ({fit_e}). Using ROI center.")
                     if st.session_state.roi_bounds:
                         center_lon = (st.session_state.roi_bounds[0] + st.session_state.roi_bounds[2]) / 2
                         center_lat = (st.session_state.roi_bounds[1] + st.session_state.roi_bounds[3]) / 2
                         results_map.location = [center_lat, center_lon]
                         results_map.zoom_start = DEFAULT_ZOOM # Use configured zoom


                # Display the results map
                # Use a unique key based on map_key to ensure redraw after processing
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
                 st.warning("Processing completed, but the resulting GeoJSON file is empty or contains no valid features.")


        except FileNotFoundError:
            st.error(f"❌ Error: Could not find the results file: {st.session_state.geojson_path}")
            st.session_state.processing_complete = False # Reset state
        except ImportError:
            st.error("❌ Error: Geopandas might not be installed correctly or is missing.")
            st.session_state.processing_complete = False # Reset state
        except Exception as e:
            st.error(f"❌ An error occurred displaying the results: {type(e).__name__} - {e}")
            import traceback
            st.error(traceback.format_exc()) # Uncomment for detailed debugging
            st.session_state.processing_complete = False # Reset state

    elif st.session_state.roi_bounds and not st.session_state.processing_complete and not confirm_button:
         # Case where ROI is selected, but processing hasn't run or failed previously
         st.info("Click the 'Confirm ROI and Start Processing' button above to generate results.")
    elif not st.session_state.roi_bounds:
        # Case where no ROI has been selected yet
        st.info("Draw a rectangle on the first map and confirm it to see results here.")

# --- End of Streamlit App ---
