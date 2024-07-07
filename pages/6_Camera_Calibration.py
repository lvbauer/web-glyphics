import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

st.set_page_config(page_title="Camera Calibration Tool")

def main():

    st.title("Camera Calibration Tool")

    with st.expander("About & Downloads"):
        st.markdown("Adapted from [this OpenCV publication](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).")

        st.write("To use, print this chessboard and take a series of pictures with the chessboard present.")

        with open("assets/chessboard.pdf", "rb") as f:
            pdf_bytes = f.read()

        st.download_button("Chessboard PDF", data=pdf_bytes, file_name="calibration_chessboard.pdf", mime='application/octet-stream')

    calc_tab, corr_tab = st.tabs(["Calculate Correction", "Correct Images"])

    with calc_tab:
        st.subheader("Chessboard Image Upload")
        st.write("Upload images containing checkerboard below.")

        user_image_upload = st.file_uploader("Chessboard Reference Images:", type=["png","jpg","tiff", "tif","jpeg"], 
                                            accept_multiple_files=True, key="file_uploader")

        if (user_image_upload is not None):
            user_image_list = user_image_upload

        st.subheader("Options")

        options_col1, options_col2 = st.columns(2)
        
        with options_col1:
            num_squares_x = int(st.number_input("Horizontal Number of Squares", min_value=1, step=1, value=6))
        with options_col2:
            num_squares_y = int(st.number_input("Vertical Number of Squares", min_value=1, step=1, value=9))

        st.subheader("Find Reference Points")

        show_pattern_bool = st.checkbox("Show Annotated Chessboards")

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prep object points
        objp = np.zeros((num_squares_x*num_squares_y,3), np.float32)
        objp[:,:2] = np.mgrid[0:num_squares_y,0:num_squares_x].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # analyze uploaded images loop
        
        if (len(user_image_list) > 0):
            for img_idx, img in enumerate(user_image_list):
                img_open = Image.open(img)
                img_arr = np.array(img_open)
                
                if (len(img_arr.shape) > 2):
                    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_arr

                ret, corners = cv2.findChessboardCorners(img_gray, (num_squares_y, num_squares_x), None)

                if (ret == True):
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners2)
                
                    if (show_pattern_bool):
                        annotated_img_arr = cv2.drawChessboardCorners(img_arr, (num_squares_y,num_squares_x), corners2, ret)
                        st.write(f"Reference Image {img_idx}: {img.name}")
                        st.image(annotated_img_arr)
        
            # perform calibration step
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

            st.session_state.mtx = mtx
            st.session_state.dist = dist

            mtx_dl_col, dist_dl_col = st.columns(2)

            st.subheader("Download Calibration Files")
            with mtx_dl_col:
                arr_download_button(mtx, button_text="Download MTX File", file_name="image_correction_mtx.npy")
            with dist_dl_col:
                arr_download_button(dist, button_text="Download DIST File", file_name="image_correction_dist.npy")

    with corr_tab:

        st.subheader("Upload Images for Correction")
        user_image_upload_corr = st.file_uploader("Images to Correct:", type=["png","jpg","tiff", "tif","jpeg"], 
            accept_multiple_files=True, key="file_uploader_corr")

        # Calibration files inputs
        st.subheader("Upload Correction Files")
        st.write("Upload correction files saved from previous calibration. If calibration values were calculated during the same session, they will be carried over from the 'Calculate Correction' menu.")

        mtx_upload_col, dist_upload_col = st.columns(2)
        with mtx_upload_col:
            user_mtx_upload = st.file_uploader("Upload MTX File", type=["npy"], 
                                               accept_multiple_files=False, key="user_mtx_upload")
        with dist_upload_col:
            user_dist_upload = st.file_uploader("Upload DIST File", type=["npy"], 
                                               accept_multiple_files=False, key="user_dist_upload")

        if (user_mtx_upload is not None) and (user_dist_upload is not None):
            working_mtx = np.frombuffer(user_mtx_upload.getbuffer(), dtype=np.float64)
            working_mtx = working_mtx[-9:].reshape((3,3))

            working_dist = np.frombuffer(user_dist_upload.getbuffer(), dtype=np.float64)
            working_dist = working_dist[-5:].reshape((1,5))
        
        elif ("mtx" in st.session_state) and ("dist" in st.session_state):
            working_mtx = st.session_state.mtx
            working_dist = st.session_state.dist

        else:
            working_mtx = None
            working_dist = None
            st.info("Upload MTX and DIST files or calibrate from images to continue.")

        with st.expander("View Matricies"):
            if (working_mtx is None) or (working_dist is None):
                st.write("No data found.")
            else:
                st.write("Camera Matrix (MTX)")
                working_mtx
                st.write("Distortion Coefficients (DIST)")
                working_dist
                st.write("Shapes & Types")
                (working_mtx.dtype, working_dist.dtype)
                (working_mtx.shape, working_dist.shape)
                st.write("No data found.")


        corrected_images_list = None
        if (len(user_image_upload_corr) > 0) and (working_mtx is not None) and (working_dist is not None):
            st.subheader("Corrected Images")
            corrected_images_list = []
            for img_idx, img in enumerate(user_image_upload_corr):
                print(img.name)
                img_open = Image.open(img)
                img_arr = np.array(img_open)
                
                h, w = img_arr.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

                dst = cv2.undistort(img_arr, mtx, dist, None, newcameramtx)

                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]

                corrected_images_list.append(dst)

                st.write(f"Corrected Image {img_idx}: {img.name}")
                st.image(dst)
                download_arr_img(dst, img.name, button_text=f"Download Corrected Image: {img.name}")


def arr_download_button(arr, button_text="Download", file_name="array.npy"):
    """From here: https://discuss.streamlit.io/t/downloading-3d-numpy-arrays/22079/2"""
    with BytesIO() as buffer:
        # Write array to buffer
        np.save(buffer, arr)
        btn = st.download_button(
            label=button_text,
            data = buffer, # Download buffer
            file_name = file_name
        )

def download_arr_img(arr, file_name, button_text="Download Full-Sized Image"):
    im_pil = Image.fromarray(arr)
    buf = BytesIO()

    format_val = file_name.split(".")[1]

    if (format_val.upper() == "JPG"):
        format_val = "JPEG"
    elif (format_val.upper() == "TIF"):
        format_val = "TIFF"

    im_pil.save(buf, format=format_val)
    bytes_img = buf.getvalue()
    st.download_button(
        label=button_text, 
        data=bytes_img, 
        file_name=image_name_corrected(file_name),
        mime="image/jpeg"
        )  

def image_name_corrected(name):
    trim_name = name.split(".")
    img_name = trim_name[0] + f"_corrected.{trim_name[1].lower()}"
    return img_name

if __name__ == "__main__":
    main()