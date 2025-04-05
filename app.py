import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime
from mtcnn import MTCNN
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import re
from model_loader import model
from detection_loader import *
from datetime import datetime, time, timedelta
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# Function to set up Google Sheets API
@st.cache_resource
def setup_google_sheets():
    SERVICE_ACCOUNT_FILE = 'crypto-argon-445314-r7-f9f259c73941.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=credentials)
    return service

# Google Sheets Setup
service = setup_google_sheets()
SPREADSHEET_ID = '1gaHG-rZ7xsYsqxvZaeh3OsXcF3FoyiYxs82rB_9Ndws'
RANGE_NAME = 'Student Attendance!A:H'

# Example usage (add your logic here)
sheet = service.spreadsheets()

def save_schedule():
    print('Click!')

# Admin authentication from Google Sheets
def authenticate_admin(username, password):
    # Google Sheets API Setup
    SERVICE_ACCOUNT_FILE = 'crypto-argon-445314-r7-f9f259c73941.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    SPREADSHEET_ID = '1gaHG-rZ7xsYsqxvZaeh3OsXcF3FoyiYxs82rB_9Ndws'
    RANGE_NAME = 'Account!A:H'

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=credentials)
    sheet = service.spreadsheets()

    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        return False  # No data found in the sheet

    # Loop through the rows to check for matching username and password
    for row in values:
        if len(row) >= 2:
            if username == row[1] and password == row[2]:
                # Store instructor details in session state
                if 'instructor_fname' not in st.session_state:
                    st.session_state.instructor_fname = row[0]
                if 'instructor_name' not in st.session_state:  
                    st.session_state.instructor_name = row[1]
                if 'instructor_password' not in st.session_state:
                    st.session_state.instructor_password = row[2]
                if 'instructor_subject' not in st.session_state:
                    st.session_state.instructor_subject = row[3]
                if 'instructor_section' not in st.session_state:
                    st.session_state.instructor_section = row[4]
                if 'instructor_class_start' not in st.session_state:
                    st.session_state.instructor_class_start = row[5]
                if 'instructor_class_end' not in st.session_state:
                    st.session_state.instructor_class_end = row[6]
                if 'instructor_class_attendance' not in st.session_state:
                    st.session_state.instructor_class_attendance = row[7]
                return True

    return False  # Invalid credentials if no match found
        
# Recognize a person using cosine similarity
def preprocess(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [100, 100])  # Resize to the required size
    image = image / 255.0  # Normalize pixel values
    return image

def verify_multi_class(detection_threshold, verification_threshold):

    # Build results dictionary to store scores and detection rates for each class
    class_scores = {}
    verified_person = 'Unknown'
    results = {}

    # Preprocess the input image once
    input_img = preprocess(os.path.join('One Shot Face Recognition', 'application_data', 'input_image', 'input_image.jpg'))
    
    # Iterate through each class folder in 'verification_images'
    for person_folder in os.listdir(os.path.join('One Shot Face Recognition', 'application_data', 'verification_images')):
        folder_path = os.path.join('One Shot Face Recognition', 'application_data', 'verification_images', person_folder)
        
        # Skip if not a folder
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Evaluating folder: {person_folder}")
        vault = []
        
        # Iterate through each image in the current class folder
        for image in os.listdir(folder_path):
            validation_img = preprocess(os.path.join(folder_path, image))
            
            # Make predictions with the model (using both the input image and validation image)
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            vault.append(result)

        print(vault)

        # Detection: Count how many predictions exceed the detection threshold
        detections = np.sum(np.squeeze(vault) > detection_threshold)

        print(f'DETECTION THRESH: {detections}')

        # Verification: Proportion of positive predictions for this class
        verification_score = detections / len(vault)

        print(f'VERIFICATION THRESH: {verification_score}')

        # Store model prediction results for this class
        results[person_folder] = vault

        # Store the verification score for this class
        class_scores[person_folder] = verification_score

    # Find the best match based on verification score
    best_match = max(class_scores, key=class_scores.get)
    best_score = class_scores[best_match]
    
    # Check if the best score exceeds the verification threshold
    epsilon = 1e-6
    if best_score >= (verification_threshold - epsilon):
        verified_person = best_match

    return results, best_score, verified_person

# Record attendance to Google Sheets
def record_attendance(student_id, name, section, subject, instructor):
    now = datetime.now()
    time_in = now.strftime('%H:%M:%S')
    date = now.strftime('%Y-%m-%d')

    # Convert session times to datetime objects for accurate comparison
    class_start_time = datetime.strptime(st.session_state.instructor_class_start, '%I:%M %p')
    class_end_time = datetime.strptime(st.session_state.instructor_class_end, '%I:%M %p')
    class_attendance_end = datetime.strptime(st.session_state.instructor_class_attendance, '%I:%M %p')

    # Add one day if the class time crosses midnight (i.e., class starts at night and ends after midnight)
    if class_start_time > class_end_time:
        class_end_time += timedelta(days=1)

    # Get the current time as a datetime object
    current_time = now

    # Determine the status based on time comparisons
    if class_start_time <= current_time <= class_attendance_end:
        status = "Present"
    elif class_start_time < current_time <= class_end_time:
        status = "Late"
    else:
        status = "Absent"

    # Prepare the row for Google Sheets
    row = [student_id, name, section, subject, instructor, date, time_in, status]

    # Append the row to Google Sheets
    sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption='USER_ENTERED',
        body={'values': [row]}
    ).execute()



# Function to get data from the Google Sheets
def get_attendance_data():
    # Google Sheets API Setup
    SERVICE_ACCOUNT_FILE = 'crypto-argon-445314-r7-f9f259c73941.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    SPREADSHEET_ID = '1gaHG-rZ7xsYsqxvZaeh3OsXcF3FoyiYxs82rB_9Ndws'
    RANGE_NAME = 'Student Attendance!A:H'

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=credentials)
    sheet = service.spreadsheets()

    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])
    return values


# Function to process the captured image and save it to the appropriate folder
def process_the_image(captured_images, label_name):
    # Extract Student Last Name and First Name using regex
    match = re.search(r'(\d+)\s+([\w]+)\s+([\w]+)', label_name)
    if not match:
        st.error("Invalid input format. Please provide Student ID, Last Name, and First Name.")
        return
    
    student_id = match.group(1)
    last_name = match.group(2)
    first_name = match.group(3)
    
    # Create the directory path
    folder_name = f"{student_id} {last_name} {first_name}"
    save_dir = os.path.join("One Shot Face Recognition", "application_data", "verification_images", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # Process and save images
    for i, img in enumerate(captured_images[:50]):  # Limit to 50 images
        faces = detector.detect_faces(img)

        for face in faces:
            x, y, width, height = face['box']
            confidence = face['confidence']

            if confidence >= 0.90:
                # Crop and resize the face
                x, y, width, height = max(0, x), max(0, y), max(0, width), max(0, height)
                cropped_face = img[y:y+height, x:x+width]
                if cropped_face.size > 0:
                    resized_face = cv2.resize(cropped_face, (100, 100))
                    img_path = os.path.join(save_dir, f"original_{i+1}.jpg")
                    cv2.imwrite(img_path, resized_face)
                    st.success(f"Saved: {img_path}")
                else:
                    st.warning(f"Skipped invalid crop for image {i+1}.")
    st.success(f"Processing completed. Images saved in {save_dir}.")


def parse_time(time_str):
    if isinstance(time_str, time):  # If it's already a time object, return as is
        return time_str
    try:
        return datetime.strptime(time_str, "%I:%M %p").time()  # Convert '1:10 PM' to time object
    except ValueError:
        return datetime.strptime(time_str, "%H:%M").time()  # Convert '13:10' to time object


# Set the path or URL of your background image
background_image_url = "https://images.hdqwalls.com/wallpapers/glowing-eye-ap.jpg"
# Add CSS to set the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;  /* Cover the whole page */
        background-position: center;  /* Center the image */
        background-repeat: no-repeat;  /* Prevent repeating the image */
        height: 100vh;  /* Full height */
        padding: 20px;  /* Add some padding */
        color: white;  /* Optional: Set default text color */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for login
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# Initialize session state if not set
if 'action' not in st.session_state:
    st.session_state['action'] = None

# If the user is not logged in, show the login system
if not st.session_state.is_admin:
    # Title and Subtitle
    st.write('<h1 style="text-align: center; color: #4CAF50; font-size: 70px; font-weight: bold;">Smart-Gaze</h1>', unsafe_allow_html=True)
    st.write('<h2 style="text-align: center; color: #777; font-size: 42px; margin-bottom: 40px;">Attendance System for Data Science Students in USTP</h2>', unsafe_allow_html=True)

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Create a login button
    login_button = st.button("Login")

    # Check if the login button is pressed
    if login_button:
        if authenticate_admin(username, password):
            st.session_state.is_admin = True
            st.success("Login successful!")
            st.rerun()  # Refresh the page to hide the login system and show the next content
        else:
            st.error("Invalid credentials. Please try again.")

else:
    # Hide login section and show sidebar categories for authenticated admin
    st.write('<h1 style = "text-align: center; color: #4CAF50; font-size: 36px; font-weight: bold;" >Smart-Gaze</h1>', unsafe_allow_html=True)
    st.write('<h2 style = "text-align: center; color: #777; font-size: 24px; margin-bottom: 40px;">Attendance System for Data Science Students in USTP</h2>', unsafe_allow_html=True)


    # Sidebar navigation
    st.sidebar.title("Admin Dashboard")
    menu = st.sidebar.radio("Choose an option", ("Dashboard", "Add a Student", "Record Attendance"))

    # Function to filter data based on the selected date, subject, and section
    def filter_data(df, selected_date, selected_subject, selected_section):
        # Filter by date, subject, and section
        df_filtered = df[df['Date'] == selected_date]
        if selected_subject:
            df_filtered = df_filtered[df_filtered['Subject'] == selected_subject]
        if selected_section:
            df_filtered = df_filtered[df_filtered['Section'] == selected_section]
        return df_filtered

    # Dashboard
    if menu == 'Dashboard':
        ins_fname = st.session_state.instructor_fname

        # Assuming parse_time is already defined and works
        # Convert stored times into `datetime.time` objects
        class_start_time = parse_time(st.session_state.instructor_class_start)
        class_end_time = parse_time(st.session_state.instructor_class_end)
        class_attendance_time = parse_time(st.session_state.instructor_class_attendance)

        # Format times as 'HH:MM' (24-hour format) for HTML input fields
        class_start_formatted = class_start_time.strftime('%H:%M')
        class_end_formatted = class_end_time.strftime('%H:%M')
        class_attendance_formatted = class_attendance_time.strftime('%H:%M')

        st.components.v1.html(f'''
            <h1 style="text-align:center; font-size:32px; font-weight:bold; color:white;">DASHBOARD SYSTEM</h1>
            <div style="float:right; background-color:rgb(146, 141, 141); border-radius: 30px; padding: 10px;">
                <h4 for="cls-start" style='text-align:center; font-weight:900; color:white;'>CLASS SCHEDULE</h4>
                <p style="text-align:center; margin-bottom:0px; font-size:11px; color:white;" for="cls-start">CLASS START</p>
                <div id="cls-start" style="float: right; margin-bottom:10px;">
                    <input type="time" id="cls-start01" value="{class_start_formatted}" style="width:250px; border-radius: 10px;">
                </div>
                <p style="text-align:center; margin-bottom:0px; font-size:11px; color:white;" for="cls-end">CLASS END</p>
                <div id="cls-end" style="float: right; margin-bottom:10px;">
                    <input type="time" id="cls-start02" value="{class_end_formatted}" style="width:250px; border-radius: 10px;">
                </div>
                <p style="text-align:center; margin-bottom:0px; font-size:11px; color:white;" for="cls-att-end">CLASS ATTENDANCE END</p>
                <div id="cls-att-end" style="float: right; margin-bottom:10px;">
                    <input type="time" id="cls-start03" value="{class_attendance_formatted}" style="width:250px; border-radius: 10px;">
                </div>
                <div style="text-align:center; margin-top:20px;">
                    <button id="save-button" style="align-items:center; margin-top:10px;">save</button>
                    <span id="done-text" style="color: rgb(83, 245, 50); font-weight: bold; margin-left: 10px; display: none;">Done!</span>
                </div>
            </div>

            <script>
                // Function to send data via URL query parameters
                function sendToURL(data) {{
                    const queryString = new URLSearchParams({{
                        action: 'save_class_schedule',
                        classStart: data.classStart,
                        classEnd: data.classEnd,
                        attendanceEnd: data.attendanceEnd,
                        insFname: data.insFname
                    }}).toString();

                    // Update the URL with query parameters (without reloading)
                    window.parent.history.pushState({{}}, '', '?' + queryString);
                }}

                document.getElementById('save-button').onclick = function() {{
                    let classStart = document.getElementById('cls-start01').value;
                    let classEnd = document.getElementById('cls-start02').value;
                    let attendanceEnd = document.getElementById('cls-start03').value;

                    var data = {{
                        action: 'save_class_schedule',
                        classStart: classStart,
                        classEnd: classEnd,
                        attendanceEnd: attendanceEnd,
                        insFname: 'Tony Stark'
                    }};
                    
                    sendToURL(data);

                    // Show the "Done" text next to the button
                    var doneText = document.getElementById('done-text');
                    doneText.style.display = 'inline';

                    // Fade out the "Done" text after 3 seconds
                    setTimeout(function() {{
                        doneText.style.opacity = 0;
                        setTimeout(function() {{
                            doneText.style.display = 'none';
                            doneText.style.opacity = 1;  // Reset opacity for the next time
                        }}, 1000);
                    }}, 3000); // 3 seconds delay before fading out
                }};
            </script>
        ''', height=400)


        # Check if the URL has changed and trigger the button press in Streamlit
        query_params = st.experimental_get_query_params()

        if st.button("UPDATE SCHEDULE"):
            # Add the logic that should run when the button is clicked
            st.write("Class Schedule saved successfully!")
                
        # Display Instructor Name
        st.write(f'''
            <div style="text-align: left; display: flex; align-items: center;">
                <p style="color: #4CAF50; font-size: 16px; font-weight: bold; margin-right: 10px;">
                    Instructor Name:
                </p>
                <p style="font-size: 32px; font-weight: bold; margin: 0;">
                    {ins_fname}
                </p>
            </div>
        ''', unsafe_allow_html=True)

        data = get_attendance_data()

        # Convert to a DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])  # The first row contains the column names

        # Filter the data for the specific instructor
        df_filtered = df[df['Instructor'] == ins_fname]

        # Allow the instructor to select a date, subject, and section
        date_picker = st.date_input("Select a Date", min_value=datetime(2023, 1, 1), max_value=datetime.today())
        selected_date = date_picker.strftime('%Y-%m-%d')  # Format date as string

        # Get available subjects and sections for the instructor
        available_subjects = df_filtered['Subject'].unique()
        available_sections = df_filtered['Section'].unique()

        # Subject and Section filters
        selected_subject = st.selectbox("Select Subject", options=[""] + list(available_subjects))
        selected_section = st.selectbox("Select Section", options=[""] + list(available_sections))

        # Filter the data based on the selected filters
        df_filtered_date = filter_data(df_filtered, selected_date, selected_subject, selected_section)

        # If data is available for the selected filters
        if not df_filtered_date.empty:
            # Create columns for statistics and charts
            col1, col2 = st.columns([2, 3])  # Adjust the ratio of columns

            with col1:
                st.write(f"### Attendance Records for {selected_date}")
                st.dataframe(df_filtered_date)

                # Calculate Attendance Statistics (Present, Absent, Late)
                attendance_counts = df_filtered_date['Present Status'].value_counts()

                # Show statistics in a compact and clear way
                st.write(f"### Attendance Summary")
                st.write(f"**Present**: {attendance_counts.get('Present', 0)}")
                st.write(f"**Absent**: {attendance_counts.get('Absent', 0)}")
                st.write(f"**Late**: {attendance_counts.get('Late', 0)}")

            with col2:
                # Pie chart for attendance status (Compact)
                fig, ax = plt.subplots(figsize=(5, 5))  # Set the size of the pie chart
                ax.pie(attendance_counts, labels=attendance_counts.index, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF6347", "#FFCC00"])
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

                # Bar chart for attendance status (Compact)
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                sns.barplot(x=attendance_counts.index, y=attendance_counts.values, ax=ax_bar, palette="Blues_d")
                ax_bar.set_ylabel('Count')
                ax_bar.set_title(f"Attendance Status for {selected_date}")
                st.pyplot(fig_bar)
            
        else:
            st.warning(f"No attendance data found for {ins_fname} on {selected_date} for the selected Subject and Section")

        def convert_to_12hr_format(time_str):
            # Convert 24-hour time string to 12-hour format with AM/PM
            time_obj = datetime.strptime(time_str, "%H:%M")
            return time_obj.strftime("%I:%M %p")

        # Function to process the class schedule and update the Google Sheet
        def process_class_schedule():
            # Google Sheets API Setup
            SERVICE_ACCOUNT_FILE = 'crypto-argon-445314-r7-f9f259c73941.json'
            SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
            SPREADSHEET_ID = '1gaHG-rZ7xsYsqxvZaeh3OsXcF3FoyiYxs82rB_9Ndws'
            RANGE_NAME = 'Account!A:H'

            # Authenticate and build the Google Sheets API service
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            service = build('sheets', 'v4', credentials=credentials)
            sheet = service.spreadsheets()

            # Get query parameters from Streamlit
            query_params = st.experimental_get_query_params()
            action = query_params.get("action", [None])[0]
            class_start = query_params.get("classStart", [None])[0]
            class_end = query_params.get("classEnd", [None])[0]
            attendance_end = query_params.get("attendanceEnd", [None])[0]
            ins_fname = query_params.get("insFname", [None])[0]

            # If action is 'save_class_schedule', update the Google Sheet
            if action == "save_class_schedule":
                # Convert the times to 12-hour format
                class_start_12hr = convert_to_12hr_format(class_start)
                class_end_12hr = convert_to_12hr_format(class_end)
                attendance_end_12hr = convert_to_12hr_format(attendance_end)

                # Fetch the current data from the Google Sheet
                result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
                values = result.get('values', [])

                # Check if the Instructor Name exists in Column A
                for i, row in enumerate(values):
                    if row and row[0] == ins_fname:  # Assuming Instructor Name is in column A
                        # Update the F, G, and H columns with the new times
                        sheet.values().update(
                            spreadsheetId=SPREADSHEET_ID,
                            range=f'Account!F{i+1}:H{i+1}',  # Row is i+1 because the API is 1-indexed
                            valueInputOption="RAW",
                            body={
                                "values": [[class_start_12hr, class_end_12hr, attendance_end_12hr]]
                            }
                        ).execute()

                        st.write("Class schedule updated successfully!")
                        #st.rerun()
                        break
                else:
                    st.write(f"Instructor {ins_fname} not found in the sheet.")
            else:
                st.write("No class schedule saved yet.")

        # Call the function to process the class schedule
        process_class_schedule()

    elif menu == "Add a Student":

        st.write('<h1 style="text-align:center; font-size:32px; font-weight:bold;">REGISTRATION OF NEW STUDENT</h1>', unsafe_allow_html=True)

        # Initialize session state if it's not already initialized
        if "run_camera" not in st.session_state:
            st.session_state.run_camera = False  # Default is camera off

        # Camera Feed Option
        run_camera = st.checkbox("Start Camera", key="camera_checkbox")

        if run_camera:

            st.session_state.run_camera = True
            video_capture = cv2.VideoCapture(0)
            FRAME_WINDOW = st.image([])  # Image placeholder for live feed

            # Input field for the label name (Student Name or ID)
            label_name = st.text_input("Enter Student ID, Student Last Name, and Student First name")
            
            # Button to capture image
            cap = st.button("Capture Image")

            if cap:
                captured_faces = []
                # Create a fixed grid of placeholders (e.g., 5 columns)
                cols = st.columns(5)
                image_placeholders = [col.empty() for col in cols]

                for i in range(50):  # Capture 50 images
                    ret, frame = video_capture.read()
                    if ret:
                        # Append the resized face to the list
                        captured_faces.append(frame)

                        # Update the placeholders with the latest images
                        for idx, face in enumerate(captured_faces[-5:]):  # Show the last 5 images
                            display_face = cv2.resize(face, (80, 80))  # Resize for display
                            frame_rgb = cv2.cvtColor(display_face, cv2.COLOR_BGR2RGB)
                            image_placeholders[idx].image(frame_rgb, caption=f"Face {len(captured_faces) - 5 + idx + 1}", use_container_width=True)

                if captured_faces:
                    st.session_state.run_camera = False

                    # Save the processed images if needed
                    process_the_image(captured_faces, label_name)

                    # Stop the camera feed
                    video_capture.release()

                else:
                    st.error("No faces were detected. Please try again.")

            # Display live camera feed until the "Capture Image" button is clicked
            while st.session_state.run_camera:
                ret, frame = video_capture.read()
                if not ret:
                    break

                rgb_live = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Display the live camera feed
                FRAME_WINDOW.image(rgb_live, channels='RGB')
        
        else:
            st.session_state.run_camera = False
            st.info("Please start the camera to start adding a student.")

                
    elif menu == "Record Attendance":
        st.write('<h1 style="text-align:center; font-size:32px; font-weight:bold;">CHECKING OF ATTENDANCE</h1>', unsafe_allow_html=True)

        run_camera = st.checkbox("Start Camera")
        if run_camera:
            video_capture = cv2.VideoCapture(0)
            FRAME_WINDOW = st.image([])

            detector = detector
            recorded_attendees = set()
            verify_clicked = False 
            click = st.button("Verify")
            verified = ''

            col1, col2 = st.columns([2, 1])

            with col1:
                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    FRAME_WINDOW.image(frame_rgb, channels='RGB')

                    if not verify_clicked and click:
                        
                        verify_clicked = True

                        faces = detector.detect_faces(frame_rgb)

                        for face in faces:
                            x, y, width, height = face['box']
                            confidence = face['confidence']

                            print(f"Detected face with confidence: {confidence}")

                            if confidence >= 0.90:
                                cropped_face = frame_rgb[y:y+height, x:x+width]

                                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

                                resize_cropped = cv2.resize(cropped_face_rgb, (100,100))

                                cv2.imwrite(os.path.join('One Shot Face Recognition', 'application_data', 'input_image', 'input_image.jpg'), resize_cropped)

                                model_result , similarity, recognized_label = verify_multi_class(0.45, 0.9)


                                if recognized_label != "Unknown":
                                    color = (0, 255, 0)
                                    match = re.search(r'(\d+)\s+([\w]+)\s+([\w]+)', recognized_label)
                                    
                                    student_id = match.group(1)
                                    last_name = match.group(2)
                                    first_name = match.group(3) 

                                    label_text = f"{last_name}, {first_name} ({similarity})"
                                    verified = f'Student Id: {student_id} has been verified'

                                    try:
                                        student_id, name = student_id, f'{last_name}, {first_name}'

                                        # Check if attendance for this person is already recorded
                                        if name not in recorded_attendees:
                                            section = st.session_state.instructor_section
                                            subject = st.session_state.instructor_subject
                                            instructor = st.session_state.instructor_fname

                                            # Record attendance
                                            record_attendance(student_id, name, section, subject, instructor)
                                            st.write(f"Attendance Recorded for {name}")
                                            recorded_attendees.add(name)

                                    except ValueError as e:
                                        st.error(f"Error: {e}")

                                else:
                                    # If confidence is not high enough, label as Unknown
                                    recognized_label = "Unknown"
                                    verified = f'Unverified'
                                    color = (0, 0, 255)  # Red for unknown face
                                    label_text = f"{recognized_label}"

                                # Draw the bounding box and label on the frame
                                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                                rgb_live = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                # Update the frame with bounding boxes and labels
                                FRAME_WINDOW.image(rgb_live, channels="RGB")

                            else:
                                verified = f'NO DETECTED FACE ({confidence})'



                    # Exit loop if verification is done
                    if verify_clicked:
                        st.success("Verification completed.")
                        break

            # Display captured face in the second column, if any
            with col2:
                if verify_clicked:
                    display_verified_or_NOT = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    st.image(display_verified_or_NOT, caption= verified, use_container_width=True, channels="RGB")

            # Release resources when the loop ends
            video_capture.release()