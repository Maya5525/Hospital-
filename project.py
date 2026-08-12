# ==========================================
# HOSPITAL BEDS MANAGEMENT ANALYSIS PROJECT
# ==========================================

# =====================
# 1. Import Libraries
# =====================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load


# ==========================================
# Streamlit Page Configuration
# ==========================================

st.set_page_config(
    page_title="Hospital Dashboard",
    layout="wide"
)


# ==========================================
# 2. Load Dataset
# ==========================================

staff = pd.read_csv("staff.csv")
patients = pd.read_csv("patients.csv")
staff_schedule = pd.read_csv("staff_schedule.csv")
services_weekly = pd.read_csv("services_weekly.csv")

df = services_weekly.copy()


# ==========================================
# 3. Cleaning Pipeline
# ==========================================

def clean_column_names(df):

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    return df


def remove_duplicates(df):

    return df.drop_duplicates()


def handle_missing_values(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:

        if not df[col].mode().empty:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

    return df


def clean_text(df):

    text_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in text_cols:

        df[col] = df[col].str.strip()

    return df


def convert_dates(df):

    date_cols = [
        "arrival_date",
        "departure_date"
    ]

    for col in date_cols:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    return df


def remove_negative_values(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        df = df[df[col] >= 0]

    return df


def cleaning_pipeline(df):

    df = clean_column_names(df)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = clean_text(df)

    df = convert_dates(df)

    df = remove_negative_values(df)

    return df


# ==========================================
# Apply Cleaning
# ==========================================

staff = cleaning_pipeline(staff)

patients = cleaning_pipeline(patients)

staff_schedule = cleaning_pipeline(
    staff_schedule
)

services_weekly = cleaning_pipeline(
    services_weekly
)

df = services_weekly.copy()


# ==========================================
# 4. Sidebar Navigation
# ==========================================

st.sidebar.title("Hospital System")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Dashboard",
        "ML Prediction"
    ]
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    # ==========================================
    # Dashboard Header
    # ==========================================

    st.title("🏥 Hospital Dashboard")

    st.write(
        "Hospital Beds Management Dashboard"
    )


    # ==========================================
    # Dataset Information
    # ==========================================

    st.success(
        "Dataset loaded successfully"
    )

    with st.expander("View Dataset"):

        st.subheader(
            "Services Weekly Dataset"
        )

        st.dataframe(
            df,
            use_container_width=True
        )


    # ==========================================
    # Explore Data
    # ==========================================

    with st.expander(
        "Explore Datasets"
    ):

        st.subheader("STAFF")

        st.dataframe(
            staff.head(),
            use_container_width=True
        )


        st.subheader("PATIENTS")

        st.dataframe(
            patients.head(),
            use_container_width=True
        )


        st.subheader(
            "STAFF SCHEDULE"
        )

        st.dataframe(
            staff_schedule.head(),
            use_container_width=True
        )


        st.subheader(
            "SERVICES WEEKLY"
        )

        st.dataframe(
            services_weekly.head(),
            use_container_width=True
        )


        st.subheader("Columns")


        st.write("Staff columns:")

        st.write(
            list(staff.columns)
        )


        st.write("Patients columns:")

        st.write(
            list(patients.columns)
        )


        st.write(
            "Staff Schedule columns:"
        )

        st.write(
            list(
                staff_schedule.columns
            )
        )


        st.write(
            "Services Weekly columns:"
        )

        st.write(
            list(
                services_weekly.columns
            )
        )


    # ==========================================
    # Filters
    # ==========================================

    st.sidebar.header("Filters")


    selected_service = st.sidebar.selectbox(

        "Selected Service",

        ["All"] +
        list(
            df["service"].unique()
        ),

        key="dashboard_service"

    )


    selected_event = st.sidebar.selectbox(

        "Selected Event",

        ["All"] +
        list(
            df["event"].unique()
        ),

        key="dashboard_event"

    )


    filtered_df = df.copy()


    if selected_service != "All":

        filtered_df = filtered_df[
            filtered_df["service"]
            == selected_service
        ]


    if selected_event != "All":

        filtered_df = filtered_df[
            filtered_df["event"]
            == selected_event
        ]


    # ==========================================
    # 5. Merge Data
    # ==========================================

    staff_data = pd.merge(

        staff,

        staff_schedule,

        on=[
            "staff_id",
            "staff_name",
            "role",
            "service"
        ],

        how="left"

    )


    hospital_data = pd.merge(

        patients,

        services_weekly,

        on="service",

        how="left"

    )


    # ==========================================
    # 6. KPI Calculation
    # ==========================================

    def calculate_kpis():

        total_patients = patients[
            "patient_id"
        ].nunique()


        total_staff = staff[
            "staff_id"
        ].nunique()


        total_services = services_weekly[
            "service"
        ].nunique()


        total_requests = services_weekly[
            "patients_request"
        ].sum()


        total_admitted = services_weekly[
            "patients_admitted"
        ].sum()


        total_refused = services_weekly[
            "patients_refused"
        ].sum()


        if total_requests > 0:

            admission_rate = (

                total_admitted
                / total_requests

            ) * 100


            refusal_rate = (

                total_refused
                / total_requests

            ) * 100

        else:

            admission_rate = 0

            refusal_rate = 0


        avg_beds = services_weekly[
            "available_beds"
        ].mean()


        patient_satisfaction = (
            services_weekly[
                "patient_satisfaction"
            ].mean()
        )


        staff_morale = services_weekly[
            "staff_morale"
        ].mean()


        attendance_rate = (

            staff_schedule[
                "present"
            ].sum()

            /

            len(staff_schedule)

        ) * 100


        return {

            "Total Patients":
                total_patients,

            "Total Staff":
                total_staff,

            "Total Services":
                total_services,

            "Average Beds":
                round(
                    avg_beds,
                    2
                ),

            "Patient Requests":
                total_requests,

            "Patients Admitted":
                total_admitted,

            "Patients Refused":
                total_refused,

            "Admission Rate %":
                round(
                    admission_rate,
                    2
                ),

            "Refusal Rate %":
                round(
                    refusal_rate,
                    2
                ),

            "Patient Satisfaction":
                round(
                    patient_satisfaction,
                    2
                ),

            "Staff Morale":
                round(
                    staff_morale,
                    2
                ),

            "Attendance Rate %":
                round(
                    attendance_rate,
                    2
                )

        }


    kpis = calculate_kpis()


    # ==========================================
    # 7. Display KPIs
    # ==========================================

    st.header(
        "Hospital KPI Dashboard"
    )


    # First row

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Patients",
            f"{kpis['Total Patients']:,}"
        )


    with col2:

        st.metric(
            "Total Staff",
            f"{kpis['Total Staff']:,}"
        )


    with col3:

        st.metric(
            "Total Services",
            f"{kpis['Total Services']:,}"
        )


    # Second row

    col4, col5, col6 = st.columns(3)


    with col4:

        st.metric(
            "Average Beds",
            f"{kpis['Average Beds']:.2f}"
        )


    with col5:

        st.metric(
            "Patient Requests",
            f"{kpis['Patient Requests']:,}"
        )


    with col6:

        st.metric(
            "Patients Admitted",
            f"{kpis['Patients Admitted']:,}"
        )


    # Third row

    col7, col8, col9 = st.columns(3)


    with col7:

        st.metric(
            "Patients Refused",
            f"{kpis['Patients Refused']:,}"
        )


    with col8:

        st.metric(
            "Admission Rate",
            f"{kpis['Admission Rate %']:.2f}%"
        )


    with col9:

        st.metric(
            "Refusal Rate",
            f"{kpis['Refusal Rate %']:.2f}%"
        )


    # Fourth row

    col10, col11, col12 = st.columns(3)


    with col10:

        st.metric(
            "Patient Satisfaction",
            f"{kpis['Patient Satisfaction']:.2f}"
        )


    with col11:

        st.metric(
            "Staff Morale",
            f"{kpis['Staff Morale']:.2f}"
        )


    with col12:

        st.metric(
            "Attendance Rate",
            f"{kpis['Attendance Rate %']:.2f}%"
        )


    # ==========================================
    # 8. Charts
    # ==========================================

    st.header(
        "Hospital Charts"
    )


    # ==========================================
    # 1. Monthly Admissions
    # ==========================================

    monthly = services_weekly.groupby(
        "month"
    )[
        "patients_admitted"
    ].sum()


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )


    ax.set_title(
        "Monthly Admissions Trend"
    )


    ax.set_xlabel("Month")


    ax.set_ylabel(
        "Patients Admitted"
    )


    ax.grid()


    st.pyplot(fig)


    plt.close(fig)


    # ==========================================
    # 2. Requests vs Admitted vs Refused
    # ==========================================

    service_compare = services_weekly.groupby(
        "service"
    )[
        [
            "patients_request",
            "patients_admitted",
            "patients_refused"
        ]
    ].sum()


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    service_compare.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Requests vs Admissions vs Refused"
    )


    ax.set_ylabel(
        "Number of Patients"
    )


    ax.set_xlabel(
        "Service"
    )


    plt.xticks(
        rotation=45
    )


    st.pyplot(fig)


    plt.close(fig)


    # ==========================================
    # 3. Available Beds by Service
    # ==========================================

    beds = services_weekly.groupby(
        "service"
    )[
        "available_beds"
    ].mean()


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        beds.index,
        beds.values
    )


    ax.set_title(
        "Average Available Beds by Service"
    )


    ax.set_xlabel(
        "Service"
    )


    ax.set_ylabel(
        "Beds"
    )


    plt.xticks(
        rotation=45
    )


    st.pyplot(fig)


    plt.close(fig)


    # ==========================================
    # 4. Top Services by Requests
    # ==========================================

    top_services = services_weekly.groupby(
        "service"
    )[
        "patients_request"
    ].sum().sort_values()


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.barh(
        top_services.index,
        top_services.values
    )


    ax.set_title(
        "Top Services by Patient Requests"
    )


    ax.set_xlabel(
        "Requests"
    )


    ax.set_ylabel(
        "Service"
    )


    st.pyplot(fig)


    plt.close(fig)


    # ==========================================
    # 5. Satisfaction by Service
    # ==========================================

    satisfaction = services_weekly.groupby(
        "service"
    )[
        "patient_satisfaction"
    ].mean()


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        satisfaction.index,
        satisfaction.values
    )


    ax.set_title(
        "Patient Satisfaction by Service"
    )


    ax.set_xlabel(
        "Service"
    )


    ax.set_ylabel(
        "Satisfaction"
    )


    plt.xticks(
        rotation=45
    )


    st.pyplot(fig)


    plt.close(fig)


    # ==========================================
    # 6. Staff Attendance
    # ==========================================

    attendance = staff_schedule[
        "present"
    ].value_counts()


    fig, ax = plt.subplots(
        figsize=(5, 5)
    )


    ax.pie(
        attendance.values,
        labels=attendance.index,
        autopct="%1.1f%%"
    )


    ax.set_title(
        "Staff Attendance"
    )


    st.pyplot(fig)


    plt.close(fig)


    # ==========================================
    # 9. Business Insights
    # ==========================================

    st.header(
        "📊 Business Insights"
    )


    st.markdown(
        f"""
- 👨 **Total Patients:** {kpis['Total Patients']}
- 👩‍⚕️ **Total Staff:** {kpis['Total Staff']}
- 🏥 **Total Services:** {kpis['Total Services']}
- ✅ **Admission Rate:** {kpis['Admission Rate %']}%
- ❌ **Refusal Rate:** {kpis['Refusal Rate %']}%
- ⭐ **Patient Satisfaction:** {kpis['Patient Satisfaction']}
- 😊 **Staff Morale:** {kpis['Staff Morale']}
- 🩺 **Attendance Rate:** {kpis['Attendance Rate %']}%
"""
    )


    # ==========================================
    # 10. Recommendations
    # ==========================================

    st.header(
        "💡 Business Recommendations"
    )


    recommendations = []


    if kpis["Refusal Rate %"] > 10:

        recommendations.append(
            "Increase bed capacity to reduce patient refusals."
        )


    if kpis["Patient Satisfaction"] < 4:

        recommendations.append(
            "Improve patient experience and reduce waiting times."
        )


    if kpis["Attendance Rate %"] < 90:

        recommendations.append(
            "Improve staff scheduling and attendance monitoring."
        )


    recommendations.append(
        "Allocate resources based on high-demand services."
    )


    recommendations.append(
        "Monitor weekly trends to prepare for peak periods."
    )


    if not filtered_df.empty:

        top_service = (

            filtered_df.groupby(
                "service"
            )[
                "patients_request"
            ].sum().idxmax()

        )


        recommendations.append(

            f"Allocate additional resources to "
            f"{top_service} because it has the highest "
            f"patient demand."

        )


    for r in recommendations:

        st.success(r)


    # ==========================================
    # Automated Insights
    # ==========================================

    st.header(
        "Automated Insights"
    )


    if not filtered_df.empty:

        # Top service by requests

        top_service = (

            filtered_df.groupby(
                "service"
            )[
                "patients_request"
            ].sum().idxmax()

        )


        top_service_requests = (

            filtered_df.groupby(
                "service"
            )[
                "patients_request"
            ].sum().max()

        )


        st.write(
            f"🏥 **Service with Highest Patient Requests:** "
            f"{top_service}"
        )


        st.write(
            f"👥 **Patient Requests:** "
            f"{top_service_requests:,}"
        )


        # Top admission service

        top_admission_service = (

            filtered_df.groupby(
                "service"
            )[
                "patients_admitted"
            ].sum().idxmax()

        )


        top_admission_value = (

            filtered_df.groupby(
                "service"
            )[
                "patients_admitted"
            ].sum().max()

        )


        st.write(
            f"✅ **Service with Highest Admissions:** "
            f"{top_admission_service}"
        )


        st.write(
            f"👥 **Patients Admitted:** "
            f"{top_admission_value:,}"
        )


        # Lowest satisfaction

        lowest_satisfaction_service = (

            filtered_df.groupby(
                "service"
            )[
                "patient_satisfaction"
            ].mean().idxmin()

        )


        lowest_satisfaction_value = (

            filtered_df.groupby(
                "service"
            )[
                "patient_satisfaction"
            ].mean().min()

        )


        st.write(
            f"⚠️ **Service with Lowest Patient Satisfaction:** "
            f"{lowest_satisfaction_service}"
        )


        st.write(
            f"⭐ **Patient Satisfaction:** "
            f"{lowest_satisfaction_value:.2f}"
        )


        # Highest satisfaction

        highest_satisfaction_service = (

            filtered_df.groupby(
                "service"
            )[
                "patient_satisfaction"
            ].mean().idxmax()

        )


        highest_satisfaction_value = (

            filtered_df.groupby(
                "service"
            )[
                "patient_satisfaction"
            ].mean().max()

        )


        st.write(
            f"⭐ **Service with Highest Patient Satisfaction:** "
            f"{highest_satisfaction_service}"
        )


        st.write(
            f"⭐ **Patient Satisfaction:** "
            f"{highest_satisfaction_value:.2f}"
        )


        # Top event

        top_event = (

            filtered_df.groupby(
                "event"
            )[
                "patients_request"
            ].sum().idxmax()

        )


        top_event_requests = (

            filtered_df.groupby(
                "event"
            )[
                "patients_request"
            ].sum().max()

        )


        st.write(
            f"📊 **Event with Highest Patient Requests:** "
            f"{top_event}"
        )


        st.write(
            f"👥 **Patient Requests:** "
            f"{top_event_requests:,}"
        )


        # Most available beds

        top_bed_service = (

            filtered_df.groupby(
                "service"
            )[
                "available_beds"
            ].mean().idxmax()

        )


        top_bed_value = (

            filtered_df.groupby(
                "service"
            )[
                "available_beds"
            ].mean().max()

        )


        st.write(
            f"🛏️ **Service with Most Available Beds:** "
            f"{top_bed_service}"
        )


        st.write(
            f"🛏️ **Average Available Beds:** "
            f"{top_bed_value:.2f}"
        )


    # ==========================================
    # Hospital Status
    # ==========================================

    st.header(
        "📈 Hospital Status"
    )


    # Patient Satisfaction

    if kpis["Patient Satisfaction"] >= 80:

        st.success(
            "Patient satisfaction is good."
        )

    elif kpis["Patient Satisfaction"] >= 70:

        st.warning(
            "Patient satisfaction is moderate. "
            "Consider improving patient experience."
        )

    else:

        st.error(
            "Patient satisfaction is low. "
            "Improvement is needed."
        )


    # Refusal Rate

    if kpis["Refusal Rate %"] > 10:

        st.warning(
            "Patient refusal rate is high. "
            "Consider increasing bed capacity."
        )


    # Staff Morale

    if kpis["Staff Morale"] < 70:

        st.warning(
            "Staff morale is low. "
            "Consider improving staff working conditions."
        )


    # Attendance Rate

    if kpis["Attendance Rate %"] < 90:

        st.warning(
            "Staff attendance is below 90%. "
            "Review staff scheduling."
        )


    # ==========================================
    # Filtered Dataset
    # ==========================================

    st.subheader(
        "Filtered Dataset"
    )


    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# ==========================================
# ML PREDICTION PAGE
# PREDICT PATIENTS ADMITTED
# ==========================================

elif page == "ML Prediction":

    st.title(
        "🤖 Patients Admitted Prediction"
    )


    st.write(
        "Enter the hospital information below "
        "to predict the number of patients admitted."
    )


    # ==========================================
    # Load Model
    # ==========================================

    try:

        model = load(
            "patients_admitted_model.pkl"
        )

    except FileNotFoundError:

        st.error(
            "patients_admitted_model.pkl was not found. "
            "Make sure you run train_model.py first "
            "and keep the .pkl file in the same folder "
            "as your Streamlit project."
        )

        st.stop()


    # ==========================================
    # User Inputs
    # ==========================================

    available_beds_input = st.number_input(

        "Available Beds",

        min_value=0,

        value=0,

        key="admitted_available_beds"

    )


    patients_request_input = st.number_input(

        "Patients Request",

        min_value=0,

        value=0,

        key="admitted_patients_request"

    )


    patients_refused_input = st.number_input(

        "Patients Refused",

        min_value=0,

        value=0,

        key="admitted_patients_refused"

    )


    staff_morale_input = st.number_input(

        "Staff Morale",

        min_value=0.0,

        value=0.0,

        key="admitted_staff_morale"

    )


    week_input = st.number_input(

        "Week",

        min_value=1,

        value=1,

        key="admitted_week"

    )


    month_input = st.number_input(

        "Month",

        min_value=1,

        max_value=12,

        value=1,

        key="admitted_month"

    )


    service_input = st.selectbox(

        "Service",

        df["service"].unique(),

        key="admitted_service"

    )


    event_input = st.selectbox(

        "Event",

        df["event"].unique(),

        key="admitted_event"

    )


    # ==========================================
    # Create Input Data
    # ==========================================

    input_data = pd.DataFrame({

        "week": [
            week_input
        ],

        "month": [
            month_input
        ],

        "service": [
            service_input
        ],

        "available_beds": [
            available_beds_input
        ],

        "patients_request": [
            patients_request_input
        ],

        "patients_refused": [
            patients_refused_input
        ],

        "staff_morale": [
            staff_morale_input
        ],

        "event": [
            event_input
        ]

    })


    # ==========================================
    # Show Input Data
    # ==========================================

    st.subheader(
        "Input Data"
    )


    st.dataframe(
        input_data,
        use_container_width=True
    )


    # ==========================================
    # Prediction Button
    # ==========================================

    if st.button(

        "Predict Patients Admitted",

        key="predict_admitted_button"

    ):

        try:

            prediction = model.predict(
                input_data
            )


            predicted_admitted = prediction[0]


            # Patients admitted cannot be negative

            predicted_admitted = max(
                0,
                predicted_admitted
            )


            # ==========================================
            # Display Prediction
            # ==========================================

            st.subheader(
                "Predicted Patients Admitted"
            )


            st.success(

                f"Predicted Patients Admitted: "
                f"{predicted_admitted:.0f}"

            )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )
