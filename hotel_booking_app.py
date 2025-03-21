import streamlit as st
import joblib
import datetime
import pycountry
import pandas as pd
from fpdf import FPDF
import io


# Initialize Streamlit app
st.title("Hotel Booking App")
st.write("This app can predict the chance of cancellation and the expected price for a hotel customer upon entry of their details in the siderbar to the left.")
st.write("This app is intended to be used by professionals in the hotel industry during room booking to provide actionable insights on customer behaviour.")


### Input Fields

# Move configuration to sidebar
st.sidebar.title("Enter guest details below")

# Create form for remaining inputs
with st.sidebar.form("hotel_booking_form"):

    # Date input
    today = datetime.date.today()

    # Create two columns to display date inputs
    col1, col2 = st.columns(2)

    # Get Arrival Date
    with col1:
        arrival_date = st.date_input("Arrival Date", value=(today + datetime.timedelta(days=1)))

    # Get Departure Date
    with col2:
        departure_date = st.date_input("Departure Date", value=(arrival_date + datetime.timedelta(days=1)))

    # Use datetime to extract the parts needed for the model
    arrival_date_year = arrival_date.year
    arrival_date_month = arrival_date.month
    arrival_date_week_number = arrival_date.isocalendar()[1]
    arrival_date_day_of_month = arrival_date.day

    # Calculate the lead time
    if arrival_date > today:
        lead_time = (arrival_date - today).days
    else:
        st.error("Arrival date must be in the future!")
        lead_time = 0

    # Calculate the number of weekend/weekday nights
    if departure_date > arrival_date:
        total_nights = (departure_date - arrival_date).days
    else:
        st.error("Departure date must be after arrival date!")
        total_nights = 0

    stays_in_week_nights = 0
    stays_in_weekend_nights = 0

    for i in range(total_nights):
        current_day = arrival_date + datetime.timedelta(days=i)
        if current_day.weekday() < 5:
            stays_in_week_nights += 1
        else:
            stays_in_weekend_nights += 1

    # Occupant input
    with st.expander("Occupant Input"):
        adults = st.number_input("Adults", min_value=1, value=2)
        children = st.number_input("Children", min_value=0, value=0)
        babies = st.number_input("Babies", min_value=0, value=0)
        meal_plan = st.selectbox("Meal Plan", options=["Bed & Breakfast", "Half Board", "Full Board", "Self-Catering", "Undefined"])
        meal_code = {"Bed & Breakfast": "BB", "Half Board": "HB", "Full Board": "FB", "Self-Catering": "SC", "Undefined": "Undefined"}
        meal = meal_code.get(meal_plan, "Undefined")
        required_car_parking_spaces = st.number_input("Car Parking Spaces", min_value=0, value=0)
        countries = [country.name for country in pycountry.countries]
        selected_country = st.selectbox("Choose a country:", countries)
        country = pycountry.countries.get(name=selected_country).alpha_3

    # Distribution input
    with st.expander("Distribution Input"):
        hotel = st.selectbox("Hotel Type", options=["Resort Hotel", "City Hotel"])
        market_segment = st.selectbox("Market Segment", options=["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation", "Undefined"])
        distribution_channel = st.selectbox("Distribution Channel", options=["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
        agent = st.number_input("Agent", min_value=0, value=0)

    # Loyalty input
    with st.expander("Loyalty Input"):
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
        is_repeated_guest = 1 if previous_bookings_not_canceled >0 else 0

    # Remaining input fields
    with st.expander("Other Input"):
        reserved_room_type = st.selectbox("Reserved Room Type", options=["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
        booking_changes = st.number_input("Booking Changes", min_value=0, value=0)
        deposit_type = st.selectbox("Deposit Type", options=["No Deposit", "Refundable", "Non Refund"])
        days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, value=0)
        customer_type = st.selectbox("Customer Type", options=["Transient", "Transient-Party", "Contract", "Group"])
        total_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)

    # Submit form
    submitted = st.form_submit_button("Generate Predictions")


### Input Dataframe

# Create a DataFrame with the user inputs
input_data = pd.DataFrame({
    "hotel": [hotel],
    "lead_time": [lead_time],
    "arrival_date_year": [arrival_date_year],
    "arrival_date_month": [arrival_date_month],
    "arrival_date_week_number": [arrival_date_week_number],
    "arrival_date_day_of_month": [arrival_date_day_of_month],
    "stays_in_weekend_nights": [stays_in_weekend_nights],
    "stays_in_week_nights": [stays_in_week_nights],
    "adults": [adults],
    "children": [children],
    "babies": [babies],
    "meal": [meal],
    "country": [country],
    "market_segment": [market_segment],
    "distribution_channel": [distribution_channel],
    "is_repeated_guest": [is_repeated_guest],
    "previous_cancellations": [previous_cancellations],
    "previous_bookings_not_canceled": [previous_bookings_not_canceled],
    "reserved_room_type": [reserved_room_type],
    "booking_changes": [booking_changes],
    "deposit_type": [deposit_type],
    "agent": [agent],
    "days_in_waiting_list": [days_in_waiting_list],
    "customer_type": [customer_type],
    "required_car_parking_spaces": [required_car_parking_spaces],
    "total_of_special_requests": [total_of_special_requests]
})


### Predictions

# Create cache for historic predictions
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame()
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create function to export prediction PDF
def generate_pdf(cancellation_proba_out, predicted_pricing, average_cancellation_proba, average_price):
    pdf = FPDF()
    pdf.add_page()

    # Title bar
    pdf.set_fill_color(63, 81, 181)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 12, 'Hotel Booking Prediction Report', ln=True, align='C', fill=True)
    pdf.ln(8)

    # Initiate Cancellation section
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Cancellation Prediction', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.ln(2)

    # Cancellation probability
    pdf.cell(100, 8, f"Cancellation Probability:", ln=False)
    pdf.cell(0, 8, f"{cancellation_proba_out:.2f}%", ln=True)
    pdf.cell(100, 8, f"Average Cancellation Probability:", ln=False)
    pdf.cell(0, 8, f"{average_cancellation_proba:.2f}%", ln=True)
    pdf.ln(4)

    # Cancellation insight box
    if cancellation_proba_out > average_cancellation_proba:
        pdf.set_fill_color(255, 205, 210)
        pdf.multi_cell(0, 10, "Higher than average cancellation risk.", border=1, fill=True)
    else:
        pdf.set_fill_color(200, 230, 201)
        pdf.multi_cell(0, 10, "Lower than average cancellation risk.", border=1, fill=True)
    pdf.ln(3)

    # Initiate Cancellation section
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Price Prediction', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.ln(2)

    # Predicted price
    pdf.cell(100, 8, f"Predicted Price:", ln=False)
    pdf.cell(0, 8, f"${predicted_pricing:.2f}", ln=True)
    pdf.cell(100, 8, f"Average Price:", ln=False)
    pdf.cell(0, 8, f"${average_price:.2f}", ln=True)
    pdf.ln(10)

    # Price insight box
    if predicted_pricing < average_price:
        pdf.set_fill_color(255, 205, 210)
        pdf.multi_cell(0, 10, "Price is below average.", border=1, fill=True)
    else:
        pdf.set_fill_color(200, 230, 201)
        pdf.multi_cell(0, 10, "Price is above average.", border=1, fill=True)
    pdf.ln(3)

    # Output to bytes buffer
    return io.BytesIO(pdf.output(dest="S").encode("latin1"))


# Calculate predictions
if submitted:

    # Create status tracking
    with st.status("Generating Predictions..."):

        # Load the pre-trained models
        if 'cancellation_model' not in st.session_state or 'pricing_model' not in st.session_state:
            st.write("Loading Models...")
            st.session_state.cancellation_model = joblib.load("cancellation.pkl")
            st.session_state.pricing_model = joblib.load("pricing.pkl")
        else:
            st.write("Loading Models... already loaded!")

        # Predict cancellation probability (using the previously trained & loaded classifier)
        st.write("Predicting Cancellation Probability...")
        cancellation_proba = st.session_state.cancellation_model.predict_proba(input_data)[0][1]
        cancellation_proba_out = cancellation_proba * 100
        average_cancellation_proba = 37.0416  # From streamlit_model_full.ipynb
        
        # Predict price (using the previously trained & loaded regressor)
        st.write("Predicting Price...")
        predicted_pricing = st.session_state.pricing_model.predict(input_data)[0]
        average_price = 101.831122  # From streamlit_model_full.ipynb

    # Create tabs to order output
    tab1, tab2, tab3 = st.tabs(["Predictions", "Input Data", "Historic Predictions"])
    
    # Create prediction output tab
    with tab1:

        # Create two columns to display outputs
        outer_col1, outer_col2 = st.columns(2)

        with outer_col1:
            st.subheader("Cancellation Probability")

            # Create progress bar with labels
            st.progress(cancellation_proba_out / 100)
            inner_col1, inner_col2 = st.columns([6, 1])
            with inner_col1:
                st.write("0%")
            with inner_col2:
                st.write("100%")

            # Create metric to compare prediction to average
            st.metric(label="Current Prediction", value=f"{cancellation_proba_out:.2f}%", delta=f"{(cancellation_proba_out - average_cancellation_proba):.2f}% vs avg.", delta_color="inverse")
            st.write(f"Average Cancellation Rate: **{average_cancellation_proba:.2f}%**")
        
        with outer_col2:
            st.subheader("Predicted Price")

            # Create progress bar with labels
            st.progress(predicted_pricing / 250)
            inner_col1, inner_col2 = st.columns([6, 1])
            with inner_col1:
                st.write("0$")
            with inner_col2:
                st.write("250$")

            # Create metric to compare prediction to average
            st.metric(label="Current Prediction", value=f"{predicted_pricing:.2f}$", delta=f"{(predicted_pricing - average_price):.2f}$ vs avg.")
            st.write(f"Average Price: **{average_price:.2f}$**")

        st.divider()

        # Create actionable insights
        st.subheader("Insights")

        if cancellation_proba_out > average_cancellation_proba:
            st.warning("Higher than average cancellation risk.")
        else:
            st.success("Lower than average cancellation risk.")
        if predicted_pricing > average_price:
            st.success("Price is above average.")
        else:
            st.warning("Price is below average.")

        # Add PDF download button
        pdf_bytes = generate_pdf(cancellation_proba_out, predicted_pricing, average_cancellation_proba, average_price)
        st.download_button(
            label="Download Predictions PDF",
            data=pdf_bytes,
            file_name="predictions_report.pdf",
            mime="application/pdf",
            key="pdf_download"
        )
    
    # Create raw prediction input tab
    with tab2:
        st.subheader("Raw input data for predictions")
        
        # Display input dataframe
        display_df = input_data.T
        display_df.columns = ["Input"]
        display_df.index.name = "Features"
        st.dataframe(display_df, height=950, width=400)

    # Create historic predictions tab
    with tab3:
        st.subheader("All predictions of current session")

        # Add current prediction to dataframe
        new_prediction = input_data.iloc[0].to_dict()
        new_prediction["timestamp"] = timestamp
        new_prediction["cancellation_probability"] = cancellation_proba
        new_prediction["predicted_price"] = predicted_pricing
        new_prediction_df = pd.DataFrame([new_prediction]).set_index("timestamp")
        col_order = ["cancellation_probability", "predicted_price"] + [col for col in new_prediction_df.columns if col not in ["cancellation_probability", "predicted_price"]]
        new_prediction_df = new_prediction_df[col_order]

        # Append the new row to the historic predictions DataFrame stored in session state.
        st.session_state.prediction_history = pd.concat(
            [st.session_state.prediction_history, new_prediction_df]
        )

        # Display historic predictions
        if not st.session_state.prediction_history.empty:
            st.dataframe(st.session_state.prediction_history.sort_index(ascending=False))
        else:
            st.write("No predictions have been made yet.")
