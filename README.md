# Hotel Booking App

This app uses a dataset with hotel bookings to predict the cancellation probability and price for a certain booking!


## Approach

From the dataset I have extracted two target variables: room price (“adr”) and cancellation (“is_canceled”). With the help of various features that constitute the information used in a hotel booking, one can predict these two variables for a given booking. This is why this tool is intended to be used by a hotel employee, to check whether a certain booking they have received is likely to being cancelled and what price they can charge the person for the specific booking. With the help of the cancellation indicator, the hotel can optimize their occupancy, e.g. by double booking room a room when it’s likely that a customer will cancel. With the help of the pricing indicator, the hotel can optimize their yield management, e.g. by making sure they always set a fair price which the customer is willing to pay.


## Usage

Using the sidebar, users can enter the details of a specififc booking.

The output after predictions generation is divided into 3 tabs:

1. "Predictions": The user can see the probability for a cancellation of that booking and the predicted price together with some actionable insights.
2. "Input": The user can see how the inputs got converted into the features for the prediction models.
3. "Historic Predictions": The user has a list of all predictions from the session.


## Setup

For the two target variables, two models ("RandomForestClassifier" as "cancellation.pkl" and "RandomForestRegressor" as "pricing.pkl") were trained and saved using joblib. When the app ("hotel_booking_app.py") is run, it loads the pretrained models and runs the user input through them to get the predictions.


## Link

https://hotel-booking-app.streamlit.app/
