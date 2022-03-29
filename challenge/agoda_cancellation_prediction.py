from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from sklearn.model_selection import train_test_split
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()

    # drop samples with booking date after checking date
    booking_dates = pd.to_datetime(full_data["booking_datetime"]).dt.date
    checkin_dates = pd.to_datetime(full_data["checkin_date"]).dt.date
    full_data = full_data[booking_dates <= checkin_dates]

    # add column for days of stay
    checkout_dates = pd.to_datetime(full_data["checkout_date"]).dt.date
    full_data["days_of_stay"] = (checkout_dates - checkin_dates).dt.days

    # drop samples with checkout not-after checkin (one day)
    full_data = full_data[full_data["days_of_stay"] >= 1]

    # transform Nan in requests to 0
    special_requests = ["request_nonesmoke", "request_latecheckin", "request_highfloor", "request_largebed",
                        "request_twinbeds", "request_airport"]
    for request in special_requests:
        full_data[request] = full_data[request].fillna(0)

    # transform charge option to int
    # pay_int_replace = {"Pay Now": 1, "Pay Later": 3, "Pay at Check-in": 2}
    # full_data = full_data.replace({"charge_option": pay_int_replace})
    pay_int_replace = {0: -1, 1:1}
    full_data = full_data.replace({"is_first_booking": pay_int_replace})

    full_data[special_requests] = np.where(full_data[special_requests] == 0, -1, 1)

    # STANDARTIZE original selling amount
    mean_selling_amount = full_data["original_selling_amount"].mean()
    full_data["original_selling_amount"] /= mean_selling_amount

    # create labels - 1 for cancellation, 0 otherwise
    labels = full_data["cancellation_datetime"].between("2018-07-12","2018-13-12").astype(int)
    # labels = labels.fillna(0)
    # labels[labels != 0] = 1

    numbers = ["no_of_room", "no_of_extra_bed", "no_of_children", "no_of_adults"]
    features = pd.concat([full_data[["days_of_stay", "hotel_star_rating"] + special_requests + numbers], pd.get_dummies(full_data["charge_option"], drop_first=True),
                          pd.get_dummies(full_data["accommadation_type_name"], drop_first=True),
                          pd.get_dummies(full_data["guest_nationality_country_name"], drop_first=True),
                          pd.get_dummies(full_data["language"], drop_first=True),
                          pd.get_dummies(full_data["original_payment_currency"], drop_first=True),
                          pd.get_dummies(full_data["request_earlycheckin"], drop_first=True)], axis=1)



    return features, labels


# def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
#     """
#     Export to specified file the prediction results of given estimator on given testset.
#
#     File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
#     predicted values.
#
#     Parameters
#     ----------
#     estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
#         Fitted estimator to use for prediction
#
#     X: ndarray of shape (n_samples, n_features)
#         Test design matrix to predict its responses
#
#     filename:
#         path to store file at
#
#     """
#     pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    cancellation_labels = cancellation_labels.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(df, cancellation_labels, test_size=0.9)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(X_train, y_train)
    print(estimator.loss(X_test, y_test))

    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")