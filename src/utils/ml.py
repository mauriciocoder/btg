def print_coefficients(model, feature_names):
    feature_coefficients = dict(zip(feature_names, model.coef_))
    feature_coefficients = dict(
        sorted(feature_coefficients.items(), key=lambda item: item[1])
    )
    # Display the results
    print("Feature coefficients:")
    for feature, coef in feature_coefficients.items():
        print(f"{feature}: {coef}")
    print("Intercept:", model.intercept_)  # intercept term
