from predict import predict_url


TEST_URLS = [

    "https://google.com",

    "https://github.com",

    "https://www.microsoft.com",

    "http://example.com/login",

    "http://secure-login-account-verification.example.com",

    "http://192.168.1.1/login",

]


for url in TEST_URLS:

    prediction, probabilities = (
        predict_url(url)
    )

    print("=" * 70)

    print(
        f"URL: {url}"
    )

    print(
        f"Prediction: {prediction}"
    )

    print("\nProbabilities:")

    for (
        class_name,
        probability
    ) in sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"  {class_name:<12}"
            f"{probability:.2%}"
        )