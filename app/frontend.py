import requests
import streamlit as st

BASE_URL = "http://localhost:80"

st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

user_input = st.text_input(
    label="Enter your message to analyze: ",
    placeholder="Example: Congratulations you won a free iphone worth 1000$. Contact on xyz@gmail.com to confirm your phone.",
)

if not user_input.strip():
    st.warning("Please enter a valid non-empty message.")
else:
    try:
        payload = {"message": user_input}

        response = requests.post(
            url=f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            with st.spinner(text="Analyzing your message with Naive Bayes"):
                result = response.json()

                is_spam = result["is_spam"]
                human_label = result["label"]
                confidence_score = result["confidence"]

                if is_spam:
                    st.error("The message is SPAM.")
                    st.metric(label="Confidence Score", value=confidence_score)
                    st.warning("Keep an eye out for this message. This looks spam.")
                else:
                    st.success("The message is NOT SPAM.")
                    st.metric(label="Confidence Score", value=confidence_score)
                    st.info("This message appears to be safe.")

        else:
            st.error(f"Backend returned an error: {response.status_code}")
            st.error(response.json())
    except requests.exceptions.ConnectionError as e:
        st.error("Could not connect to the backend.")
        st.error(e)
