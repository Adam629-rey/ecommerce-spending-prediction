"""Interactive demonstration; no serialized model is loaded."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from spending.model import FEATURES, load_data, train, predict

st.set_page_config(page_title="Customer Spending", page_icon="📊", layout="wide")
st.title("E-commerce Customer Spending")
st.caption("An interpretable linear regression demo · Educational use")

@st.cache_resource
def resources():
    return train(load_data())

model, report, training, predictions = resources()
prediction_tab, evaluation_tab, about_tab = st.tabs(["Predict", "Model evaluation", "About"])
with prediction_tab:
    st.write("Enter a customer profile to estimate yearly spending (dataset units).")
    st.info("Time and membership units are not documented in the supplied data. Match the original dataset conventions.")
    with st.form("customer"):
        values = {}
        for feature in FEATURES:
            values[feature] = st.number_input(feature, min_value=0.0,
                value=float(training[feature].median()), step=0.1, format="%.2f")
        submitted = st.form_submit_button("Estimate yearly spending")
    if submitted:
        sample = pd.DataFrame([values])
        outside = [f for f in FEATURES if not training[f].min() <= values[f] <= training[f].max()]
        if outside:
            st.warning("Outside the training range: " + ", ".join(outside) + ". This estimate may be unreliable.")
        estimate = float(predict(model, sample)[0])
        st.metric("Estimated yearly spending", f"{estimate:,.2f}")
        if estimate < 0:
            st.warning("The unconstrained linear model produced a negative estimate; this profile is not reliable.")
        st.caption("A point estimate, not a guaranteed outcome or a prediction interval.")
    with st.expander("Training input ranges"):
        st.dataframe(training.describe().loc[["min", "50%", "max"]])
with evaluation_tab:
    st.write("Held-out test set: 30% · seed 101 · model trained on the remaining 70%")
    cols = st.columns(3)
    for col, key, label in zip(cols, ["mae", "rmse", "r2"], ["MAE", "RMSE", "R²"]):
        col.metric(label, f"{report['holdout'][key]:.3f}")
    st.subheader("Linear model versus training-mean baseline")
    st.dataframe(pd.DataFrame({"Linear regression": report["holdout"], "Mean baseline": report["baseline"]}).T)
    st.subheader("Five-fold validation on training data only")
    st.dataframe(pd.DataFrame(report["cross_validation"]).T[["mean", "std"]])
    fig, ax = plt.subplots()
    ax.scatter(predictions.predicted, predictions.residual, alpha=.7)
    ax.axhline(0, linestyle="--", color="black")
    ax.set(xlabel="Predicted spending", ylabel="Actual - predicted")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Coefficients in original feature units")
    st.dataframe(pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_}))
    st.caption("Coefficients describe conditional associations, not causal effects. Their magnitudes depend on feature units.")
with about_tab:
    st.write("Predictors: session length, app time, website time and membership length. Direct identifiers were excluded.")
    st.write("The dataset was supplied with a course project. Its real-world origin, units and redistribution license have not been established.")
    st.warning("This random split assumes independent observations. Performance does not establish future or external validity. Confirm data rights before public redistribution.")
