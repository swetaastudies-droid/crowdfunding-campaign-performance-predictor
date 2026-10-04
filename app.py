# ============================================================
# ECO SIP - CROWDFUNDING CAMPAIGN INTELLIGENCE PLATFORM
# Integrated Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="EcoSip Crowdfunding Intelligence",
    page_icon="🥤",
    layout="wide"
)

MODEL_FILE = "improved_crowdfunding_random_forest.pkl"
DATA_FILE = "crowdfunding_historical_synthetic_500.csv"
VALIDATION_FILE = "step6_real_campaign_validation_results.csv"

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_FILE}' was not found. "
        "Keep it in the same folder as this app.py file."
    )
    st.stop()

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("🥤 EcoSip")
st.sidebar.caption("Crowdfunding Intelligence Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "🔮 Campaign Predictor",
        "📊 Historical Dashboard",
        "💡 Campaign Recommendations",
        "📱 Social Media Readiness",
        "🚀 Final Launch Assessment"
    ]
)

# ------------------------------------------------------------
# COMMON DATA LOADER
# ------------------------------------------------------------

@st.cache_data
def load_historical_data():
    df = pd.read_csv(DATA_FILE)

    numeric_columns = [
        "success",
        "funding_goal",
        "amount_raised",
        "number_of_backers",
        "campaign_duration",
        "funding_percentage"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ------------------------------------------------------------
# CAMPAIGN PREDICTOR
# ------------------------------------------------------------

if page == "🔮 Campaign Predictor":

    st.title("🥤 EcoSip Crowdfunding Campaign Predictor")
    st.subheader("Sip Smart. Live Sustainable.")

    st.write(
        "Estimate the likelihood of crowdfunding campaign success "
        "using pre-launch campaign characteristics."
    )

    st.info(
        "⚠️ This is an evidence-based model estimate, not a guarantee "
        "of campaign success."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("📋 Campaign Details")

        category = st.selectbox(
            "Category",
            [
                "Technology",
                "Consumer Products",
                "Sustainability",
                "Design",
                "Education",
                "Gaming",
                "Food",
                "Fashion"
            ]
        )

        subcategory_options = {
            "Technology": ["Smart Devices", "Software", "Electronics"],
            "Consumer Products": ["Home", "Lifestyle", "Personal Care"],
            "Sustainability": ["Eco Products", "Green Living", "Clean Technology"],
            "Design": ["Product Design", "Industrial Design", "Creative"],
            "Education": ["EdTech", "Learning", "Training"],
            "Gaming": ["Board Games", "Video Games", "Gaming Hardware"],
            "Food": ["Food Products", "Beverages", "Kitchen"],
            "Fashion": ["Apparel", "Accessories", "Sustainable Fashion"]
        }

        subcategory = st.selectbox(
            "Subcategory",
            subcategory_options[category]
        )

        country = st.selectbox(
            "Country",
            ["India", "United States", "United Kingdom", "Canada", "Australia"]
        )

        launch_month = st.selectbox(
            "Expected Launch Month",
            [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ]
        )

        funding_goal = st.number_input(
            "Funding Goal (₹)",
            min_value=10000.0,
            max_value=500000.0,
            value=75000.0,
            step=5000.0
        )

        campaign_duration = st.number_input(
            "Campaign Duration (Days)",
            min_value=20,
            max_value=45,
            value=30,
            step=1
        )

        reward_tiers = st.number_input(
            "Number of Reward Tiers",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

    with col2:
        st.header("📣 Pre-Launch & Marketing")

        lowest_reward = st.number_input(
            "Lowest Reward (₹)",
            min_value=100.0,
            max_value=10000.0,
            value=499.0,
            step=100.0
        )

        average_reward = st.number_input(
            "Average Reward (₹)",
            min_value=100.0,
            max_value=50000.0,
            value=1999.0,
            step=100.0
        )

        highest_reward = st.number_input(
            "Highest Reward (₹)",
            min_value=500.0,
            max_value=100000.0,
            value=4999.0,
            step=500.0
        )

        video_available = st.selectbox(
            "Campaign Video Available?",
            ["Yes", "No"]
        )

        video_value = 1 if video_available == "Yes" else 0

        number_of_images = st.number_input(
            "Number of Campaign Images",
            min_value=1,
            max_value=30,
            value=8,
            step=1
        )

        story_length = st.number_input(
            "Story Length (Words)",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )

        social_media_followers = st.number_input(
            "Social Media Followers",
            min_value=0,
            max_value=1000000,
            value=5000,
            step=500
        )

        email_list_size = st.number_input(
            "Email List Size",
            min_value=0,
            max_value=100000,
            value=1000,
            step=100
        )

        pre_launch_community_size = st.number_input(
            "Pre-Launch Community Size",
            min_value=0,
            max_value=100000,
            value=500,
            step=50
        )

        marketing_spend = st.number_input(
            "Marketing Budget (₹)",
            min_value=0.0,
            max_value=500000.0,
            value=25000.0,
            step=1000.0
        )

        landing_page_traffic = st.number_input(
            "Expected Landing Page Traffic",
            min_value=0,
            max_value=1000000,
            value=10000,
            step=500
        )

    st.header("📝 Campaign Description")

    description = st.text_area(
        "Enter your campaign description",
        height=180,
        placeholder=(
            "Describe your product, customer problem, solution "
            "and campaign value proposition..."
        )
    )

    if description.strip():
        sentiment_polarity = TextBlob(description).sentiment.polarity
    else:
        sentiment_polarity = 0.0

    sentiment_col1, sentiment_col2 = st.columns(2)

    with sentiment_col1:
        st.metric(
            "Description Sentiment Score",
            round(sentiment_polarity, 3)
        )

    with sentiment_col2:
        if sentiment_polarity > 0.05:
            sentiment_label = "Positive"
        elif sentiment_polarity < -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        st.metric("Sentiment", sentiment_label)

    st.divider()

    predict_button = st.button(
        "🔮 Predict Campaign Success",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame({
            "category": [category],
            "subcategory": [subcategory],
            "country": [country],
            "launch_month": [launch_month],
            "funding_goal": [funding_goal],
            "campaign_duration": [campaign_duration],
            "reward_tiers": [reward_tiers],
            "lowest_reward": [lowest_reward],
            "average_reward": [average_reward],
            "highest_reward": [highest_reward],
            "video_available": [video_value],
            "number_of_images": [number_of_images],
            "story_length": [story_length],
            "social_media_followers": [social_media_followers],
            "email_list_size": [email_list_size],
            "pre_launch_community_size": [pre_launch_community_size],
            "marketing_spend": [marketing_spend],
            "landing_page_traffic": [landing_page_traffic],
            "sentiment_polarity": [sentiment_polarity]
        })

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            unsuccessful_probability = probability[0] * 100
            successful_probability = probability[1] * 100

            st.session_state["prediction"] = int(prediction)
            st.session_state["success_probability"] = float(successful_probability)
            st.session_state["campaign_inputs"] = input_data.iloc[0].to_dict()
            st.session_state["campaign_description"] = description

            st.divider()
            st.header("🎯 Campaign Prediction")

            if prediction == 1:
                st.success("### 🟢 Predicted Outcome: SUCCESSFUL")
            else:
                st.error("### 🔴 Predicted Outcome: UNSUCCESSFUL")

            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                st.metric(
                    "Success Probability",
                    f"{successful_probability:.2f}%"
                )

            with prob_col2:
                st.metric(
                    "Unsuccessful Probability",
                    f"{unsuccessful_probability:.2f}%"
                )

            st.write("### Success Probability")
            st.progress(
                min(max(int(successful_probability), 0), 100)
            )

            if successful_probability >= 70:
                st.success(
                    "The model indicates a relatively strong success "
                    "likelihood based on the information provided."
                )
            elif successful_probability >= 50:
                st.warning(
                    "The model indicates a moderate success likelihood. "
                    "Further campaign improvements may be beneficial."
                )
            else:
                st.error(
                    "The model indicates a relatively low success likelihood. "
                    "Consider improving the campaign before launch."
                )

            with st.expander("📊 View Campaign Inputs"):
                st.dataframe(
                    input_data.T.rename(columns={0: "Value"}),
                    use_container_width=True
                )

            st.info(
                "Go to **Campaign Recommendations** and **Final Launch "
                "Assessment** in the sidebar to interpret this prediction."
            )

        except Exception as e:
            st.error(
                "The model could not process these inputs. "
                f"Technical detail: {e}"
            )

# ------------------------------------------------------------
# HISTORICAL DASHBOARD
# ------------------------------------------------------------

elif page == "📊 Historical Dashboard":

    st.title("📊 Historical Campaign Performance Dashboard")

    try:
        df = load_historical_data()
    except FileNotFoundError:
        st.error(
            f"Dataset '{DATA_FILE}' was not found. "
            "Keep it in the same folder as app.py."
        )
        st.stop()

    st.info(
        "Data source: Synthetic historical crowdfunding dataset created "
        "for the EcoSip project. These results are analytical patterns, "
        "not actual industry success rates."
    )

    st.sidebar.header("🔎 Dashboard Filters")

    categories = ["All"] + sorted(
        df["category"].dropna().astype(str).unique().tolist()
    )

    selected_category = st.sidebar.selectbox(
        "Select Category",
        categories,
        key="dashboard_category"
    )

    countries = ["All"] + sorted(
        df["country"].dropna().astype(str).unique().tolist()
    )

    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries,
        key="dashboard_country"
    )

    filtered_df = df.copy()

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["category"] == selected_category
        ]

    if selected_country != "All":
        filtered_df = filtered_df[
            filtered_df["country"] == selected_country
        ]

    total_campaigns = len(filtered_df)
    successful_campaigns = int(filtered_df["success"].sum())
    unsuccessful_campaigns = total_campaigns - successful_campaigns

    success_rate = (
        successful_campaigns / total_campaigns * 100
        if total_campaigns > 0 else 0
    )

    average_goal = filtered_df["funding_goal"].mean()
    average_raised = filtered_df["amount_raised"].mean()
    average_backers = filtered_df["number_of_backers"].mean()
    average_duration = filtered_df["campaign_duration"].mean()

    st.markdown("## 📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Campaigns", f"{total_campaigns:,}")
    col2.metric("Successful Campaigns", f"{successful_campaigns:,}")
    col3.metric("Unsuccessful Campaigns", f"{unsuccessful_campaigns:,}")
    col4.metric("Historical Success Rate", f"{success_rate:.2f}%")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Average Funding Goal", f"₹{average_goal:,.0f}")
    col6.metric("Average Amount Raised", f"₹{average_raised:,.0f}")
    col7.metric("Average Backers", f"{average_backers:,.0f}")
    col8.metric("Average Duration", f"{average_duration:.1f} days")

    st.divider()

    st.markdown("## 1️⃣ Campaign Outcome Analysis")

    outcome_data = pd.DataFrame({
        "Campaign Outcome": ["Successful", "Unsuccessful"],
        "Campaigns": [successful_campaigns, unsuccessful_campaigns]
    })

    st.bar_chart(outcome_data.set_index("Campaign Outcome"))

    st.markdown("## 2️⃣ Success Rate by Category")

    category_analysis = (
        filtered_df.groupby("category")["success"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(category_analysis)

    st.markdown("## 3️⃣ Average Backers by Category")

    backer_analysis = (
        filtered_df.groupby("category")["number_of_backers"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(backer_analysis)

    st.markdown("## 4️⃣ Campaign Duration Analysis")

    duration_df = filtered_df.copy()

    duration_df["duration_band"] = pd.cut(
        duration_df["campaign_duration"],
        bins=[0, 20, 30, 45, float("inf")],
        labels=["<20 days", "20–30 days", "31–45 days", ">45 days"]
    )

    duration_success = (
        duration_df.groupby("duration_band", observed=False)["success"]
        .mean()
        .mul(100)
    )

    st.bar_chart(duration_success)

    st.markdown("## 5️⃣ Funding Percentage Distribution")

    funding_bands = pd.cut(
        filtered_df["funding_percentage"],
        bins=[
            -float("inf"), 25, 50, 75, 100, 150, float("inf")
        ],
        labels=[
            "<25%", "25–49%", "50–74%", "75–99%", "100–149%", "150%+"
        ]
    )

    funding_distribution = funding_bands.value_counts().sort_index()

    st.bar_chart(funding_distribution)

    st.markdown("## 6️⃣ Funding Goal vs Amount Raised")

    goal_analysis = filtered_df[
        ["funding_goal", "amount_raised"]
    ].head(50).copy()

    goal_analysis.index = range(1, len(goal_analysis) + 1)

    st.line_chart(goal_analysis)

    st.markdown("## 📋 Historical Dataset Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Total Campaigns",
            "Successful Campaigns",
            "Unsuccessful Campaigns",
            "Success Rate",
            "Average Funding Goal",
            "Average Amount Raised",
            "Average Backers",
            "Average Campaign Duration"
        ],
        "Value": [
            f"{total_campaigns:,}",
            f"{successful_campaigns:,}",
            f"{unsuccessful_campaigns:,}",
            f"{success_rate:.2f}%",
            f"₹{average_goal:,.0f}",
            f"₹{average_raised:,.0f}",
            f"{average_backers:,.0f}",
            f"{average_duration:.1f} days"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------
# CAMPAIGN RECOMMENDATIONS
# ------------------------------------------------------------

elif page == "💡 Campaign Recommendations":

    st.title("💡 EcoSip Campaign Improvement Advisor")

    st.write(
        "This module translates the campaign prediction into practical "
        "pre-launch actions."
    )

    st.info(
        "Recommendations are rule-based decision-support suggestions. "
        "They are not guarantees of campaign success."
    )

    if "success_probability" not in st.session_state:
        st.warning(
            "Run the Campaign Predictor first. The recommendation engine "
            "will then use your latest campaign inputs."
        )
    else:
        probability = st.session_state["success_probability"]
        inputs = st.session_state["campaign_inputs"]

        funding_goal = float(inputs["funding_goal"])
        campaign_duration = int(inputs["campaign_duration"])
        email_list = int(inputs["email_list_size"])
        prelaunch_community = int(inputs["pre_launch_community_size"])
        marketing_budget = float(inputs["marketing_spend"])
        landing_page_traffic = int(inputs["landing_page_traffic"])
        number_of_images = int(inputs["number_of_images"])
        video_value = int(inputs["video_available"])

        # User-entered planning assumption for business analysis
        st.markdown("## 💰 Funding Feasibility Assumption")

        average_contribution = st.number_input(
            "Expected Average Contribution (₹)",
            min_value=1,
            value=2500,
            step=100
        )

        required_backers = funding_goal / average_contribution

        # Explicitly label this as an assumption
        expected_conversion = st.number_input(
            "Illustrative Conversion Assumption (%)",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            step=0.1
        )

        conversion_decimal = expected_conversion / 100
        required_visitors = (
            required_backers / conversion_decimal
            if conversion_decimal > 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated Success Probability",
            f"{probability:.2f}%"
        )

        col2.metric(
            "Required Backers",
            f"{required_backers:,.0f}"
        )

        col3.metric(
            "Required Visitors",
            f"{required_visitors:,.0f}"
        )

        if probability >= 80:
            st.success("🟢 Strong Readiness")
        elif probability >= 65:
            st.warning("🟡 Promising – Improvements Recommended")
        elif probability >= 50:
            st.warning("🟠 Significant Risk")
        else:
            st.error("🔴 Major Improvement Required")

        st.markdown("---")
        st.markdown("## 💡 Priority Recommendations")

        recommendations = []

        if probability < 50:
            recommendations.append(
                "🔴 **Overall Readiness:** The estimated probability is below "
                "50%. Strengthen the campaign fundamentals before launch."
            )
        elif probability < 65:
            recommendations.append(
                "🟠 **Overall Readiness:** The campaign is near the decision "
                "boundary. Improve the weakest campaign factors before launch."
            )
        elif probability < 80:
            recommendations.append(
                "🟡 **Overall Readiness:** The campaign appears promising, "
                "but further preparation can reduce execution risk."
            )
        else:
            recommendations.append(
                "🟢 **Overall Readiness:** The campaign shows strong estimated "
                "readiness under the current assumptions."
            )

        if required_backers > 500:
            recommendations.append(
                "💰 **Funding Goal:** The current goal requires more than "
                "500 backers at the assumed average contribution. Review "
                "the funding goal, reward pricing and expected audience size."
            )

        if prelaunch_community < 1000:
            recommendations.append(
                "👥 **Pre-Launch Community:** Build a stronger pre-launch "
                "community before launch through email, social content, "
                "partnerships and early supporters."
            )

        if email_list < 2000:
            recommendations.append(
                "📧 **Email Audience:** Strengthen the pre-launch email list "
                "to create an owned audience for launch communication."
            )

        if marketing_budget < 50000:
            recommendations.append(
                "📣 **Marketing Readiness:** Review whether the planned "
                "marketing budget is sufficient for the required traffic "
                "and acquisition activities."
            )

        if landing_page_traffic < required_visitors:
            recommendations.append(
                "🌐 **Traffic Requirement:** Planned campaign traffic is below "
                "the illustrative visitor requirement. Strengthen acquisition "
                "or revisit the funding/contribution assumptions."
            )

        if video_value == 0:
            recommendations.append(
                "🎥 **Campaign Video:** Add a product demonstration video "
                "to communicate EcoSip's value proposition clearly."
            )

        if number_of_images < 5:
            recommendations.append(
                "🖼️ **Campaign Visuals:** Add more high-quality product images "
                "showing features, usage and benefits."
            )

        if campaign_duration < 20:
            recommendations.append(
                "⏱️ **Campaign Duration:** Review whether the short duration "
                "allows enough time for marketing and backer acquisition."
            )
        elif campaign_duration > 45:
            recommendations.append(
                "⏱️ **Campaign Duration:** Review whether the extended duration "
                "is justified by the marketing and engagement plan."
            )

        for recommendation in recommendations:
            st.write(recommendation)

        st.markdown("---")
        st.markdown("## 🚀 Recommended Action Plan")

        action_plan = [
            "1. Review funding goal feasibility.",
            "2. Validate expected average contribution using reward-tier research.",
            "3. Strengthen the pre-launch community.",
            "4. Build and test the campaign landing page.",
            "5. Prepare product video and campaign visuals.",
            "6. Establish the marketing and traffic-generation plan.",
            "7. Prepare a first-day and first-week launch strategy.",
            "8. Re-run the predictor after improving the campaign inputs."
        ]

        for action in action_plan:
            st.write(action)

# ------------------------------------------------------------
# SOCIAL MEDIA READINESS
# ------------------------------------------------------------

elif page == "📱 Social Media Readiness":

    st.title("📱 EcoSip Pre-Launch Social Media Analyzer")

    st.write(
        "Evaluate the strength of EcoSip's pre-launch digital audience "
        "and social-media readiness."
    )

    st.info(
        "Enter observed or planned values. Do not enter invented engagement "
        "or audience figures. This module is a project-specific decision-"
        "support framework, not an industry benchmark."
    )

    st.sidebar.header("📊 Social Media Inputs")

    instagram_followers = st.sidebar.number_input(
        "Instagram Followers", min_value=0, value=0, step=100,
        key="social_instagram"
    )

    linkedin_followers = st.sidebar.number_input(
        "LinkedIn Followers", min_value=0, value=0, step=100,
        key="social_linkedin"
    )

    twitter_mentions = st.sidebar.number_input(
        "X/Twitter Mentions", min_value=0, value=0, step=10,
        key="social_twitter"
    )

    youtube_views = st.sidebar.number_input(
        "YouTube Product Video Views", min_value=0, value=0, step=100,
        key="social_youtube"
    )

    email_subscribers = st.sidebar.number_input(
        "Email Subscribers", min_value=0, value=0, step=100,
        key="social_email"
    )

    engagement_rate = st.sidebar.number_input(
        "Average Social Engagement Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        key="social_engagement"
    )

    confirmed_partners = st.sidebar.number_input(
        "Confirmed Influencers / Partners",
        min_value=0,
        value=0,
        step=1,
        key="social_partners"
    )

    total_social_audience = (
        instagram_followers + linkedin_followers
    )

    score = 0

    if instagram_followers >= 5000:
        score += 15
    elif instagram_followers >= 1000:
        score += 10
    elif instagram_followers > 0:
        score += 5

    if linkedin_followers >= 3000:
        score += 10
    elif linkedin_followers >= 1000:
        score += 7
    elif linkedin_followers > 0:
        score += 3

    if twitter_mentions >= 500:
        score += 10
    elif twitter_mentions >= 100:
        score += 7
    elif twitter_mentions > 0:
        score += 3

    if youtube_views >= 10000:
        score += 10
    elif youtube_views >= 1000:
        score += 7
    elif youtube_views > 0:
        score += 3

    if email_subscribers >= 5000:
        score += 15
    elif email_subscribers >= 2000:
        score += 10
    elif email_subscribers > 0:
        score += 5

    if engagement_rate >= 5:
        score += 15
    elif engagement_rate >= 2:
        score += 10
    elif engagement_rate > 0:
        score += 5

    if confirmed_partners >= 5:
        score += 15
    elif confirmed_partners >= 2:
        score += 10
    elif confirmed_partners > 0:
        score += 5

    if score >= 80:
        readiness = "🟢 Strong Social Readiness"
    elif score >= 60:
        readiness = "🟡 Moderate Social Readiness"
    elif score >= 40:
        readiness = "🟠 Needs Improvement"
    else:
        readiness = "🔴 Weak Pre-Launch Readiness"

    st.markdown("## 📊 Social Media Readiness")

    col1, col2, col3 = st.columns(3)

    col1.metric("Social Audience", f"{total_social_audience:,}")
    col2.metric("Email Subscribers", f"{email_subscribers:,}")
    col3.metric("Readiness Score", f"{score}/100")

    st.subheader(readiness)

    platform_data = {
        "Instagram": instagram_followers,
        "LinkedIn": linkedin_followers,
        "X/Twitter Mentions": twitter_mentions,
        "YouTube Views": youtube_views,
        "Email Subscribers": email_subscribers,
        "Confirmed Partners": confirmed_partners
    }

    st.markdown("---")
    st.markdown("## 📱 Platform Analysis")
    st.bar_chart(platform_data)

    st.markdown("---")
    st.markdown("## 💡 Social Media Recommendations")

    recommendations = []

    if instagram_followers < 1000:
        recommendations.append(
            "📸 Build Instagram visibility through product demonstrations, "
            "sustainability content and launch teasers."
        )

    if linkedin_followers < 1000:
        recommendations.append(
            "💼 Strengthen LinkedIn presence through founder updates, "
            "product development stories and sustainability-focused posts."
        )

    if twitter_mentions < 100:
        recommendations.append(
            "💬 Increase genuine conversation around EcoSip and monitor "
            "mentions and campaign-related interest."
        )

    if youtube_views < 1000:
        recommendations.append(
            "🎥 Develop a product demonstration video explaining EcoSip's "
            "features and customer benefits."
        )

    if email_subscribers < 2000:
        recommendations.append(
            "📧 Build the email community before launch and use it for "
            "launch announcements and early-backer communication."
        )

    if engagement_rate < 2:
        recommendations.append(
            "📈 Focus on meaningful engagement such as comments, shares, "
            "saves and clicks rather than follower count alone."
        )

    if confirmed_partners < 2:
        recommendations.append(
            "🤝 Explore relevant partnerships with student communities, "
            "sustainability groups, creators and lifestyle communities."
        )

    if not recommendations:
        recommendations.append(
            "🟢 Current social-media indicators meet the basic readiness "
            "thresholds used by this project-specific framework."
        )

    for item in recommendations:
        st.write(item)

    st.markdown("---")
    st.markdown("## ⚠️ Interpretation")

    st.write(
        "A larger audience does not automatically translate into backers. "
        "The important next step is to measure how effectively social audiences "
        "become campaign visitors, engaged users and eventually backers."
    )

# ------------------------------------------------------------
# FINAL LAUNCH ASSESSMENT
# ------------------------------------------------------------

elif page == "🚀 Final Launch Assessment":

    st.title("🚀 EcoSip Final Launch Assessment")

    st.write(
        "This page combines the latest model prediction, funding feasibility "
        "and social-media readiness into one management-level assessment."
    )

    if "success_probability" not in st.session_state:
        st.warning(
            "Run the Campaign Predictor first to generate the final assessment."
        )
    else:
        probability = st.session_state["success_probability"]
        inputs = st.session_state["campaign_inputs"]

        funding_goal = float(inputs["funding_goal"])
        prelaunch = int(inputs["pre_launch_community_size"])
        email_list = int(inputs["email_list_size"])
        marketing_budget = float(inputs["marketing_spend"])
        landing_traffic = int(inputs["landing_page_traffic"])

        average_contribution = st.number_input(
            "Expected Average Contribution (₹)",
            min_value=1,
            value=2500,
            step=100,
            key="final_avg_contribution"
        )

        required_backers = funding_goal / average_contribution

        expected_conversion = st.number_input(
            "Illustrative Conversion Assumption (%)",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            step=0.1,
            key="final_conversion"
        )

        required_visitors = (
            required_backers / (expected_conversion / 100)
        )

        # Social readiness values from the current session inputs
        social_score = 0

        instagram = st.session_state.get("social_instagram", 0)
        linkedin = st.session_state.get("social_linkedin", 0)
        twitter = st.session_state.get("social_twitter", 0)
        youtube = st.session_state.get("social_youtube", 0)
        email_social = st.session_state.get("social_email", 0)
        engagement = st.session_state.get("social_engagement", 0.0)
        partners = st.session_state.get("social_partners", 0)

        if instagram >= 5000:
            social_score += 15
        elif instagram >= 1000:
            social_score += 10
        elif instagram > 0:
            social_score += 5

        if linkedin >= 3000:
            social_score += 10
        elif linkedin >= 1000:
            social_score += 7
        elif linkedin > 0:
            social_score += 3

        if twitter >= 500:
            social_score += 10
        elif twitter >= 100:
            social_score += 7
        elif twitter > 0:
            social_score += 3

        if youtube >= 10000:
            social_score += 10
        elif youtube >= 1000:
            social_score += 7
        elif youtube > 0:
            social_score += 3

        if email_social >= 5000:
            social_score += 15
        elif email_social >= 2000:
            social_score += 10
        elif email_social > 0:
            social_score += 5

        if engagement >= 5:
            social_score += 15
        elif engagement >= 2:
            social_score += 10
        elif engagement > 0:
            social_score += 5

        if partners >= 5:
            social_score += 15
        elif partners >= 2:
            social_score += 10
        elif partners > 0:
            social_score += 5

        # Readiness components
        prediction_component = probability

        funding_component = 100 if landing_traffic >= required_visitors else 50
        audience_component = 100 if prelaunch >= 1000 and email_list >= 2000 else 50
        marketing_component = 100 if marketing_budget >= 50000 else 50
        social_component = social_score

        # Project-specific management readiness score.
        final_score = (
            prediction_component * 0.50
            + funding_component * 0.15
            + audience_component * 0.15
            + marketing_component * 0.10
            + social_component * 0.10
        )

        st.markdown("## 📊 Overall Readiness")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Model Success Probability",
            f"{probability:.2f}%"
        )

        c2.metric(
            "Social Readiness",
            f"{social_score}/100"
        )

        c3.metric(
            "Overall Project Readiness",
            f"{final_score:.1f}/100"
        )

        if final_score >= 80:
            st.success("🟢 STRONG READINESS")
            final_decision = "Proceed toward launch preparation."
        elif final_score >= 65:
            st.warning("🟡 PROMISING — IMPROVEMENTS RECOMMENDED")
            final_decision = "Continue preparation and address the identified gaps."
        elif final_score >= 50:
            st.warning("🟠 SIGNIFICANT RISK")
            final_decision = "Do not launch yet; strengthen the major risk areas."
        else:
            st.error("🔴 MAJOR REWORK REQUIRED")
            final_decision = "Improve the campaign fundamentals before launch."

        st.subheader("Final Management Recommendation")
        st.write(final_decision)

        st.markdown("---")
        st.markdown("## 🎯 Key Decision Factors")

        factors = pd.DataFrame({
            "Factor": [
                "AI Model Probability",
                "Required Backers",
                "Required Visitors",
                "Planned Visitors",
                "Pre-Launch Community",
                "Email List",
                "Marketing Budget",
                "Social Readiness"
            ],
            "Value": [
                f"{probability:.2f}%",
                f"{required_backers:,.0f}",
                f"{required_visitors:,.0f}",
                f"{landing_traffic:,.0f}",
                f"{prelaunch:,}",
                f"{email_list:,}",
                f"₹{marketing_budget:,.0f}",
                f"{social_score}/100"
            ]
        })

        st.dataframe(
            factors,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("## 🔧 Highest-Priority Actions")

        actions = []

        if probability < 65:
            actions.append(
                "Improve the campaign inputs that are associated with the lower model probability."
            )

        if landing_traffic < required_visitors:
            actions.append(
                "Increase planned campaign traffic or revisit the funding/contribution assumptions."
            )

        if prelaunch < 1000:
            actions.append(
                "Build the pre-launch community before launch."
            )

        if email_list < 2000:
            actions.append(
                "Strengthen the owned email audience."
            )

        if marketing_budget < 50000:
            actions.append(
                "Review the marketing budget and acquisition plan."
            )

        if social_score < 60:
            actions.append(
                "Strengthen social-media and partnership readiness."
            )

        if not actions:
            actions.append(
                "Maintain the current preparation level and validate assumptions with additional evidence."
            )

        for action in actions:
            st.write(f"• {action}")

        st.markdown("---")
        st.markdown("## 🧪 Model Validation")

        try:
            validation_df = pd.read_csv(VALIDATION_FILE)

            # Try to identify an accuracy column if available.
            st.write(
                "The external validation dataset contains the real-campaign "
                "comparison used in Step 6."
            )

            st.metric(
                "External Validation Sample",
                f"{len(validation_df)} campaigns"
            )

            st.caption(
                "Step 6 external validation should be interpreted cautiously "
                "because several real-campaign pre-launch variables were unavailable "
                "and had to be imputed from the synthetic training data."
            )

        except FileNotFoundError:
            st.info(
                "Step 6 validation results file was not found. "
                "Run step6_testing_validation.py if you want to display validation evidence."
            )

        st.markdown("---")
        st.caption(
            "Important: All predictions and readiness scores are decision-support "
            "estimates. They do not guarantee crowdfunding success."
        )
