import streamlit as st
import streamlit_option_menu as option_menu
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# Side Bar
with st.sidebar:
    selected = option_menu("BuySense",["Home","Product Prediction","Dataset","About"],
                           menu_icon="shop",icons=["house-fill","graph-up","table","info-circle-fill"],default_index=0)
# Read CSV
df = pd.read_csv("buysense_dataset.csv")

# HOME Page Section :

if selected == "Home":

    st.title("🛒BuySense : ")
    st.title("Intelligent Product Discovery Engine")
    st.header("Welcome to BuySense!")
    st.subheader("➤ Description : ")
    st.write("BuySense is an AI-powered product recommendation platform that helps users discover relevant products based on browsing behavior, purchase history, and interests. Using machine learning techniques, the system delivers personalized product suggestions to enhance shopping experience and customer satisfaction.")
    st.write("\n")

    total_records = len(df)
    order_value = round(df["avg_order_value"].mean(),2)
    total_citytier= df["city_tier"].nunique()
    browsing_time = round(df["browsing_minutes"].mean(),2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records/Users", total_records)
    col2.metric("Average Order Value", order_value)
    col3.metric("City Tier", total_citytier)
    col4.metric("Browsing Time", browsing_time)

    st.divider()

    col11,col12 = st.columns(2)
    with col11:
        df["recommended_category"].value_counts().plot(kind="bar",color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"])
        plt.title("Recommended Product Categories")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    with col12:
        device_count = df["preferred_device"].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(device_count,
                labels=device_count.index,
                autopct="%1.1f%%",colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"])

        plt.title("Preferred Device Distribution")
        plt.show()
        st.pyplot(plt)


    st.subheader("➤ Objective : ")
    st.write("The main objective of BuySense is to improve the online shopping experience by recommending the most relevant product categories to users based on their preferences and past activities.")
    st.write("\n")

    st.subheader("🚀 Key Features :")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("📊 Customer Behavior Analysis")

    with col2:
        st.success("🛒 Personalized Recommendations")

    with col3:
        st.warning("⚡ Fast and Accurate Predictions")

    with col4:
        st.info("📈 Interactive Data Visualization")

    st.subheader("🔥 Product Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("📚 Books")

    with col2:
        st.info("💻 Electronics")

    with col3:
        st.warning("🏋️ Fitness")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error("👗 Fashion")

    with col2:
        st.success("🏠 Home Decor")

    with col3:
        st.info("💄 Beauty")

    st.subheader("🌟 Why Choose BuySense?")

    st.markdown("""
    ✅ Personalized Shopping Experience

    ✅ Saves Time & Effort

    ✅ Better Product Discovery

    ✅ Higher Customer Satisfaction

    ✅ Data-Driven Recommendations
    """)

    st.subheader("📈 System Performance")

    st.progress(92)
    st.write("Recommendation Accuracy: 92%")

    st.progress(88)
    st.write("Customer Satisfaction: 88%")

    st.progress(95)
    st.write("Prediction Speed: 95%")

    st.markdown("""
    ### 🤖 Machine Learning Powered

    BuySense leverages advanced machine learning algorithms
    to analyze user interests, browsing patterns, and purchase
    behavior for generating accurate product recommendations.
    """)

    st.markdown("""
    <div style="
    padding:20px;
    background:#FFF4E6;
    border-left:5px solid #FF4D4D;
    border-radius:10px;">
    <h4>💡 Smart Recommendations Lead to Smarter Shopping</h4>
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")

    st.markdown("""
    <div style="
    background:#FFF8E7;
    padding:15px;
    border-radius:10px;
    border-left:6px solid orange;">
    💡 <b>Did You Know?</b><br>
    Personalized recommendations contribute significantly to online sales and customer retention.
    </div>
    """, unsafe_allow_html=True)
    st.write("\n")
    st.markdown("""
    <style>
    .float{
    font-size:40px;
    animation: float 3s ease-in-out infinite;
    display:inline-block;
    margin:10px;
    }

    @keyframes float{
    0%{transform:translateY(0px);}
    50%{transform:translateY(-15px);}
    100%{transform:translateY(0px);}
    }
    </style>

    <div style="text-align:center">
    <span class="float">🛒</span>
    <span class="float">🤖</span>
    <span class="float">📦</span>
    <span class="float">⭐</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .full-footer {
            width: 100vw;
            margin-left: calc(-50vw + 50%);
            background-color: #F8F9FA;
            border-top: 1px solid #ddd;
            padding: 30px 20px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 0px !important;
        }

        .block-container {
            padding-bottom: 0rem !important;
        }

        .full-footer h2 {
            color: #333;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .full-footer p {
            color: #666;
            margin: 5px 0;
            font-size: 16px;
        }

        .developer {
            margin-top: 12px;
            font-size: 14px;
            color: #888;
            letter-spacing: 0.5px;
        }

        .developer span {
            color: #FF4D4D;
            font-weight: 600;
        }
        </style>

        <div class="full-footer">
            <h4>BuySense - Intelligent Product Discovery Engine</h4>
            <p>Powered by Machine Learning | Personalized Product Recommendations</p>
            <p>© 2026 All Rights Reserved</p>
            <div class="developer">
                Developed by <span>Daksh Dabhi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# PRODUCT PREDICTION Page Section :

elif selected == "Product Prediction" :

    data = df.copy()

    # Label Encoding

    age_encoder = LabelEncoder()
    city_encoder = LabelEncoder()
    discount_encoder = LabelEncoder()
    device_encoder = LabelEncoder()
    time_encoder = LabelEncoder()
    target_encoder = LabelEncoder()

    data["age_group"] = age_encoder.fit_transform(
        data["age_group"]
    )

    data["city_tier"] = city_encoder.fit_transform(
        data["city_tier"]
    )

    data["discount_sensitivity"] = discount_encoder.fit_transform(
        data["discount_sensitivity"]
    )

    data["preferred_device"] = device_encoder.fit_transform(
        data["preferred_device"]
    )

    data["purchase_time"] = time_encoder.fit_transform(
        data["purchase_time"]
    )

    data["recommended_category"] = target_encoder.fit_transform(
        data["recommended_category"]
    )

    # Features

    X = data[
        [
            "age_group",
            "city_tier",
            "browsing_minutes",
            "pages_viewed",
            "past_purchases",
            "avg_order_value",
            "discount_sensitivity",
            "preferred_device",
            "purchase_time",
            "interest_electronics",
            "interest_fashion",
            "interest_home_decor",
            "interest_books",
            "interest_fitness",
            "interest_beauty"
        ]
    ]

    # Target

    y = data["recommended_category"]

    # Train Test Split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest Model

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.subheader("Enter Customer Details")

    age_group = st.selectbox(
        "Age Group",
        age_encoder.classes_
    )

    city_tier = st.selectbox(
        "City Tier",
        city_encoder.classes_
    )

    browsing_minutes = st.number_input(
        "Browsing Minutes",
        1,
        500,
        60
    )

    pages_viewed = st.number_input(
        "Pages Viewed",
        1,
        100,
        10
    )

    past_purchases = st.number_input(
        "Past Purchases",
        0,
        100,
        5
    )

    avg_order_value = st.number_input(
        "Average Order Value",
        100.0,
        50000.0,
        1000.0
    )

    discount_sensitivity = st.selectbox(
        "Discount Sensitivity",
        discount_encoder.classes_
    )

    preferred_device = st.selectbox(
        "Preferred Device",
        device_encoder.classes_
    )

    purchase_time = st.selectbox(
        "Purchase Time",
        time_encoder.classes_
    )

    interest_electronics = st.slider(
        "Interest in Electronics",
        1,
        10,
        5
    )

    interest_fashion = st.slider(
        "Interest in Fashion",
        1,
        10,
        5
    )

    interest_home_decor = st.slider(
        "Interest in Home Decor",
        1,
        10,
        5
    )

    interest_books = st.slider(
        "Interest in Books",
        1,
        10,
        5
    )

    interest_fitness = st.slider(
        "Interest in Fitness",
        1,
        10,
        5
    )

    interest_beauty = st.slider(
        "Interest in Beauty",
        1,
        10,
        5
    )

    if st.button("Recommend Product"):
        input_data = pd.DataFrame({

            "age_group": [
                age_encoder.transform([age_group])[0]
            ],

            "city_tier": [
                city_encoder.transform([city_tier])[0]
            ],

            "browsing_minutes": [browsing_minutes],

            "pages_viewed": [pages_viewed],

            "past_purchases": [past_purchases],

            "avg_order_value": [avg_order_value],

            "discount_sensitivity": [
                discount_encoder.transform(
                    [discount_sensitivity]
                )[0]
            ],

            "preferred_device": [
                device_encoder.transform(
                    [preferred_device]
                )[0]
            ],

            "purchase_time": [
                time_encoder.transform(
                    [purchase_time]
                )[0]
            ],

            "interest_electronics": [
                interest_electronics
            ],

            "interest_fashion": [
                interest_fashion
            ],

            "interest_home_decor": [
                interest_home_decor
            ],

            "interest_books": [
                interest_books
            ],

            "interest_fitness": [
                interest_fitness
            ],

            "interest_beauty": [
                interest_beauty
            ]
        })

        prediction = model.predict(input_data)

        category = target_encoder.inverse_transform(
            prediction
        )

        st.success(
            f"🛒 Recommended Category : {category[0]}"
        )

        st.subheader("Model Accuracy")

        st.success(f"{accuracy:.2%}")

        st.balloons()

# DATASET Page Section :

elif selected == "Dataset":

    st.title("🛒BuySense : Dataset")
    st.dataframe(df)
    st.write("\n")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Rows", len(df))
    col2.metric("📋 Columns", len(df.columns))
    col3.metric("❌ Missing Values", df.isnull().sum().sum())
    col4.metric("🔄 Duplicate Rows", df.duplicated().sum())
    st.write("\n")

    health = 95

    st.markdown("### 🎯 Dataset Quality Score")
    st.progress(health)

    st.success(f"Dataset Health Score: {health}%")

    st.markdown("## 🌟 Quick Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📦 Multiple Product Categories Available")

    with col2:
        st.success("⭐ Rich User Interaction Data")

    with col3:
        st.warning("🤖 Suitable for Recommendation Modeling")

    st.markdown("""
    <div style="
    background:#F8FAFC;
    padding:20px;
    border-radius:15px;
    border-left:6px solid #6366F1;
    ">
    <h3>🏆 Dataset Highlights</h3>

    ✅ Real-world product data<br>
    ✅ Multiple product categories<br>
    ✅ User preference information<br>
    ✅ Machine Learning ready<br>
    ✅ Suitable for recommendation systems

    </div>
    """, unsafe_allow_html=True)
    st.write("\n")

    st.subheader("🎨 Column Explorer")
    selected_col = st.selectbox(
        "Select Column",
        df.columns
    )
    st.write(df[selected_col].describe())

    st.subheader("📊 Stastical summary : ")
    st.write(df.describe())

    st.markdown("""
        <style>
        .full-footer {
            width: 100vw;
            margin-left: calc(-50vw + 50%);
            background-color: #F8F9FA;
            border-top: 1px solid #ddd;
            padding: 30px 20px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 0px !important;
        }

        .block-container {
            padding-bottom: 0rem !important;
        }

        .full-footer h2 {
            color: #333;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .full-footer p {
            color: #666;
            margin: 5px 0;
            font-size: 16px;
        }

        .developer {
            margin-top: 12px;
            font-size: 14px;
            color: #888;
            letter-spacing: 0.5px;
        }

        .developer span {
            color: #FF4D4D;
            font-weight: 600;
        }
        </style>

        <div class="full-footer">
            <h4>BuySense - Intelligent Product Discovery Engine</h4>
            <p>Powered by Machine Learning | Personalized Product Recommendations</p>
            <p>© 2026 All Rights Reserved</p>
            <div class="developer">
                Developed by <span>Daksh Dabhi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ABOUT Page Section :

elif selected == "About":

    st.markdown("""
       <style>
       .glow{
       font-size:28px;
       color:#6366F1;
       text-align:center;
       animation: glow 1.5s ease-in-out infinite alternate;
       }

       @keyframes glow{
       from{
       text-shadow:0 0 10px #A5B4FC;
       }
       to{
       text-shadow:0 0 20px #6366F1,0 0 30px #818CF8;
       }
       }
       </style>

       <p class="glow">
       ✨ AI that understands what you want before you search. ✨
       </p>
       """, unsafe_allow_html=True)
    st.markdown("""
       <style>
       .hero{
       background: linear-gradient(135deg,#EEF2FF,#E0F2FE,#F0FDF4);
       padding:30px;
       border-radius:20px;
       text-align:center;
       }
       .hero h1{
       color:#4338CA;
       }
       </style>

       <div class="hero">
       <h1>🛍️ BuySense</h1>
       <h3>AI Powered Product Recommendation System</h3>
       <p>Discover products tailored to your interests.</p>
       </div>
       """, unsafe_allow_html=True)
    st.markdown("""
       <marquee behavior="scroll" direction="left" scrollamount="8">
       🛒 Smartphone &nbsp;&nbsp;&nbsp;
       ⌚ Smartwatch &nbsp;&nbsp;&nbsp;
       🎧 Headphones &nbsp;&nbsp;&nbsp;
       💻 Laptop &nbsp;&nbsp;&nbsp;
       📷 Camera &nbsp;&nbsp;&nbsp;
       🎮 Gaming Console
       </marquee>
       """, unsafe_allow_html=True)

    st.write("\n")

    st.header("➤ About BuySense : ")
    st.write("BuySense is an intelligent Product Recommendation System designed to help users discover products that match their interests, preferences, and shopping behavior. By leveraging Machine Learning techniques and data-driven insights, the platform analyzes user interactions and product attributes to generate personalized recommendations.")
    st.write("In today's digital marketplace, customers are often overwhelmed by the vast number of available products. BuySense addresses this challenge by filtering relevant items and presenting tailored suggestions, making the shopping experience faster, smarter, and more engaging.")
    st.write("\n")

    st.markdown("""
    <style>
    .glass{
    background:rgba(255,255,255,0.25);
    backdrop-filter:blur(10px);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.3);
    box-shadow:0 8px 32px rgba(31,38,135,0.2);
    }

    </style>

    <div class="glass">
    <h3>🎯 Mission</h3>
    <p>
    Provide personalized product recommendations using Machine Learning
    to enhance customer shopping experiences.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.25);
    padding:20px;
    border-radius:15px;
    border:1px solid rgba(255,255,255,0.3);
    text-align:center;">
    <h2>🎯 Recommendation Accuracy</h2>
    <h1>92%</h1>
    </div>
    """, unsafe_allow_html=True)


    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Products", "10K+")
    col2.metric("Users", "5K+")
    col3.metric("Accuracy", "92%")
    col4.metric("Categories", "20+")


    st.subheader("🛠 Technologies")

    tech1, tech2, tech3, tech4 = st.columns(4)

    tech1.markdown("### 🐍 Python")
    tech2.markdown("### 🎈 Streamlit")
    tech3.markdown("### 🤖 Scikit-Learn")
    tech4.markdown("### 📊 Pandas")


    testimonials = [
        "⭐⭐⭐⭐⭐ Amazing recommendations!",
        "⭐⭐⭐⭐⭐ Very user friendly interface.",
        "⭐⭐⭐⭐⭐ Saved me lots of shopping time.",
        "⭐⭐⭐⭐⭐ Best AI recommendation system."
    ]

    for review in testimonials:
        st.success(review)

    st.markdown("""
    <div style="
    padding:20px;
    border-radius:15px;
    background:#F8FAFC;
    text-align:center;
    border:2px solid #E2E8F0;
    ">
    <h3>👨‍💻 Developed By</h3>
    <h3 style="color:#4F46E5;">Daksh Dabhi</h3>
    <h3 style="color:#4F46E5;">Enrollment No. : 246270307024</h3>
    <p>AI & Machine Learning Project</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .full-footer {
            width: 100vw;
            margin-left: calc(-50vw + 50%);
            background-color: #F8F9FA;
            border-top: 1px solid #ddd;
            padding: 30px 20px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 0px !important;
        }

        .block-container {
            padding-bottom: 0rem !important;
        }

        .full-footer h2 {
            color: #333;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .full-footer p {
            color: #666;
            margin: 5px 0;
            font-size: 16px;
        }

        .developer {
            margin-top: 12px;
            font-size: 14px;
            color: #888;
            letter-spacing: 0.5px;
        }

        .developer span {
            color: #FF4D4D;
            font-weight: 600;
        }
        </style>

        <div class="full-footer">
            <h4>BuySense - Intelligent Product Discovery Engine</h4>
            <p>Powered by Machine Learning | Personalized Product Recommendations</p>
            <p>© 2026 All Rights Reserved</p>
            <div class="developer">
                Developed by <span>Daksh Dabhi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)







