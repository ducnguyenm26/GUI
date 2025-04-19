# Standard library imports
import re
import warnings
import sys
import subprocess

# Initialize numpy first
import numpy as np
import numpy.core.multiarray

# Other data science libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Install and import NLTK
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# NLTK imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Install and import Surprise
try:
    from surprise import Dataset, Reader, SVD
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-surprise'])
    from surprise import Dataset, Reader, SVD

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure warnings and style
warnings.filterwarnings('ignore')
plt.style.use('default')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_string(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into string
    return ' '.join(tokens)

# Set matplotlib style
plt.style.use('default')

# Customize plot appearance
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#f0f0f0'
})

st.set_page_config(
    page_title="Shopee Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ee4d2d;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .stTitle {
        color: #ee4d2d;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛍️ Hệ Thống Khoa Học Dữ Liệu")
st.write("## Hệ Thống Gợi Ý Sản Phẩm Shopee")

@st.cache_data
def display_product(product):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("#### 💶 Chi Tiết Sản Phẩm:")
        st.write(f"""
        - **Tên:** {product['product_name']}
        - **Danh Mục:** {product['category']}
        - **Danh Mục Con:** {product['sub_category']}
        - **Giá:** {product['price']:,.0f}đ
        - **Đánh Giá:** {product['rating']:.1f}/5
        - **Mô tả:** {product['description'][:200]}...
        """)
    
    with col2:
        if product['image'] and str(product['image']).strip() != '':
            try:
                st.image(product['image'], width=200)
            except:
                pass

@st.cache_data
def preprocess_all_descriptions(df):
    """Pre-process all product descriptions at once"""
    return df['description'].apply(lambda x: preprocess_string(str(x).lower()))

@st.cache_data
def create_tfidf_matrix(processed_descriptions):
    """Create TF-IDF matrix once"""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(processed_descriptions)

@st.cache_data
def train_svd_model(ratings_data):
    """Train SVD model once"""
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(ratings_data[['user_id', 'product_id', 'rating']], reader)
    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(data.build_full_trainset())
    return algo

# Modify load_data() to include preprocessing
@st.cache_data
def load_data():
    products_df = pd.read_csv('cleaned_product_data.csv')
    ratings_df = pd.read_csv('cleaned_rating_data.csv')
    
    # Pre-process descriptions
    products_df['processed_description'] = preprocess_all_descriptions(products_df)
    
    # Create TF-IDF matrix
    tfidf_matrix = create_tfidf_matrix(products_df['processed_description'])
    
    # Train SVD model
    svd_model = train_svd_model(ratings_df)
    
    return products_df, ratings_df, tfidf_matrix, svd_model

# Update data loading
products_df, ratings_df, tfidf_matrix, svd_model = load_data()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Shopee.svg/2560px-Shopee.svg.png", width=200)
    menu = ["Tổng Quan", "Phân Tích Dữ Liệu", "Gợi Ý Theo Nội Dung", "Gợi Ý Theo Đánh Giá", "Gợi Ý Kết Hợp"]
    choice = st.selectbox('Menu', menu)

if choice == 'Tổng Quan':    
    st.subheader("Tổng Quan Hệ Thống")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        #### 🏪 Giới thiệu về Shopee
        Shopee là nền tảng thương mại điện tử hàng đầu Đông Nam Á và Đài Loan. Ra mắt năm 2015, 
        nền tảng thương mại Shopee được xây dựng nhằm cung cấp cho người dùng trải nghiệm mua sắm 
        trực tuyến dễ dàng, an toàn và nhanh chóng thông qua những thanh toán và hỗ trợ đảm bảo.

        #### 🎯 Mục tiêu
        Xây dựng một hệ thống Recommendation System hiệu quả cho Shopee.vn để đề xuất và gợi ý 
        sản phẩm cho người dùng/khách hàng.
        """)
    
    with col2:
        st.image("https://cf.shopee.vn/file/88d3ec246773d6a4ebd9c9f87b1ea3db", use_column_width=True)
    
    st.write("""
    #### 🔍 Hệ thống sử dụng ba phương pháp chính:
    
    1. **Gợi ý dựa trên nội dung:**
       - Phân tích mô tả sản phẩm
       - So sánh độ tương đồng
       - Đề xuất dựa trên đặc điểm sản phẩm
    
    2. **Gợi ý dựa trên đánh giá (SVD):**
       - Sử dụng ma trận đánh giá người dùng
       - Áp dụng kỹ thuật SVD
       - Dự đoán điểm đánh giá
    
    3. **Phương pháp kết hợp:**
       - Kết hợp ưu điểm của cả hai phương pháp
       - Cân bằng giữa nội dung và đánh giá
       - Đề xuất chính xác hơn
    """)

elif choice == 'Phân Tích Dữ Liệu':
    st.subheader("📊 Phân Tích Dữ Liệu")
    
    tab1, tab2, tab3 = st.tabs(["📋 Tổng Quan", "📈 Phân Tích Chi Tiết", "🔍 Thống Kê"])
    
    with tab1:
        st.write("#### Kiểu dữ liệu của các bảng")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bảng Đánh Giá (Rating):**")
            st.write("Kiểu dữ liệu của các cột trong rating_data_clean:")
            st.code("""
product_id     int64
user_id        int64
user          object
rating         int64
dtype: object
            """)
            
        with col2:
            st.write("**Bảng Sản Phẩm (Product):**")
            st.write("Kiểu dữ liệu của các cột trong product_data_clean:")
            st.code("""
product_id        int64
product_name     object
category         object
sub_category     object
link             object
image            object
price           float64
rating          float64
description      object
dtype: object
            """)
        
        st.write("#### 📊 Thống Kê Cơ Bản")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Dữ Liệu Sản Phẩm:**")
            st.write(f"📦 Tổng số sản phẩm: {len(products_df):,}")
            st.write(f"🏷️ Số danh mục: {len(products_df['category'].unique())}")
            st.write(f"📑 Số danh mục con: {len(products_df['sub_category'].unique())}")
            st.write(f"💰 Giá trung bình: {products_df['price'].mean():,.2f}đ")
            st.write(f"⭐ Đánh giá trung bình: {products_df['rating'].mean():.2f}/5")
            
        with col2:
            st.info("**Dữ Liệu Đánh Giá:**")
            st.write(f"📝 Tổng số đánh giá: {len(ratings_df):,}")
            st.write(f"👥 Số người dùng: {len(ratings_df['user_id'].unique()):,}")
            st.write(f"⭐ Điểm đánh giá trung bình: {ratings_df['rating'].mean():.2f}/5")
            st.write(f"📊 Số đánh giá/người dùng: {len(ratings_df)/len(ratings_df['user_id'].unique()):.1f}")
    
    with tab2:
        st.write("#### 📈 Phân Tích Chi Tiết")
        
        # Sample Data Display
        st.write("#### Dữ liệu mẫu")
        tab_data1, tab_data2 = st.tabs(["Dữ liệu Đánh Giá", "Dữ liệu Sản Phẩm"])
        
        with tab_data1:
            st.dataframe(ratings_df.head())
            
        with tab_data2:
            st.dataframe(products_df.head())
        
        # Rating Distribution from DataAnalysis.ipynb
        st.write("#### Phân Tích Đánh Giá")
        col1, col2 = st.columns(2)
        
        with col1:
            # Create rating distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(data=ratings_df, x='rating', bins=5, color='skyblue')
            plt.title('Phân bố điểm đánh giá', fontsize=12, pad=15)
            plt.xlabel('Điểm đánh giá', fontsize=10)
            plt.ylabel('Số lượng', fontsize=10)
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            plt.close()
        
        with col2:
            # Average rating by category from product data
            avg_rating_by_cat = products_df.groupby('category')['rating'].mean().sort_values(ascending=True)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=avg_rating_by_cat.values, y=avg_rating_by_cat.index, color='skyblue')
            plt.title('Điểm đánh giá trung bình theo danh mục', fontsize=12, pad=15)
            plt.xlabel('Điểm đánh giá trung bình', fontsize=10)
            st.pyplot(plt)
            plt.close()
        
        # Description Length Analysis
        st.write("#### Phân Tích Độ Dài Mô Tả")
        col1, col2 = st.columns(2)
        
        with col1:
            # Description length distribution
            plt.figure(figsize=(10, 6))
            description_lengths = products_df['description'].str.len()
            sns.histplot(data=description_lengths, bins=30, color='purple')
            plt.title('Phân bố độ dài mô tả sản phẩm', fontsize=12, pad=15)
            plt.xlabel('Độ dài mô tả', fontsize=10)
            plt.ylabel('Số lượng sản phẩm', fontsize=10)
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            plt.close()
        
        with col2:
            # Box plot of description length by category
            plt.figure(figsize=(10, 6))
            description_by_category = pd.DataFrame({
                'category': products_df['category'],
                'description_length': products_df['description'].str.len()
            })
            sns.boxplot(data=description_by_category, x='description_length', y='category', color='purple')
            plt.title('Phân bố độ dài mô tả theo danh mục', fontsize=12, pad=15)
            plt.xlabel('Độ dài mô tả', fontsize=10)
            plt.ylabel('Danh mục', fontsize=10)
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            plt.close()

        # Price Analysis
        st.write("#### Phân Tích Giá")
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=products_df, x='price', bins=50, color='skyblue')
            plt.title('Phân bố giá sản phẩm', fontsize=12, pad=15)
            plt.xlabel('Giá (VNĐ)', fontsize=10)
            plt.ylabel('Số lượng', fontsize=10)
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            plt.close()
        
        with col2:
            # Box plot of prices by category
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=products_df, x='price', y='category', color='skyblue')
            plt.title('Phân bố giá theo danh mục', fontsize=12, pad=15)
            plt.xlabel('Giá (VNĐ)', fontsize=10)
            plt.ylabel('Danh mục', fontsize=10)
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            plt.close()
        
        # Category Analysis
        st.write("#### Phân Tích Danh Mục")
        col1, col2 = st.columns(2)
        
        with col1:
            # Product distribution by category
            plt.figure(figsize=(10, 6))
            category_counts = products_df['category'].value_counts()
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(category_counts)))
            plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            plt.title('Phân bố sản phẩm theo danh mục', fontsize=12, pad=15)
            st.pyplot(plt)
            plt.close()
        
        with col2:
            # Top 10 subcategories
            plt.figure(figsize=(10, 6))
            top_subcats = products_df['sub_category'].value_counts().head(10)
            sns.barplot(x=top_subcats.values, y=top_subcats.index, color='skyblue')
            plt.title('Top 10 danh mục con phổ biến', fontsize=12, pad=15)
            plt.xlabel('Số lượng sản phẩm', fontsize=10)
            plt.grid(True, alpha=0.3)
            st.pyplot(plt)
            plt.close()
    
    with tab3:
        st.write("#### 📊 Thống Kê Chi Tiết theo Danh Mục")
        
        category_stats = products_df.groupby('category').agg({
            'product_id': 'count',
            'price': ['mean', 'min', 'max'],
            'rating': ['mean', 'count']
        }).round(2)
        
        category_stats.columns = [
            'Số Sản Phẩm', 
            'Giá TB', 'Giá Min', 'Giá Max',
            'Đánh Giá TB', 'Số Lượng Đánh Giá'
        ]
        
        for col in ['Giá TB', 'Giá Min', 'Giá Max']:
            category_stats[col] = category_stats[col].apply(lambda x: f"{x:,.0f}đ")
            
        st.dataframe(
            category_stats,
            use_container_width=True,
            height=400
        )
        
        # Category Distribution Bar Plot
        fig_cat_dist = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Số Lượng Sản Phẩm theo Danh Mục",
            labels={"x": "Số Lượng", "y": "Danh Mục"}
        )
        st.plotly_chart(fig_cat_dist, use_container_width=True)

elif choice == 'Gợi Ý Theo Nội Dung':
    st.subheader("🔍 Gợi Ý Dựa Trên Nội Dung")
    
    with st.container():
        search_term = st.text_input("🔎 Tìm kiếm sản phẩm:")

        if search_term:
            with st.spinner('🔄 Đang tìm kiếm sản phẩm tương tự...'):
                matches = products_df[products_df['product_name'].str.contains(search_term, case=False, na=False)]

                if matches.empty:
                    st.warning("⚠️ Không tìm thấy sản phẩm phù hợp.")
                else:
                    selected_product = st.selectbox(
                        "Chọn sản phẩm:",
                        matches['product_name'].tolist()
                    )

                    if selected_product:
                        product_details = products_df[products_df['product_name'] == selected_product].iloc[0]

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write("#### 📦 Chi Tiết Sản Phẩm:")
                            st.write(f"""
                            - **Tên:** {product_details['product_name']}
                            - **Danh Mục:** {product_details['category']}
                            - **Danh Mục Con:** {product_details['sub_category']}
                            - **Giá:** {product_details['price']:,.0f}đ
                            - **Đánh Giá:** {product_details['rating']:.1f}/5
                            """)

                        with col2:
                            if product_details['image'] and str(product_details['image']).strip() != '':
                                try:
                                    st.image(product_details['image'], width=200)
                                except:
                                    pass

                        # In 'Gợi Ý Theo Nội Dung' section, replace the content-based recommendation code:
                        if st.button("🔍 Xem Gợi Ý Sản Phẩm Tương Tự", use_container_width=True):
                            with st.spinner('Đang tìm kiếm sản phẩm tương tự...'):
                                products_df['processed_description'] = products_df['description'].apply(
                                    lambda x: preprocess_string(str(x).lower())
                                )

                                vectorizer = TfidfVectorizer()
                                tfidf_matrix = vectorizer.fit_transform(products_df['processed_description'])

                                product_idx = products_df[products_df['product_name'] == selected_product].index[0]
                                similarities = cosine_similarity(tfidf_matrix[product_idx], tfidf_matrix)
                                similar_indices = similarities.argsort()[0][-6:-1][::-1]

                                st.write("#### 🏷️ Sản Phẩm Tương Tự:")
                                displayed_count = 0

                                for idx in similar_indices:
                                    prod = products_df.iloc[idx]
                                    if prod['product_id'] != product_details['product_id']:
                                        with st.container():
                                            col1, col2 = st.columns([1, 3])

                                            with col1:
                                                if prod['image'] and str(prod['image']).strip() != '':
                                                    try:
                                                        st.image(prod['image'], width=150)
                                                    except:
                                                        pass

                                            with col2:
                                                st.write(f"**{prod['product_name']}**")
                                                st.write(f"💰 Giá: {prod['price']:,.0f}đ")
                                                st.write(f"⭐ Đánh giá: {prod['rating']:.1f}/5")
                                                st.write(f"📁 Danh mục: {prod['category']} > {prod['sub_category']}")
                                                st.write(f"📖 Mô tả: {prod['description'][:200]}...")
                                            st.write("---")
                                        displayed_count += 1

elif choice == 'Gợi Ý Theo Đánh Giá':
    st.subheader("👥 Gợi Ý Dựa Trên Đánh Giá (SVD)")
    
    with st.container():
        user_id = st.number_input("👤 Nhập ID người dùng:", min_value=1, value=1)
        
        if st.button("🔍 Xem Gợi Ý", use_container_width=True):
            with st.spinner('🔄 Đang phân tích sở thích và tìm kiếm sản phẩm phù hợp...'):
                # Kiểm tra xem người dùng đã có đánh giá nào chưa
                user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                
                if user_ratings.empty:
                    st.warning("⚠️ Người dùng chưa có đánh giá nào!")
                else:
                    # Huấn luyện mô hình SVD
                    reader = Reader(rating_scale=(0, 5))
                    data = Dataset.load_from_df(ratings_df[['user_id', 'product_id', 'rating']], reader)
                    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
                    algo.fit(data.build_full_trainset())
                    
                    # Tìm các sản phẩm chưa được đánh giá
                    rated_products = set(user_ratings['product_id'])
                    all_products = set(products_df['product_id'])
                    unrated_products = list(all_products - rated_products)
                    
                    # Dự đoán đánh giá cho các sản phẩm chưa đánh giá
                    predictions = []
                    for prod_id in unrated_products:
                        pred = algo.predict(user_id, prod_id)
                        predictions.append((prod_id, pred.est))
                    
                    # Sắp xếp theo đánh giá dự đoán
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    
                    st.write("#### 🏷️ Sản Phẩm Gợi Ý:")
                    displayed_count = 0
                    
                    for prod_id, pred_rating in predictions:
                        if displayed_count >= 5:
                            break
                            
                        product = products_df[products_df['product_id'] == prod_id].iloc[0]
                        
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                if product['image'] and str(product['image']).strip() != '':
                                    try:
                                        st.image(product['image'], width=150)
                                    except:
                                        pass
                            
                            with col2:
                                st.write(f"**{product['product_name']}**")
                                st.write(f"💰 Giá: {product['price']:,.0f}đ")
                                st.write(f"⭐ Đánh giá dự đoán: {pred_rating:.1f}/5")
                                st.write(f"📁 Danh mục: {product['category']} > {product['sub_category']}")
                                st.write(f"📖 Mô tả: {product['description'][:200]}...")
                            st.write("---")
                            
                        displayed_count += 1

elif choice == 'Gợi Ý Kết Hợp':
    st.subheader("🔄 Gợi Ý Kết Hợp")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.number_input("👤 Nhập ID người dùng:", min_value=1, value=1)
        
        with col2:
            search_term = st.text_input("🔎 Tìm kiếm sản phẩm:")
        
        if search_term and st.button("🔍 Xem Gợi Ý Kết Hợp", use_container_width=True):
            with st.spinner('🔄 Đang tìm kiếm sản phẩm...'):
                matches = products_df[products_df['product_name'].str.contains(search_term, case=False, na=False)]
                if not matches.empty:
                    selected_product = matches.iloc[0]
                    st.write("##### Sản phẩm được chọn:")
                    display_product(selected_product)
                    
                    with st.spinner('🔄 Đang phân tích nội dung và đánh giá...'):
                        # Tính điểm tương đồng nội dung
                        products_df['processed_description'] = products_df['description'].apply(
                            lambda x: preprocess_string(str(x).lower())
                        )
                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform(products_df['processed_description'])
                        
                        product_idx = products_df[products_df['product_id'] == selected_product['product_id']].index[0]
                        content_similarities = cosine_similarity(tfidf_matrix[product_idx], tfidf_matrix)
                        
                        # Kiểm tra xem người dùng đã có đánh giá nào chưa
                        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                        
                        if user_ratings.empty:
                            st.warning("⚠️ Người dùng chưa có đánh giá nào! Chỉ sử dụng gợi ý dựa trên nội dung.")
                            hybrid_similarities = content_similarities
                        else:
                            # Huấn luyện mô hình SVD
                            reader = Reader(rating_scale=(0, 5))
                            data = Dataset.load_from_df(ratings_df[['user_id', 'product_id', 'rating']], reader)
                            algo = SVD(n_factors=50, n_epochs=20, random_state=42)
                            algo.fit(data.build_full_trainset())
                            
                            # Dự đoán đánh giá cho tất cả sản phẩm
                            cf_scores = []
                            for idx, _ in enumerate(products_df['product_id']):
                                pred = algo.predict(user_id, products_df.iloc[idx]['product_id'])
                                cf_scores.append(pred.est)
                            
                            # Kết hợp điểm tương đồng nội dung và đánh giá hợp tác
                            hybrid_similarities = 0.6 * content_similarities[0] + 0.4 * np.array(cf_scores)
                        
                        # Tạo danh sách sản phẩm gợi ý
                        hybrid_scores = [(idx, score) for idx, score in enumerate(hybrid_similarities)]
                        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        st.write("#### 🏷️ Sản Phẩm Được Gợi Ý (Kết Hợp):")
                        displayed_count = 0
                        
                        for idx, score in hybrid_scores:
                            if displayed_count >= 5:
                                break
                                
                            product = products_df.iloc[idx]
                            if product['product_id'] != selected_product['product_id']:
                                with st.container():
                                    col1, col2 = st.columns([1, 3])
                                    
                                    with col1:
                                        if product['image'] and str(product['image']).strip() != '':
                                            try:
                                                st.image(product['image'], width=150)
                                            except:
                                                pass
                                    
                                    with col2:
                                        st.write(f"**{product['product_name']}**")
                                        st.write(f"💰 Giá: {product['price']:,.0f}đ")
                                        st.write(f"⭐ Đánh giá: {product['rating']:.1f}/5")
                                        st.write(f"📁 Danh mục: {product['category']} > {product['sub_category']}")
                                        st.write(f"📖 Mô tả: {product['description'][:200]}...")
                                        st.write(f"🎉 Độ phù hợp: {score:.2f}")
                                    st.write("---")
                                    displayed_count += 1
                else:
                    st.warning("⚠️ Không tìm thấy sản phẩm phù hợp.")