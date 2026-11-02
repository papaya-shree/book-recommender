import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .book-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .book-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .book-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 8px;
        margin-bottom: 4px;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.5rem;
    }
    .book-author {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 6px;
        font-style: italic;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-rating {
        background-color: #FEF3C7;
        color: #D97706;
    }
    .badge-votes {
        background-color: #E0E7FF;
        color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to ensure HTTPS for image URLs
def clean_image_url(url):
    if not url or pd.isna(url):
        return "https://via.placeholder.com/150x220?text=No+Cover"
    url_str = str(url).strip()
    if url_str.startswith("http://"):
        url_str = "https://" + url_str[7:]
    return url_str

# Cache data loading
@st.cache_resource(show_spinner="Loading models & datasets...")
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'popular.pkl'), 'rb') as f:
        popular_df = pickle.load(f)
    with open(os.path.join(base_dir, 'pt.pkl'), 'rb') as f:
        pt = pickle.load(f)
    with open(os.path.join(base_dir, 'books.pkl'), 'rb') as f:
        books = pickle.load(f)
    with open(os.path.join(base_dir, 'similarity_scores.pkl'), 'rb') as f:
        similarity_scores = pickle.load(f)
    return popular_df, pt, books, similarity_scores

try:
    popular_df, pt, books, similarity_scores = load_data()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Recommendation logic
def get_recommendations(book_name, top_n=5):
    try:
        index = np.where(pt.index == book_name)[0][0]
    except IndexError:
        return []
    
    similar_indices = sorted(
        list(enumerate(similarity_scores[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    recommendations = []
    for i, score in similar_indices:
        matched_title = pt.index[i]
        temp_df = books[books['Book-Title'] == matched_title].drop_duplicates('Book-Title')
        
        if not temp_df.empty:
            author = temp_df['Book-Author'].values[0]
            image = clean_image_url(temp_df['Image-URL-M'].values[0])
        else:
            author = "Unknown Author"
            image = clean_image_url(None)
            
        recommendations.append({
            'title': matched_title,
            'author': author,
            'image': image,
            'similarity': round(float(score) * 100, 1)
        })
    return recommendations

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3389/3389081.png", width=64)
    st.title("Navigation")
    page = st.radio("Choose a section:", ["🔍 Recommend Books", "🌟 Top 50 Popular Books", "ℹ️ About System"])
    
    st.divider()
    st.markdown("**Dataset Stats**")
    st.write(f"• **Popular Collection**: {len(popular_df)} books")
    st.write(f"• **Collaborative Index**: {len(pt.index)} books")
    st.write(f"• **Unique Active Users**: {pt.shape[1]}")

# ----------------- PAGE 1: RECOMMEND BOOKS -----------------
if page == "🔍 Recommend Books":
    st.markdown('<div class="main-header">📚 Book Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select a book you love to discover similar reading suggestions powered by Collaborative Filtering.</div>', unsafe_allow_html=True)
    
    book_list = sorted(list(pt.index))
    
    col_input, col_num = st.columns([3, 1])
    with col_input:
        selected_book = st.selectbox(
            "Select or type a book title:",
            book_list,
            index=0 if "1984" not in book_list else book_list.index("1984"),
            help="Choose from 700+ famous books evaluated by readers."
        )
    with col_num:
        num_recommendations = st.slider("Number of results:", min_value=3, max_value=10, value=5)
        
    if st.button("✨ Get Recommendations", type="primary", use_container_width=False):
        results = get_recommendations(selected_book, top_n=num_recommendations)
        
        if not results:
            st.warning("No recommendations found for the selected book.")
        else:
            st.subheader(f"Readers who enjoyed '{selected_book}' also liked:")
            
            # Display books in columns (up to 5 per row)
            cols_per_row = 5
            for row_start in range(0, len(results), cols_per_row):
                row_items = results[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for idx, item in enumerate(row_items):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="book-card">
                            <img src="{item['image']}" style="width:100%; height:200px; object-fit:cover; border-radius:8px;" onerror="this.onerror=null;this.src='https://via.placeholder.com/150x220?text=No+Cover';" />
                            <div class="book-title" title="{item['title']}">{item['title']}</div>
                            <div class="book-author">by {item['author']}</div>
                            <div>
                                <span class="badge badge-votes">{item['similarity']}% match</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                st.write("") # spacing between rows

# ----------------- PAGE 2: TOP 50 POPULAR BOOKS -----------------
elif page == "🌟 Top 50 Popular Books":
    st.markdown('<div class="main-header">🌟 Top 50 Most Popular Books</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Books with highest average ratings having at least 250 verified user reviews.</div>', unsafe_allow_html=True)
    
    search_query = st.text_input("Filter top books by title or author:", placeholder="Type to filter...")
    
    display_df = popular_df.copy()
    if search_query:
        display_df = display_df[
            display_df['Book-Title'].str.contains(search_query, case=False, na=False) |
            display_df['Book-Author'].str.contains(search_query, case=False, na=False)
        ]
    
    cols_per_row = 5
    items = display_df.to_dict('records')
    
    if not items:
        st.info("No books match your search.")
    else:
        for row_start in range(0, len(items), cols_per_row):
            row_items = items[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for idx, item in enumerate(row_items):
                with cols[idx]:
                    img_url = clean_image_url(item.get('Image-URL-M', ''))
                    avg_rating = round(float(item.get('avg_rating', 0.0)), 2)
                    num_ratings = int(item.get('num_ratings', 0))
                    title = item.get('Book-Title', 'Unknown')
                    author = item.get('Book-Author', 'Unknown')
                    
                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{img_url}" style="width:100%; height:200px; object-fit:cover; border-radius:8px;" onerror="this.onerror=null;this.src='https://via.placeholder.com/150x220?text=No+Cover';" />
                        <div class="book-title" title="{title}">{title}</div>
                        <div class="book-author">by {author}</div>
                        <div>
                            <span class="badge badge-rating">⭐ {avg_rating} / 10</span>
                            <span class="badge badge-votes">{num_ratings} votes</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.write("")

# ----------------- PAGE 3: ABOUT -----------------
else:
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(r"""
    This project is an end-to-end **Book Recommender System** built with Machine Learning and Python, using the Kaggle Book-Crossing dataset.
    
    ### 🔬 How it Works
    
    1. **Popularity-Based Filtering**:
       - Displays the highest-rated books across the platform.
       - A strict threshold is applied: only books that received at least **250 reviews** are considered, avoiding skew from books with a single 10-star rating.
    
    2. **Collaborative Filtering (Item-Based)**:
       - Uses reader behavior to discover books that share similar audiences.
       - Filters out inactive readers: only users who rated > 200 books are selected (`padhe_likhe_users`).
       - Filters out niche titles: only books with >= 50 ratings are included.
       - Constructs a user-item matrix ($706$ books $\times$ $810$ users) and calculates **Cosine Similarity** between all book vectors.
    
    ### 🛠️ Tech Stack
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Algorithm**: Scikit-Learn (Cosine Similarity)
    - **Deployment Target**: Streamlit Community Cloud
    """)
