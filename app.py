import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# ---------------- UI config ----------------
st.set_page_config(page_title="Hybrid Search Engine", page_icon="🔍", layout="wide")
st.markdown(
    """
    <style>
    /* App-wide styling */
    .main {padding: 1.5rem;}
    .result-card {
        background-color: #f7f9fc;
        border: 1px solid #e6ecf3;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .score-pill {
        display: inline-block;
        background: #eaf2ff;
        color: #1f4ed8;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 6px;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .subtext {
        color: #566b7f;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Data ----------------
documents = [
    # Animals
    "Cats are small domesticated animals that love to sleep.",
    "Dogs are loyal animals often called man's best friend.",
    "Parrots are colorful birds known for their ability to mimic human speech.",
    "Dolphins are highly intelligent marine mammals that communicate through clicks and whistles.",
    "Lions are known as the kings of the jungle and live in pride groups.",
    "Penguins are flightless birds adapted to life in cold Antarctic waters.",
    "Elephants are the largest land mammals and have remarkable memory capabilities.",
    "Giraffes have long necks that allow them to reach leaves in tall trees.",
    "Cheetahs are the fastest land animals, capable of incredible sprinting speeds.",
    "Owls are nocturnal birds with exceptional night vision abilities.",

    # Science & Space
    "The Earth revolves around the Sun once every year.",
    "Black holes are regions of space where gravity is so strong nothing can escape.",
    "DNA contains the genetic instructions for the development of all known living organisms.",
    "Quantum physics deals with behavior of matter and energy at the molecular level.",
    "The human brain contains approximately 86 billion neurons.",
    "Stars undergo nuclear fusion, converting hydrogen into helium.",
    "Climate change is affecting global weather patterns and ecosystems.",
    "The periodic table organizes chemical elements by their properties.",
    "Photosynthesis is how plants convert sunlight into chemical energy.",
    "The Milky Way galaxy contains billions of stars and planets.",

    # Technology & Programming
    "Python is a programming language used in data science.",
    "Artificial Intelligence is transforming various industries globally.",
    "Cloud computing enables remote data storage and processing.",
    "Machine learning algorithms improve through experience and data.",
    "JavaScript is essential for modern web development.",
    "Blockchain technology provides secure and decentralized record-keeping.",
    "5G networks promise faster mobile internet connectivity.",
    "Cybersecurity protects systems from digital attacks and threats.",
    "Virtual reality creates immersive digital experiences.",
    "Big data analytics helps organizations make data-driven decisions.",

    # Sports & Recreation
    "Football is a popular sport played worldwide.",
    "Basketball requires both physical skill and strategic thinking.",
    "Tennis can be played as singles or doubles matches.",
    "Swimming is an excellent full-body workout activity.",
    "Cricket is popular in many Commonwealth nations.",
    "Golf involves hitting a small ball into holes with various clubs.",
    "Volleyball can be played on indoor courts or beach sand.",
    "Marathon running tests athletes' endurance over 26.2 miles.",
    "Ice hockey is played on frozen rinks with skates and sticks.",
    "Rock climbing combines physical strength with problem-solving skills.",

    # Culture & Arts
    "Classical music often features complex orchestral arrangements.",
    "Photography captures moments through light and composition.",
    "Literature encompasses various forms of written artistic expression.",
    "Dance is a form of artistic physical expression through movement.",
    "Theater combines acting, music, and visual arts on stage.",
    "Architecture blends artistic design with practical engineering.",
    "Sculpture creates three-dimensional artistic forms.",
    "Cinema tells stories through moving images and sound.",
    "Poetry uses language to evoke emotions and imagery.",
    "Fashion design combines creativity with practical clothing construction."

    # Animals
    "Tigers are powerful big cats known for their striped coats.",
    "Kangaroos are marsupials native to Australia with strong hind legs.",
    "Pandas are bears native to China, famous for eating bamboo.",
    "Whales are the largest marine mammals, some species migrate thousands of miles.",
    "Eagles are birds of prey with excellent vision and hunting skills.",

    # Science & Space
    "Mars is often called the Red Planet due to its iron oxide surface.",
    "Satellites orbit Earth and provide communication and weather data.",
    "Vaccines help the immune system fight infectious diseases.",
    "The Hubble Space Telescope has provided deep images of the universe.",
    "Earthquakes are caused by sudden movements of tectonic plates.",

    # Technology & Programming
    "Linux is an open-source operating system widely used in servers.",
    "HTML is the standard markup language for creating web pages.",
    "Quantum computing uses qubits to perform complex calculations.",
    "Git is a version control system for tracking code changes.",
    "Robotics combines engineering and AI to build intelligent machines.",

    # Sports & Recreation
    "The Olympic Games bring together athletes from around the world.",
    "Baseball is a bat-and-ball game popular in the United States and Japan.",
    "Table tennis is a fast-paced indoor sport played with paddles.",
    "Badminton is a racquet sport played with a shuttlecock.",
    "Surfing involves riding waves on a board in the ocean.",

    # Culture & Arts
    "Painting is one of the oldest forms of artistic expression.",
    "Opera combines singing, music, and drama in theatrical performances.",
    "Calligraphy is the art of beautiful handwriting.",
    "Mythology consists of traditional stories explaining natural phenomena.",
    "Museums preserve and display cultural and historical artifacts."
]

links = [
    # Animals
    "https://en.wikipedia.org/wiki/Cat",
    "https://en.wikipedia.org/wiki/Dog",
    "https://en.wikipedia.org/wiki/Parrot",
    "https://en.wikipedia.org/wiki/Dolphin",
    "https://en.wikipedia.org/wiki/Lion",
    "https://en.wikipedia.org/wiki/Penguin",
    "https://en.wikipedia.org/wiki/Elephant",
    "https://en.wikipedia.org/wiki/Giraffe",
    "https://en.wikipedia.org/wiki/Cheetah",
    "https://en.wikipedia.org/wiki/Owl",

    # Science & Space
    "https://en.wikipedia.org/wiki/Earth",
    "https://en.wikipedia.org/wiki/Black_hole",
    "https://en.wikipedia.org/wiki/DNA",
    "https://en.wikipedia.org/wiki/Quantum_mechanics",
    "https://en.wikipedia.org/wiki/Human_brain",
    "https://en.wikipedia.org/wiki/Star",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Periodic_table",
    "https://en.wikipedia.org/wiki/Photosynthesis",
    "https://en.wikipedia.org/wiki/Milky_Way",

    # Technology & Programming
    "https://www.python.org/",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Cloud_computing",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/JavaScript",
    "https://en.wikipedia.org/wiki/Blockchain",
    "https://en.wikipedia.org/wiki/5G",
    "https://en.wikipedia.org/wiki/Computer_security",
    "https://en.wikipedia.org/wiki/Virtual_reality",
    "https://en.wikipedia.org/wiki/Big_data",

    # Sports & Recreation
    "https://en.wikipedia.org/wiki/Association_football",
    "https://en.wikipedia.org/wiki/Basketball",
    "https://en.wikipedia.org/wiki/Tennis",
    "https://en.wikipedia.org/wiki/Swimming",
    "https://en.wikipedia.org/wiki/Cricket",
    "https://en.wikipedia.org/wiki/Golf",
    "https://en.wikipedia.org/wiki/Volleyball",
    "https://en.wikipedia.org/wiki/Marathon",
    "https://en.wikipedia.org/wiki/Ice_hockey",
    "https://en.wikipedia.org/wiki/Rock_climbing",

    # Culture & Arts
    "https://en.wikipedia.org/wiki/Classical_music",
    "https://en.wikipedia.org/wiki/Photography",
    "https://en.wikipedia.org/wiki/Literature",
    "https://en.wikipedia.org/wiki/Dance",
    "https://en.wikipedia.org/wiki/Theatre",
    "https://en.wikipedia.org/wiki/Architecture",
    "https://en.wikipedia.org/wiki/Sculpture",
    "https://en.wikipedia.org/wiki/Film",
    "https://en.wikipedia.org/wiki/Poetry",
    "https://en.wikipedia.org/wiki/Fashion_design"

    # Animals
    "https://kids.nationalgeographic.com/animals/mammals/facts/tiger",
    "https://kids.nationalgeographic.com/animals/mammals/facts/kangaroo",
    "https://www.britannica.com/animal/giant-panda",
    "https://www.nationalgeographic.com/animals/mammals/facts/whales",
    "https://www.britannica.com/animal/eagle-bird",

    # Science & Space
    "https://mars.nasa.gov/all-about-mars/overview/",
    "https://www.nesdis.noaa.gov/what-satellite",
    "https://www.who.int/news-room/fact-sheets/detail/vaccines-and-immunization-what-is-vaccination",
    "https://www.nasa.gov/mission/hubble/",
    "https://www.usgs.gov/programs/earthquake-hazards/earthquakes",

    # Technology & Programming
    "https://www.linux.org/",
    "https://developer.mozilla.org/en-US/docs/Web/HTML",
    "https://www.ibm.com/quantum",
    "https://git-scm.com/",
    "https://www.britannica.com/technology/robotics",

    # Sports & Recreation
    "https://olympics.com/en/",
    "https://www.mlb.com/",
    "https://www.ittf.com/",
    "https://bwfbadminton.com/",
    "https://www.surfline.com/surfing-a-to-z",

    # Culture & Arts
    "https://www.britannica.com/art/painting",
    "https://www.metopera.org/",
    "https://www.britannica.com/art/calligraphy",
    "https://www.britannica.com/topic/myth",
    "https://www.louvre.fr/en"
]
# ---------------- Indexing ----------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# ---------------- Sidebar controls ----------------
st.sidebar.header("⚙️ Settings")
alpha = st.sidebar.slider("Fusion weight (Dense vs Sparse)", 0.0, 1.0, 0.5, help="0 = only sparse (TF-IDF), 1 = only dense (embeddings)")
top_k = st.sidebar.number_input("Top K results", min_value=1, max_value=10, value=5, step=1)
st.sidebar.markdown("Try different queries and tune fusion weight to see how rankings change.")

# ---------------- Header ----------------

st.markdown('<div class="header-title">🔍 Mini Hybrid Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Semantic + keyword retrieval with a clean, adaptive UI</div>', unsafe_allow_html=True)


# ---------------- Query input ----------------
query = st.text_input("Enter your query", placeholder="e.g., Which animal is loyal? or Tell me about planets...", help="Press Enter to search")

# ---------------- Retrieval + UI ----------------
if query:
    # Sparse retrieval
    query_vec = vectorizer.transform([query])
    sparse_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Dense retrieval
    query_embedding = model.encode(query, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

    # Fusion
    combined_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

    # Top-k results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    # Tabs for different views
    tab_list, tab_chart = st.tabs(["📋 List view", "📊 Score chart"])

    with tab_list:
     st.subheader(f"Top {top_k} results")
    for i, idx in enumerate(top_indices, start=1):
        st.markdown(
            f"""
            <div class="result-card">
                <div><b>{i}. <a href="{links[idx]}" target="_blank">{documents[idx]}</a></b></div>
                <div class="score-pill">Score: {combined_scores[idx]:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tab_chart:
        df = pd.DataFrame({
            "Document": [documents[i] for i in top_indices],
            "Score": [float(combined_scores[i]) for i in top_indices],
        })
        st.bar_chart(df.set_index("Document"))