import streamlit as st

# --- Global Page Configuration (Must be first Streamlit command) ---
st.set_page_config(
    page_title="ForgeMe - Master Interviews, Land Your Dream Job",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Minimalist Custom CSS for Branding and Spacing ---
st.markdown("""
<style>
    /* General body font and color */
    body {
        font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
        color: #1f2937; /* Dark gray for main text */
    }

    /* Streamlit's main content block padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 900px; /* Keep content readable */
    }

    /* Customizing Streamlit's header/top bar for a cleaner look */
    .stApp > header {
        background-color: #ffffff; /* White background */
        border-bottom: 1px solid #f0f2f6; /* Subtle border */
        position: sticky;
        top: 0;
        z-index: 1000;
        padding: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Light shadow */
    }

    /* Gradient for key text elements */
    .gradient-text {
        background: linear-gradient(to right, #8b5cf6, #ec4899); /* Purple to Pink */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent; /* Fallback */
        display: inline-block;
    }

    /* Main button styling (Streamlit's default primary/secondary are good, but we can enhance) */
    .stButton > button {
        border-radius: 0.75rem; /* More rounded */
        font-weight: 600; /* Semi-bold */
        padding: 0.8rem 1.8rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Subtle lift */
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15); /* More pronounced lift */
    }

    /* Specific styles for Primary CTA buttons */
    .stButton button[data-testid*="stButton-primary"] {
        background-color: #8b5cf6; /* Main brand purple */
        color: white;
        border: none;
    }
    .stButton button[data-testid*="stButton-primary"]:hover {
        background-color: #7c3aed; /* Darker purple on hover */
    }

    /* Styling for the secondary "Watch Demo" type buttons */
    .stButton button[data-testid*="stButton-secondary"] {
        background-color: #f3f4f6; /* Light gray */
        color: #374151; /* Darker text */
        border: 1px solid #e5e7eb;
    }
    .stButton button[data-testid*="stButton-secondary"]:hover {
        background-color: #e5e7eb; /* Slightly darker gray on hover */
    }

    /* Streamlit specific: Hide the Streamlit footer and menu button for a cleaner page */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;} /* We'll render our own simple header */

    /* Custom Header (for our simple nav) */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem;
        max-width: 900px;
        margin: 0 auto;
    }
    .custom-header .logo {
        font-size: 1.8rem;
        font-weight: 800;
        color: #8b5cf6;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .custom-header .nav-links button {
        background: none !important;
        border: none !important;
        color: #4b5563 !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: none !important;
        transform: none !important;
    }
    .custom-header .nav-links button:hover {
        color: #1f2937 !important;
        background-color: #f9fafb !important;
    }

    /* Section padding */
    .section-spacing {
        padding-top: 4rem;
        padding-bottom: 4rem;
    }

    /* Icon size for feature cards etc. */
    .icon-large {
        width: 3rem;
        height: 3rem;
        color: #8b5cf6; /* Primary color for icons */
    }

    /* Feature card simple styling */
    .feature-card {
        background-color: #ffffff;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        height: 100%; /* Ensure consistent height in columns */
        display: flex;
        flex-direction: column;
        align-items: center; /* Center content */
        text-align: center;
    }
    .feature-card h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #374151;
    }
    .feature-card p {
        font-size: 0.95rem;
        color: #6b7280;
    }

    /* Testimonial card simple styling */
    .testimonial-card {
        background-color: #f9fafb; /* Light gray background */
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .testimonial-card p {
        font-style: italic;
        color: #4b5563;
        margin-bottom: 1.5rem;
        flex-grow: 1;
    }
    .testimonial-author {
        font-weight: 600;
        color: #1f2937;
    }
    .testimonial-role {
        font-size: 0.9rem;
        color: #6b7280;
    }
    .testimonial-stars {
        color: #fbbf24; /* Yellow for stars */
        margin-bottom: 0.5rem;
    }

    /* Footer styling */
    .main-footer {
        background-color: #1f2937; /* Dark background */
        color: #d1d5db; /* Light gray text */
        padding: 3rem 1rem;
        text-align: center;
        font-size: 0.9rem;
    }
    .main-footer .logo {
        font-size: 1.5rem;
        font-weight: 800;
        color: #8b5cf6; /* Purple logo */
        margin-bottom: 1rem;
    }
    .main-footer a {
        color: #a78bfa; /* Lighter purple for links */
        text-decoration: none;
        margin: 0 0.5rem;
    }
    .main-footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# --- Navigation Bar ---
def render_navigation():
    """Renders a simple, clean navigation bar."""
    st.markdown("""
    <div class="custom-header">
        <div class="logo">ForgeMe</div>
        <div class="nav-links">
            <button onclick="document.getElementById('features-section').scrollIntoView({behavior: 'smooth'});">Features</button>
            <button onclick="document.getElementById('how-it-works-section').scrollIntoView({behavior: 'smooth'});">How It Works</button>
            <button onclick="document.getElementById('testimonials-section').scrollIntoView({behavior: 'smooth'});">Success Stories</button>
            <button onclick="document.getElementById('cta-section').scrollIntoView({behavior: 'smooth'});">Start Free Trial</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Hero Section ---
def render_hero_section():
    """Renders the main hero section with a strong value proposition."""
    st.container() # Use a container for better spacing control if needed
    st.markdown("<div class='section-spacing' style='text-align: center;'>", unsafe_allow_html=True)

    st.markdown("<h1 style='font-size: 3.5rem; font-weight: 800; line-height: 1.1; margin-bottom: 1rem;'>Crack Any Interview with <span class='gradient-text'>AI Confidence</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.3rem; color: #4b5563; max-width: 600px; margin: 0 auto 2rem auto;'>Stop guessing, start performing. ForgeMe gives you **hyper-personalized AI practice** and **actionable feedback** to ace your interviews and land your dream job.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        if st.button("🚀 Start Your Free Trial", key="hero_cta_trial", type="primary", use_container_width=True):
            st.success("Redirecting to your personalized training dashboard!") # Placeholder action
    with col2:
        if st.button("🎬 Watch a Quick Demo", key="hero_cta_demo", type="secondary", use_container_width=True):
            st.info("Loading demo video...") # Placeholder action

    st.markdown("<p style='font-size: 0.9rem; color: #6b7280; margin-top: 1.5rem;'>No credit card required. Cancel anytime within 14 days.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---") # Simple separator

# --- How It Works Section ---
def render_how_it_works_section():
    """Explains the simple 3-step process."""
    st.markdown("<div id='how-it-works-section' class='section-spacing'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 2.2rem; font-weight: 700; margin-bottom: 1.5rem;'>How ForgeMe Works in 3 Simple Steps</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #4b5563; max-width: 700px; margin: 0 auto 3rem auto;'>Our intelligent platform makes interview preparation straightforward and highly effective.</p>", unsafe_allow_html=True)

    cols = st.columns(3)
    steps = [
        {"title": "1. Set Your Target", "icon": "📄", "desc": "Upload your **resume** and the **job description**. Our AI instantly analyzes your profile and the role's needs."},
        {"title": "2. Practice Realistically", "icon": "🗣️", "desc": "Engage in **verbal mock interviews** with our AI. Experience scenario-based questions tailored just for you."},
        {"title": "3. Get Instant Feedback", "icon": "💡", "desc": "**Receive immediate, actionable insights** on your content, delivery, and areas for improvement. Track your growth!"},
    ]

    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <span style="font-size: 2.5rem; margin-bottom: 0.5rem;">{step['icon']}</span>
                <h3>{step['title']}</h3>
                <p>{step['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# --- Features Section ---
def render_features_section():
    """Highlights key benefits and features."""
    st.markdown("<div id='features-section' class='section-spacing'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 2.2rem; font-weight: 700; margin-bottom: 1.5rem;'>Why ForgeMe is Your Ultimate Interview Advantage</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #4b5563; max-width: 700px; margin: 0 auto 3rem auto;'>We go beyond basic prep. Our AI is designed to mimic real-world interview dynamics.</p>", unsafe_allow_html=True)

    features = [
        {"title": "AI-Driven Question Generation", "desc": "Our advanced NLP models craft **unique, role-specific questions**, ensuring you practice exactly what you'll face.", "icon": "🧠"},
        {"title": "Personalized, Actionable Feedback", "desc": "Get instant reports on your **content, clarity, confidence, and verbal cues**. Know exactly what to improve, not just vague suggestions.", "icon": "📊"},
        {"title": "Realistic Mock Scenarios", "desc": "Practice for behavioral, technical, case, and stress interviews. Our diverse library covers **every major industry and role**.", "icon": "🎭"},
        {"title": "Voice & Tone Analysis", "desc": "Understand how your voice impacts your message. Improve pacing, pitch, and overall delivery for a **confident impression**.", "icon": "🎙️"},
        {"title": "Progress Tracking Dashboard", "desc": "See your skills evolve. Our dashboard visualizes your strengths and weaknesses over time, pinpointing areas for **focused improvement**.", "icon": "📈"},
        {"title": "Expert Strategies & Tips", "desc": "Access a curated library of insights from **top recruiters and industry experts**, giving you an unfair edge.", "icon": "🌟"},
    ]

    # Create a 2x3 grid for features
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]: # Distribute cards into two columns
            st.markdown(f"""
            <div class="feature-card" style="margin-bottom: 1.5rem;">
                <span style="font-size: 2.5rem; margin-bottom: 0.5rem;">{feature['icon']}</span>
                <h3>{feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# --- Success Stories (Testimonials) Section ---
def render_testimonials_section():
    """Showcases user testimonials for social proof."""
    st.markdown("<div id='testimonials-section' class='section-spacing'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 2.2rem; font-weight: 700; margin-bottom: 1.5rem;'>Hear From Our Successful Users</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #4b5563; max-width: 700px; margin: 0 auto 3rem auto;'>Real people, real results. See how ForgeMe has helped others land their dream jobs.</p>", unsafe_allow_html=True)

    testimonials = [
        {"text": "ForgeMe's personalized feedback was a game-changer. I landed my dream Software Engineer role at Google thanks to their realistic simulations!", "author": "Sarah J.", "role": "Software Engineer, Google", "stars": 5},
        {"text": "The voice analysis helped me identify and fix my nervous habits. I felt incredibly confident in my Product Manager interview and got the offer!", "author": "Michael C.", "role": "Product Manager, Tech Startup", "stars": 5},
        {"text": "As a non-native speaker, practicing with ForgeMe improved my fluency and confidence immensely. I secured a Marketing Specialist position at a multinational.", "author": "Aisha R.", "role": "Marketing Specialist, Global Corp", "stars": 5},
    ]

    cols = st.columns(3) # Create 3 columns for testimonials
    for i, testimonial in enumerate(testimonials):
        with cols[i]:
            stars_html = "⭐" * testimonial['stars']
            st.markdown(f"""
            <div class="testimonial-card">
                <div class="testimonial-stars">{stars_html}</div>
                <p>"{testimonial['text']}"</p>
                <div>
                    <div class="testimonial-author">{testimonial['author']}</div>
                    <div class="testimonial-role">{testimonial['role']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# --- Call to Action (CTA) Section ---
def render_cta_section():
    """Final call to action to convert visitors."""
    st.markdown("<div id='cta-section' class='section-spacing' style='text-align: center; background-color: #f0f2f6; padding: 4rem 2rem; border-radius: 1rem;'>", unsafe_allow_html=True) # Light background for CTA

    st.markdown("<h2 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;'>Ready to Master Your Next Interview?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #4b5563; max-width: 600px; margin: 0 auto 2.5rem auto;'>Join thousands of successful job seekers. Start your journey to interview excellence today.</p>", unsafe_allow_html=True)

    if st.button("🌟 Start Your **Free** Trial Now", key="cta_main_trial", type="primary"):
        st.success("Welcome to ForgeMe! Your personalized training is being set up.")
        # In a real application, you'd redirect or initiate the signup flow here.

    st.markdown("<p style='font-size: 0.85rem; color: #6b7280; margin-top: 1.5rem;'>No credit card required. Instant access. Cancel anytime.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# --- Footer Section ---
def render_footer():
    """Simple footer for legal/contact info."""
    st.markdown("<div class='main-footer'>", unsafe_allow_html=True) # <-- ADDED HERE
    st.markdown("<div class='logo'>ForgeMe</div>", unsafe_allow_html=True) # <-- ADDED HERE
    st.markdown("<p style='margin-top: 0.5rem;'>Empowering your career with cutting-edge AI.</p>", unsafe_allow_html=True) # <-- ADDED HERE
    st.markdown("<p style='margin-top: 1rem;'>", unsafe_allow_html=True) # <-- ADDED HERE
    st.markdown("<a href='#'>Privacy Policy</a> | <a href='#'>Terms of Service</a> | <a href='#'>Contact Us</a>", unsafe_allow_html=True) # <-- ADDED HERE
    st.markdown("</p>", unsafe_allow_html=True) # <-- ADDED HERE
    st.markdown("<p style='margin-top: 1rem;'>&copy; 2025 ForgeMe. All rights reserved.</p>", unsafe_allow_html=True) # <-- ADDED HERE
    st.markdown("</div>", unsafe_allow_html=True) # <-- ADDED HERE


# --- Main App Execution Flow ---
if __name__ == "__main__":
    render_navigation()
    render_hero_section()
    render_how_it_works_section()
    render_features_section()
    render_testimonials_section() # Changed order for better flow: Features -> Proof -> CTA
    render_cta_section()
    render_footer()