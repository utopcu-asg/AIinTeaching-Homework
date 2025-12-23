"""Main entry point for the ASE 370C Homework multi-page app."""

import streamlit as st
import hw1
import hw2

def require_password() -> bool:
    """Prompt for the password stored in Streamlit secrets before showing the app."""
    secret_password = st.secrets.get("password")
    if not secret_password:
        st.error("Password not configured. Add `password` to your `.streamlit/secrets.toml`.")
        return False

    if st.session_state.get("password_correct"):
        return True

    def _password_entered() -> None:
        if st.session_state["app_password"] == secret_password:
            st.session_state["password_correct"] = True
            st.session_state.pop("app_password", None)
            st.session_state.pop("password_error", None)
        else:
            st.session_state["password_correct"] = False
            st.session_state["password_error"] = True

    st.text_input(
        "Enter password to access the homework",
        type="password",
        key="app_password",
        on_change=_password_entered,
    )
    if st.session_state.get("password_error"):
        st.error("Incorrect password. Please try again.")
    else:
        st.info("This collection of demos is password protected.")
    return False

def show_home():
    st.title("ASE 370C: Feedback Control Systems")
    st.subheader("Spring 2026 Homework")
    st.markdown("""
    Welcome to the interactive homework for ASE 370C. 
    Select a homework from the sidebar to begin.
    
    ### Available Exercises:
    - **Homework 1**: Feedback Control Systems - First-order systems, stability, and identification.
    - **Homework 2**: Linear Systems - Matrix exponentials, eigenvalues, and phase portraits.
    """)

def main():
    st.set_page_config(
        page_title="ASE 370C Homework",
        page_icon="📝",
        layout="wide"
    )

    if not require_password():
        st.stop()

    # Define pages
    pages = {
        "General": [
            st.Page(show_home, title="Home", icon="🏠"),
        ],
        "Homeworks": [
            st.Page(hw1.run_hw1, title="Homework 1: Feedback"),
            st.Page(hw2.run_hw2, title="Homework 2: Linear"),
        ]
    }

    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
