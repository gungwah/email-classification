import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import time
from datetime import datetime
import altair as alt

class BaseTicketClassifier:
    def __init__(self, api_key):
        self.client = InferenceClient(api_key=api_key)
        self.categories = [
            "Hardware", "HR Support", "Access", "Miscellaneous",
            "Storage", "Purchase", "Internal Project", "Administrative rights"
        ]
    
    def create_prompt(self, ticket_text):
        return f"""You are an expert IT support ticket classifier. Your task is to classify the following support ticket into exactly one of these categories:

        Categories and their definitions:
        - Hardware: Issues with physical devices, computers, printers, or equipment
        - HR Support: Human resources related requests, employee matters
        - Access: Login issues, permissions, account access, passwords
        - Miscellaneous: General inquiries or requests that don't fit other categories
        - Storage: Data storage, disk space, file storage related issues
        - Purchase: Procurement requests, buying equipment or software
        - Internal Project: Project-related tasks and updates
        - Administrative rights: Requests for admin privileges or system permissions

        Support Ticket: "{ticket_text}"

        Instructions:
        1. Read the ticket carefully
        2. Match the main issue with the category definitions above
        3. Respond with only the category name, nothing else
        4. If a ticket could fit multiple categories, choose the most specific one
        5. Focus on the primary issue, not secondary mentions

        Category:"""

class LlamaClassifier(BaseTicketClassifier):
    def predict(self, ticket):
        messages = [{"role": "user", "content": self.create_prompt(ticket)}]
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",  # Updated model
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Llama Error: {e}")
            return "Error"

class MixtralClassifier(BaseTicketClassifier):  # Renamed from MistralClassifier
    def predict(self, ticket):
        messages = [{"role": "user", "content": self.create_prompt(ticket)}]
        try:
            response = self.client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Updated model
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Mixtral Error: {e}")
            return "Error"

def load_and_cache_data():
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(
            "https://raw.githubusercontent.com/gungwah/email-ticket-classification-auto-response/refs/heads/main/all_tickets_processed_improved_v3.csv"
        )
    return st.session_state.df

def plot_category_distribution(df):
    category_counts = df['Topic_group'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    chart = alt.Chart(category_counts).mark_bar().encode(
        x='Category',
        y='Count',
        color=alt.value('#1f77b4')
    ).properties(
        title='Distribution of Ticket Categories'
    )
    
    st.altair_chart(chart, use_container_width=True)

def main():
    # Set up the page configuration
    st.set_page_config(page_title="IT Ticket Classifier", page_icon="🎫", layout="wide")
    
    # Add a banner image (replace the URL with your logo or image path)
    st.image("https://via.placeholder.com/800x150?text=Support+Ticket+Classifier", use_column_width=True)
    
    # Add a cleaner title and description
    st.title("🎫 IT Support Ticket Classifier")
    st.markdown("""
    Welcome to the **IT Support Ticket Classifier**!  
    This tool leverages advanced machine learning models to classify IT support tickets, improving efficiency and productivity.  
    Use the tabs to classify a single ticket, analyze a batch, or explore the dataset.
    """)

    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.secrets['HFkey']
        st.markdown("---")
        st.header("Model Selection")
        selected_models = st.multiselect(
            "Choose models to compare:",
            ["Llama-3.2-3B", "Mixtral-8x7B"],  # Updated model names
            default=["Llama-3.2-3B"]
        )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Single Ticket", "Batch Analysis", "Dataset Overview"])
    
    # Single Ticket Analysis
    with tab1:
        st.header("Single Ticket Classification")
        input_text = st.text_area("Enter support ticket text:", height=100)
        
        if st.button("Classify Ticket") and api_key and input_text:
            with st.spinner("Classifying ticket..."):
                for model in selected_models:
                    st.subheader(f"{model} Classification")
                    if model == "Llama-3.2-3B":
                        classifier = LlamaClassifier(api_key)
                    else:
                        classifier = MixtralClassifier(api_key)
                    
                    prediction = classifier.predict(input_text)
                    st.success(f"Predicted Category: {prediction}")
                    st.markdown("---")
    
    # Batch Analysis
    with tab2:
        st.header("Batch Analysis")
        df = load_and_cache_data()
        
        num_samples = st.slider("Number of tickets to analyze:", 1, 20, 5)
        
        if st.button("Start Batch Analysis") and api_key:
            sample_df = df.head(num_samples)
            results = []
            
            progress_text = "Analyzing tickets..."
            progress_bar = st.progress(0)
            
            for idx, row in enumerate(sample_df.iterrows()):
                ticket_text = row[1]['Document']
                actual_category = row[1]['Topic_group']
                
                result = {
                    "Ticket": ticket_text[:100] + "...",
                    "Actual": actual_category
                }
                
                for model in selected_models:
                    if model == "Llama-3.2-3B":
                        classifier = LlamaClassifier(api_key)
                    else:
                        classifier = MixtralClassifier(api_key)
                    
                    result[model] = classifier.predict(ticket_text)
                
                results.append(result)
                progress_bar.progress((idx + 1) / num_samples)
                time.sleep(2)  # Rate limiting
            
            results_df = pd.DataFrame(results)
            
            # Display accuracies
            st.subheader("Model Accuracies")
            cols = st.columns(len(selected_models))
            for idx, model in enumerate(selected_models):
                with cols[idx]:
                    accuracy = (results_df[model] == results_df['Actual']).mean() * 100
                    st.metric(f"{model} Accuracy", f"{accuracy:.1f}%")
            
            # Display results table
            st.subheader("Classification Results")
            st.dataframe(results_df)
            
            # Save results option
            if st.button("Download Results"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_df.to_csv(f'classification_results_{timestamp}.csv', index=False)
                st.success("Results saved to CSV!")
    
    # Dataset Overview
    with tab3:
        st.header("Dataset Overview")
        df = load_and_cache_data()
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tickets", len(df))
        with col2:
            st.metric("Categories", df['Topic_group'].nunique())
        with col3:
            st.metric("Most Common Category", df['Topic_group'].mode()[0])
        
        # Show category distribution
        st.subheader("Category Distribution")
        plot_category_distribution(df)
        
        # Display sample tickets
        st.subheader("Sample Tickets")
        sample_size = st.slider("Number of sample tickets to display:", 1, 10, 5)
        st.dataframe(df.sample(sample_size)[['Document', 'Topic_group']])

if __name__ == "__main__":
    main()
