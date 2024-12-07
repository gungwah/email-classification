import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import time
from datetime import datetime
import altair as alt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
from openai import OpenAI

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


class AutoResponseGenerator:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        # Initialize clients for each model type
        self.hf_client = InferenceClient(api_key=api_keys['huggingface'])
        if api_keys.get('openai'):
            self.openai_client = OpenAI(api_key=api_keys['openai'])
        if api_keys.get('gemini'):
            genai.configure(api_key=api_keys['gemini'])
            self.gemini_client = genai.GenerativeModel('gemini-pro')

    def create_response_prompt(self, ticket_text, category, tone="Professional"):
        return f"""Generate a {tone.lower()} IT support email response for the following ticket.

        Ticket Category: {category}
        Ticket Content: "{ticket_text}"

        Requirements:
        1. Start with a professional greeting
        2. Acknowledge the specific issue from the ticket
        3. Provide clear next steps based on the category
        4. Include estimated resolution timeframe
        5. Add contact information for follow-up
        6. End with a professional signature

        Additional Guidelines:
        - Use a {tone.lower()} tone
        - Be specific to the ticket's actual content
        - Include actionable information
        - Be concise but complete

        Email Response:"""

    def generate_response(self, ticket_text, category, model_name,
                         include_reference=True, 
                         include_disclaimer=True, 
                         tone="Professional"):
        try:
            prompt = self.create_response_prompt(ticket_text, category, tone)
            
            # Generate response based on selected model
            if model_name == "Llama-3.2-3B":
                messages = [{"role": "user", "content": prompt}]
                response = self.hf_client.chat.completions.create(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                email_response = response.choices[0].message.content.strip()
                
            elif model_name == "Mixtral-8x7B":
                messages = [{"role": "user", "content": prompt}]
                response = self.hf_client.chat.completions.create(
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                email_response = response.choices[0].message.content.strip()
                
            elif model_name == "GPT-4o":
                if not self.api_keys.get('openai'):
                    return "Error: OpenAI API key not configured"
                messages = [{"role": "user", "content": prompt}]
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o", 
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                email_response = response.choices[0].message.content.strip()
                
            elif model_name == "Gemini-Pro":
                if not self.api_keys.get('gemini'):
                    return "Error: Gemini API key not configured"
                response = self.gemini_client.generate_content(prompt)
                email_response = response.text.strip()
            
            else:
                return f"Error: Unknown model {model_name}"

            # Add reference number if requested
            if include_reference:
                ticket_id = datetime.now().strftime("TKT-%Y%m%d-%H%M%S")
                email_response = f"Reference: #{ticket_id}\n\n{email_response}"

            # Add disclaimer if requested
            if include_disclaimer:
                disclaimer = "\n\nThis is an auto-generated response. If you need immediate assistance or have additional questions, please reply to this email or contact our IT Support desk directly."
                email_response += disclaimer

            return email_response

        except Exception as e:
            return f"Error generating response: {str(e)}"

class LlamaClassifier(BaseTicketClassifier):
    def predict(self, ticket):
        messages = [{"role": "user", "content": self.create_prompt(ticket)}]
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Llama Error: {e}")
            return "Error"

class MixtralClassifier(BaseTicketClassifier):
    def predict(self, ticket):
        messages = [{"role": "user", "content": self.create_prompt(ticket)}]
        try:
            response = self.client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
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

def send_email(sender_email, sender_password, recipient_email, subject, body):
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Add body
        msg.attach(MIMEText(body, 'plain'))

        # Create SMTP session
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False



class GeminiClassifier(BaseTicketClassifier):
    def __init__(self, api_key):
        super().__init__(api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def predict(self, ticket):
        try:
            response = self.model.generate_content(self.create_prompt(ticket))
            return response.text.strip()
        except Exception as e:
            st.error(f"Gemini Error: {e}")
            return "Error"

class ChatGPTClassifier(BaseTicketClassifier):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)

    def predict(self, ticket):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": self.create_prompt(ticket)}
                ],
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"ChatGPT Error: {e}")
            return "Error"

def initialize_api_keys():
    """Initialize API keys from secrets"""
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'huggingface': st.secrets['HFkey'],
            'gemini': st.secrets.get('GEMINIkey', ''), 
            'openai': st.secrets.get('OPENAIkey', '')
        }

def get_available_models():
    """Get list of available models based on configured API keys"""
    available_models = ["Llama-3.2-3B", "Mixtral-8x7B"]
    
    if st.session_state.api_keys['gemini']:
        available_models.append("Gemini-Pro")
    if st.session_state.api_keys['openai']:
        available_models.append("GPT-4o")
    
    return available_models

def get_classifier(model_name, api_keys):
    """Get appropriate classifier based on model name"""
    if model_name == "Llama-3.2-3B":
        return LlamaClassifier(api_keys['huggingface'])
    elif model_name == "Mixtral-8x7B":
        return MixtralClassifier(api_keys['huggingface'])
    elif model_name == "Gemini-Pro":
        return GeminiClassifier(api_keys['gemini'])
    elif model_name == "GPT-4o":
        return ChatGPTClassifier(api_keys['openai'])
    else:
        raise ValueError(f"Unknown model: {model_name}")



def main():
    st.set_page_config(page_title="IT Ticket Classifier", layout="wide")
    
    initialize_api_keys()

   
   # Initialize session state for email settings
    if 'email_settings' not in st.session_state:
        st.session_state.email_settings = {
            'sender_email': st.secrets['email_acc'],
            'sender_password': st.secrets['email_pass'],
            'recipient_email': 'gung.wah@81media.co.id'
        }
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Display API key status
        st.subheader("API Key Status")
        for key, value in st.session_state.api_keys.items():
            status = "✅ Configured" if value else "❌ Not Configured"
            st.text(f"{key.title()}: {status}")
        
        theme_choice = st.radio("Select Theme:", ["Light", "Dark"], index=0)
        api_key = st.secrets['HFkey']
        st.markdown("---")
        st.header("Model Selection")
        available_models = get_available_models()
        selected_models = st.multiselect(
            "Choose models to compare:",
            available_models,
            default=[available_models[0]]
        )

    # Apply Theme
    theme_styles = {
        "Light": {"background": "#FFFFFF", "text": "#000000"},
        "Dark": {"background": "#1E1E1E", "text": "#FFFFFF"}
    }
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme_styles[theme_choice]["background"]};
            color: {theme_styles[theme_choice]["text"]};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.image("https://raw.githubusercontent.com/gungwah/email-classification/refs/heads/main/IT-TICKET-CLASSIFICATION.jpg", use_column_width=True)
    st.title("🎫 IT Support Ticket Classifier")
    st.markdown("""
    Welcome to the **IT Support Ticket Classifier**!
    This application leverages **LLMs** to classify IT support tickets into predefined categories.
    Analyze ticket data, compare model performance, and explore insights easily.
    """)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(
        ["🎫 Single Ticket", "📊 Batch Analysis", "📂 Dataset Overview"]
    )

    # Single Ticket Analysis
    with tab1:
        st.header("Single Ticket Classification and Response")
        input_text = st.text_area("Enter support ticket text 📝:", height=100)
        recipient_email = st.text_input("Recipient Email")
        
        # Store the response in session state
        if 'generated_response' not in st.session_state:
            st.session_state.generated_response = None
        
        # Response options
        col1, col2, col3 = st.columns(3)
        with col1:
            include_reference = st.checkbox("Include ticket reference", value=True)
        with col2:
            include_disclaimer = st.checkbox("Include disclaimer", value=True)
        with col3:
            tone = st.selectbox("Response tone", ["Professional", "Friendly", "Technical"])
        

              # Add model selection for response generation
        response_model = st.selectbox(
        "Select model for response generation",
        selected_models,
        index=0
        )


        # Generate Response button
        if st.button("Classify and Generate Response") and input_text:
            with st.spinner("Processing ticket..."):
                st.subheader("Classifications")
                predictions = {}
                for model in selected_models:
                    classifier = get_classifier(model, st.session_state.api_keys)
                    prediction = classifier.predict(input_text)
                    predictions[model] = prediction
                    st.success(f"{model} Classification: {prediction}")
                
       
                # Generate response
                response_generator = AutoResponseGenerator(st.session_state.api_keys)
                chosen_category = predictions[selected_models[0]]
                response = response_generator.generate_response(
                    input_text,
                    chosen_category,
                    response_model,  # Pass the selected model name
                    include_reference,
                    include_disclaimer,
                    tone
                )
                
                # Store response in session state
                st.session_state.generated_response = {
                    'text': response,
                    'category': chosen_category
                }
        
        # Display response and email buttons if response exists
        if st.session_state.generated_response:
            st.markdown("### Email Response Preview")
            st.code(st.session_state.generated_response['text'], language="text")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Send Email"):
                    if not all([
                        st.session_state.email_settings['sender_email'],
                        st.session_state.email_settings['sender_password'],
                        recipient_email
                    ]):
                        st.error("Please fill in all email settings and recipient email")
                    else:
                        subject = f"RE: IT Support Ticket - {st.session_state.generated_response['category']}"
                        if send_email(
                            st.session_state.email_settings['sender_email'],
                            st.session_state.email_settings['sender_password'],
                            recipient_email,
                            subject,
                            st.session_state.generated_response['text']
                        ):
                            st.success("Email sent successfully!")
            
            with col2:
                if st.button("Copy Response"):
                    st.code(st.session_state.generated_response['text'])
                    st.success("Response copied to clipboard!")
    
    # Batch Analysis
    with tab2:
        st.header("Batch Analysis")
        df = load_and_cache_data()
        
        num_samples = st.slider("Number of tickets to analyze:", 1, 20, 5)
        
        if st.button("Start Batch Analysis"):
            sample_df = df.head(num_samples)
            results = []

            progress_bar = st.progress(0)

            for idx, row in enumerate(sample_df.iterrows()):
                ticket_text = row[1]['Document']
                actual_category = row[1]['Topic_group']
                
                result = {
                    "Ticket": ticket_text[:100] + "...",
                    "Actual": actual_category
                }
                
                # Track classification time for each model separately
                for model in selected_models:
                    model_start = time.time()
                    classifier = get_classifier(model, st.session_state.api_keys)
                    prediction = classifier.predict(ticket_text)
                    model_end = time.time()
                    
                    result[model] = prediction
                    result[f"{model} Time (s)"] = round(model_end - model_start, 2)
                
                results.append(result)
                progress_bar.progress((idx + 1) / num_samples)
                time.sleep(2)

            results_df = pd.DataFrame(results)
            
            # Display model-specific timing metrics
            st.subheader("Model Processing Times")
            model_cols = st.columns(len(selected_models))
            for idx, model in enumerate(selected_models):
                with model_cols[idx]:
                    avg_time = results_df[f"{model} Time (s)"].mean()
                    max_time = results_df[f"{model} Time (s)"].max()
                    min_time = results_df[f"{model} Time (s)"].min()
                    st.metric(f"{model} Avg Time", f"{avg_time:.2f}s")
                    st.metric("Max", f"{max_time:.2f}s")
                    st.metric("Min", f"{min_time:.2f}s")
            
            # Display accuracies
            st.subheader("Model Accuracies")
            cols = st.columns(len(selected_models))  # Create columns equal to number of models
            for idx, model in enumerate(selected_models):
                with cols[idx]:  # Put each model accuracy in its own column
                    accuracy = (results_df[model] == results_df['Actual']).mean() * 100
                    st.metric(f"{model} Accuracy", f"{accuracy:.1f}%")

            st.subheader("Classification Results")
            st.dataframe(results_df)
            
            # Model timing comparison chart
            st.subheader("Model Processing Time Comparison")
            
            # Prepare data for the chart
            chart_data = pd.DataFrame({
                'Model': selected_models,
                'Processing Time (s)': [results_df[f"{model} Time (s)"].mean() for model in selected_models]
            })
            
            # Create the bar chart
            timing_chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Model:N', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Processing Time (s):Q', title='Average Processing Time (seconds)'),
                tooltip=['Model', 'Processing Time (s)']
            ).properties(
                height=300
            )
            
            st.altair_chart(timing_chart, use_container_width=True)
            
            # Save results option
            if st.button("Download Results"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_df.to_csv(f'classification_results_{timestamp}.csv', index=False)
                st.success("Results saved to CSV!")

    # Dataset Overview
    with tab3:
        st.header("Dataset Overview")
        df = load_and_cache_data()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📂 Total Tickets", len(df))
        with col2:
            st.metric("📊 Categories", df['Topic_group'].nunique())
        with col3:
            st.metric("📈 Most Common", df['Topic_group'].mode()[0])

        st.subheader("Category Distribution")
        plot_category_distribution(df)

        st.subheader("Sample Tickets")
        sample_size = st.slider("Number of sample tickets to display:", 1, 20, 5)
        st.dataframe(df.sample(sample_size)[['Document', 'Topic_group']])

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p>Developed by <strong>Team JAK</strong> | Powered by <strong>Streamlit</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
