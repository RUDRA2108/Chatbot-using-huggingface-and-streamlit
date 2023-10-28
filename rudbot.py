import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Load the DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

st.set_page_config(
    page_title="Your App's Title",
    page_icon="👾",  # Customize the favicon if desired
    layout="centered",     # Choose the layout: "centered", "wide", "narrow"
    menu_items={},          # Remove the Streamlit menu
    initial_sidebar_state="auto",  # or "expanded", "collapsed"
)

st.title("RudBot Chat👾👾")

# Initialize chat history as a list of dictionaries
chat_history = []

# Create a text input for user interactions
user_input = st.chat_input("")

if user_input:
    # Encode the user input and add the eos_token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    #chat_history.append({"speaker": "User", "message": user_input})

    bot_input_ids = new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Extract and display the response from the bot
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    #chat_history.append({"speaker": "RudBot", "message": bot_response})

# Display the chat history as chat bubbles
#for message in chat_history:
    message = st.chat_message("RUDBOT")
    message.write(f"👾👾 {bot_response}")

