from pyngrok import ngrok
import os
import time

# Kill any previous tunnels
ngrok.kill()

# Start the Streamlit app
os.system("streamlit run /Users/sivakarthick/Downloads/ML2_miniprj/ML_code/Insightface/app.py &")
time.sleep(5)  # Wait for the app to boot

# Expose the app
public_url = ngrok.connect(8501)
print(f"✅ Streamlit app is live at: {public_url}")