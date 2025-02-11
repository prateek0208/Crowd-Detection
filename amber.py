from flask import Flask, request, jsonify
from flask_cors import CORS
from twilio.rest import Client
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin communication

# Twilio configuration
TWILIO_ACCOUNT_SID = ""  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = ""    # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = ""      # Replace with your Twilio phone number
AUTHORITY_PHONE_NUMBER = ""   # Replace with the authority's phone number

# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/send_alert', methods=['POST'])
def send_alert():
    try:
        # Parse the JSON data from the request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        camera_name = data.get('cameraName', 'Unknown Camera')
        people_count = data.get('peopleCount', 0)
        crowd_density = data.get('status', 'unknown')

        # Prepare the SMS content
        message_content = (
            f"Amber Alert from {camera_name}: High crowd density detected! "
            f"Current density: {crowd_density} with {people_count} people. Immediate attention required."
        )

        # Send SMS using Twilio
        message = client.messages.create(
            body=message_content,
            from_=TWILIO_PHONE_NUMBER,
            to=AUTHORITY_PHONE_NUMBER
        )

        return jsonify({'message': 'Alert sent successfully', 'sid': message.sid}), 200

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
